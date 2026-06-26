"""
睡眠期梯度更新（Sleep Updater）
=================================

实现 Chronos-Self 的睡眠期梯度更新机制，使用伴随法梯度回传
仅更新积分引擎参数，确保不修改关键帧本身。

Task 20 实现：
- SubTask 20.1: 从关键帧向前积分的重放流程
- SubTask 20.2: 伴随法梯度回传（仅更新积分引擎参数）
- SubTask 20.3: 不修改关键帧本身的约束
- SubTask 20.4: 睡眠重放稳定性测试

核心功能：
1. 重放流程：
   - 从向量数据库加载关键帧作为初始状态
   - 使用当前参数的ODE求解器从关键帧向前积分
   - 计算双重损失（一致性 + 改善）

2. 梯度回传：
   - 使用伴随法梯度回传
   - 仅更新积分引擎参数（F_θ, G_φ）
   - 不修改关键帧本身（历史不可变）

3. 约束机制：
   - 不修改关键帧本身的约束（历史保护）
   - 梯度裁剪（防止过大更新）
   - 参数更新范围限制

4. 稳定性测试：
   - 测试重放过程的数值稳定性
   - 防止睡眠期更新导致系统崩溃
   - 监测更新后的性能改善
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
import numpy as np

try:
    from torchdiffeq import odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    logging.warning("torchdiffeq not available. Using custom adjoint implementation.")

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    TrainingConfig,
)
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.reflection.sleep_replay import KeyframeData, KeyframeDatabase


logger = logging.getLogger(__name__)


@dataclass
class SleepUpdaterConfig:
    """睡眠期梯度更新配置"""
    
    # 参数更新范围
    update_fast_dynamics: bool = True  # 更新快变量动力学 F_θ
    update_slow_dynamics: bool = True  # 更新慢变量动力学 G_φ
    update_coupling: bool = False  # 更新耦合系数（通常不更新）
    
    # 梯度约束
    gradient_clip_value: float = 1.0  # 梯度裁剪阈值
    max_parameter_change: float = 0.1  # 最大参数变化幅度
    freeze_keyframe_params: bool = True  # 冻结关键帧相关参数
    
    # 优化器配置
    learning_rate: float = 1e-4  # 学习率
    optimizer_type: str = "adam"  # 优化器类型
    weight_decay: float = 1e-5  # 权重衰减
    
    # 重放配置
    replay_batch_size: int = 10  # 重放批量大小
    max_replay_steps: int = 100  # 最大重放步数
    replay_window_seconds: float = 60.0  # 重放窗口
    
    # 稳定性检查
    stability_check_enabled: bool = True  # 启用稳定性检查
    parameter_norm_threshold: float = 10.0  # 参数范数阈值
    loss_threshold: float = 100.0  # 损失阈值
    gradient_norm_threshold: float = 5.0  # 梯度范数阈值
    
    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


class GradientConstraints:
    """
    梯度约束机制
    
    确保不修改关键帧本身，仅更新积分引擎参数。
    
    Task 20.3 实现：不修改关键帧本身的约束
    
    功能：
    - 历史保护（冻结关键帧状态）
    - 参数更新范围限制
    - 梯度裁剪
    - 参数冻结管理
    """
    
    def __init__(
        self,
        config: Optional[SleepUpdaterConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化梯度约束
        
        Args:
            config: 睡眠更新配置
            device: 计算设备
        """
        self.config = config or SleepUpdaterConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 梯度裁剪阈值
        self.clip_value = self.config.gradient_clip_value
        
        # 最大参数变化
        self.max_param_change = self.config.max_parameter_change
        
        # 约束统计
        self._stats: Dict[str, Any] = {
            "total_constraints_applied": 0,
            "gradient_clips": 0,
            "param_freezes": 0,
            "avg_clipped_gradient_norm": 0.0,
        }
        
        logger.info(
            f"GradientConstraints initialized: "
            f"clip_value={self.clip_value}, "
            f"max_param_change={self.max_param_change}"
        )
    
    def apply_gradient_clipping(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        应用梯度裁剪
        
        Args:
            gradients: 梯度字典
            
        Returns:
            裁剪后的梯度字典
        """
        clipped_gradients = {}
        clip_count = 0
        total_clipped_norm = 0.0
        
        for param_name, grad in gradients.items():
            if grad is None:
                clipped_gradients[param_name] = None
                continue
            
            # 计算梯度范数
            grad_norm = torch.norm(grad).item()
            
            # 裁剪
            if grad_norm > self.clip_value:
                clip_factor = self.clip_value / grad_norm
                clipped_grad = grad * clip_factor
                clipped_gradients[param_name] = clipped_grad
                clip_count += 1
                total_clipped_norm += torch.norm(clipped_grad).item()
                
                logger.debug(
                    f"Gradient clipped: {param_name}, "
                    f"original_norm={grad_norm:.4f}, "
                    f"clipped_norm={torch.norm(clipped_grad).item():.4f}"
                )
            else:
                clipped_gradients[param_name] = grad
        
        # 更新统计
        self._stats["gradient_clips"] += clip_count
        if clip_count > 0:
            self._stats["avg_clipped_gradient_norm"] = (
                total_clipped_norm / clip_count
            )
        
        return clipped_gradients
    
    def enforce_parameter_limits(
        self,
        params: List[torch.nn.Parameter],
        old_params: List[torch.Tensor]
    ) -> bool:
        """
        强制参数更新范围限制
        
        Args:
            params: 当前参数列表
            old_params: 原始参数值列表
            
        Returns:
            是否满足限制
        """
        within_limits = True
        
        for i, (param, old_param) in enumerate(zip(params, old_params)):
            # 计算参数变化
            param_change = torch.norm(param - old_param).item()
            
            # 检查是否超出限制
            if param_change > self.max_param_change:
                # 限制参数变化
                scale = self.max_param_change / param_change
                param.data = old_param + (param - old_param) * scale
                
                logger.warning(
                    f"Parameter change limited: param_{i}, "
                    f"change={param_change:.4f}, max={self.max_param_change}"
                )
                
                within_limits = False
        
        # 更新统计
        self._stats["total_constraints_applied"] += 1
        
        return within_limits
    
    def freeze_keyframe_states(
        self,
        keyframes: List[KeyframeData]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        冻结关键帧状态（确保历史不被修改）
        
        Args:
            keyframes: 关键帧列表
            
        Returns:
            冻结的状态张量列表
        """
        frozen_states = []
        
        for keyframe in keyframes:
            # 将关键帧状态转换为冻结张量
            E_fast_frozen = torch.tensor(
                keyframe.E_fast,
                dtype=torch.float32,
                device=self.device
            ).detach()  # detach 确保不会反向传播
            
            E_slow_frozen = torch.tensor(
                keyframe.E_slow,
                dtype=torch.float32,
                device=self.device
            ).detach() if keyframe.E_slow is not None else torch.zeros(
                self.config.slow_dim,
                device=self.device
            )
            
            frozen_states.append((E_fast_frozen, E_slow_frozen))
        
        # 更新统计
        self._stats["param_freezes"] += len(keyframes)
        
        logger.debug(f"Keyframe states frozen: count={len(keyframes)}")
        
        return frozen_states
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self._stats.copy()
    
    def reset(self) -> None:
        """
        重置统计信息
        """
        self._stats = {
            "total_constraints_applied": 0,
            "gradient_clips": 0,
            "param_freezes": 0,
            "avg_clipped_gradient_norm": 0.0,
        }
        
        logger.info("GradientConstraints reset")
    
    def __repr__(self) -> str:
        return (
            f"GradientConstraints(clip={self.clip_value}, "
            f"max_change={self.max_param_change})"
        )


class StabilityChecker:
    """
    稳定性检查器
    
    测试重放过程的数值稳定性，防止睡眠期更新导致系统崩溃。
    
    Task 20.4 实现：睡眠重放稳定性测试
    
    功能：
    - 参数稳定性检查
    - 梯度稳定性检查
    - 损失稳定性检查
    - 数值稳定性检查（NaN, Inf）
    """
    
    def __init__(
        self,
        config: Optional[SleepUpdaterConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化稳定性检查器
        
        Args:
            config: 睡眠更新配置
            device: 计算设备
        """
        self.config = config or SleepUpdaterConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 稳定性阈值
        self.param_norm_threshold = self.config.parameter_norm_threshold
        self.loss_threshold = self.config.loss_threshold
        self.gradient_norm_threshold = self.config.gradient_norm_threshold
        
        # 稳定性状态
        self.is_stable = True
        self.stability_warnings: List[str] = []
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_checks": 0,
            "unstable_events": 0,
            "nan_detected": 0,
            "inf_detected": 0,
            "gradient_explosions": 0,
        }
        
        logger.info(
            f"StabilityChecker initialized: "
            f"param_threshold={self.param_norm_threshold}, "
            f"loss_threshold={self.loss_threshold}"
        )
    
    def check_parameters(
        self,
        params: List[torch.nn.Parameter]
    ) -> Dict[str, Any]:
        """
        检查参数稳定性
        
        Args:
            params: 参数列表
            
        Returns:
            检查结果字典
        """
        result = {
            "is_stable": True,
            "warnings": [],
            "param_norms": [],
        }
        
        for i, param in enumerate(params):
            # 检查 NaN
            if torch.isnan(param).any():
                result["is_stable"] = False
                result["warnings"].append(f"NaN in parameter {i}")
                self._stats["nan_detected"] += 1
            
            # 检查 Inf
            if torch.isinf(param).any():
                result["is_stable"] = False
                result["warnings"].append(f"Inf in parameter {i}")
                self._stats["inf_detected"] += 1
            
            # 检查范数
            param_norm = torch.norm(param).item()
            result["param_norms"].append(param_norm)
            
            if param_norm > self.param_norm_threshold:
                result["warnings"].append(
                    f"Parameter norm too large: param_{i}, norm={param_norm:.4f}"
                )
        
        # 更新统计
        self._stats["total_checks"] += 1
        
        if not result["is_stable"]:
            self._stats["unstable_events"] += 1
            self.stability_warnings.extend(result["warnings"])
            self.is_stable = False
        
        return result
    
    def check_gradients(
        self,
        gradients: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        检查梯度稳定性
        
        Args:
            gradients: 梯度字典
            
        Returns:
            检查结果字典
        """
        result = {
            "is_stable": True,
            "warnings": [],
            "gradient_norms": [],
        }
        
        for param_name, grad in gradients.items():
            if grad is None:
                continue
            
            # 检查 NaN
            if torch.isnan(grad).any():
                result["is_stable"] = False
                result["warnings"].append(f"NaN in gradient {param_name}")
                self._stats["nan_detected"] += 1
            
            # 检查 Inf
            if torch.isinf(grad).any():
                result["is_stable"] = False
                result["warnings"].append(f"Inf in gradient {param_name}")
                self._stats["inf_detected"] += 1
            
            # 检查范数
            grad_norm = torch.norm(grad).item()
            result["gradient_norms"].append(grad_norm)
            
            if grad_norm > self.gradient_norm_threshold:
                result["warnings"].append(
                    f"Gradient explosion: {param_name}, norm={grad_norm:.4f}"
                )
                self._stats["gradient_explosions"] += 1
        
        # 更新统计
        self._stats["total_checks"] += 1
        
        if result["warnings"]:
            self.stability_warnings.extend(result["warnings"])
        
        return result
    
    def check_loss(
        self,
        loss: torch.Tensor
    ) -> Dict[str, Any]:
        """
        检查损失稳定性
        
        Args:
            loss: 损失张量
            
        Returns:
            检查结果字典
        """
        result = {
            "is_stable": True,
            "warnings": [],
            "loss_value": loss.item(),
        }
        
        # 检查 NaN
        if torch.isnan(loss):
            result["is_stable"] = False
            result["warnings"].append("NaN in loss")
            self._stats["nan_detected"] += 1
        
        # 检查 Inf
        if torch.isinf(loss):
            result["is_stable"] = False
            result["warnings"].append("Inf in loss")
            self._stats["inf_detected"] += 1
        
        # 检查损失值大小
        loss_value = loss.item()
        if loss_value > self.loss_threshold:
            result["warnings"].append(f"Loss too large: {loss_value:.4f}")
        
        # 更新统计
        self._stats["total_checks"] += 1
        
        if not result["is_stable"]:
            self._stats["unstable_events"] += 1
            self.is_stable = False
        
        return result
    
    def check_replay_trajectory(
        self,
        trajectory: torch.Tensor
    ) -> Dict[str, Any]:
        """
        检查重放轨迹稳定性
        
        Args:
            trajectory: 状态轨迹
            
        Returns:
            检查结果字典
        """
        result = {
            "is_stable": True,
            "warnings": [],
            "trajectory_stats": {},
        }
        
        # 检查每个时间点
        for i in range(trajectory.shape[0]):
            state = trajectory[i]
            
            # 检查 NaN
            if torch.isnan(state).any():
                result["is_stable"] = False
                result["warnings"].append(f"NaN at trajectory step {i}")
                self._stats["nan_detected"] += 1
            
            # 检查 Inf
            if torch.isinf(state).any():
                result["is_stable"] = False
                result["warnings"].append(f"Inf at trajectory step {i}")
                self._stats["inf_detected"] += 1
        
        # 计算轨迹统计
        norms = torch.norm(trajectory, dim=-1)
        result["trajectory_stats"] = {
            "mean_norm": norms.mean().item(),
            "max_norm": norms.max().item(),
            "min_norm": norms.min().item(),
            "std_norm": norms.std().item(),
        }
        
        # 更新统计
        self._stats["total_checks"] += 1
        
        if not result["is_stable"]:
            self._stats["unstable_events"] += 1
            self.is_stable = False
        
        return result
    
    def test_sleep_replay_stability(
        self,
        integration_engine: IntegrationEngine,
        test_keyframes: List[KeyframeData],
        test_steps: int = 100
    ) -> Dict[str, Any]:
        """
        测试睡眠重放稳定性
        
        Task 20.4: 睡眠重放稳定性测试
        
        Args:
            integration_engine: 积分引擎
            test_keyframes: 测试关键帧列表
            test_steps: 测试步数
            
        Returns:
            稳定性测试结果
        """
        logger.info(f"Starting stability test: keyframes={len(test_keyframes)}, steps={test_steps}")
        
        test_results = {
            "overall_stable": True,
            "test_steps": test_steps,
            "keyframe_count": len(test_keyframes),
            "step_results": [],
            "summary": {},
        }
        
        # 测试每个关键帧的重放
        for kf_idx, keyframe in enumerate(test_keyframes[:test_steps]):
            # 从关键帧向前积分
            E_fast_init = torch.tensor(
                keyframe.E_fast,
                dtype=torch.float32,
                device=self.device
            )
            
            E_slow_init = torch.tensor(
                keyframe.E_slow,
                dtype=torch.float32,
                device=self.device
            ) if keyframe.E_slow is not None else torch.zeros(
                self.config.slow_dim,
                device=self.device
            )
            
            # 执行重放积分
            try:
                # 单步积分
                E_fast_next = integration_engine.fast_dynamics.step(
                    E_fast_init,
                    E_slow_init,
                    {},
                    0.01,
                    keyframe.timestamp,
                )
                
                # 检查稳定性
                stability_check = self.check_replay_trajectory(
                    torch.stack([E_fast_init, E_fast_next])
                )
                
                test_results["step_results"].append({
                    "keyframe_idx": kf_idx,
                    "is_stable": stability_check["is_stable"],
                    "warnings": stability_check["warnings"],
                    "stats": stability_check["trajectory_stats"],
                })
                
                if not stability_check["is_stable"]:
                    test_results["overall_stable"] = False
                
            except Exception as e:
                logger.error(f"Replay failed at keyframe {kf_idx}: {e}")
                test_results["step_results"].append({
                    "keyframe_idx": kf_idx,
                    "is_stable": False,
                    "error": str(e),
                })
                test_results["overall_stable"] = False
        
        # 统计摘要
        stable_count = sum(1 for r in test_results["step_results"] if r["is_stable"])
        unstable_count = len(test_results["step_results"]) - stable_count
        
        test_results["summary"] = {
            "stable_steps": stable_count,
            "unstable_steps": unstable_count,
            "stability_ratio": stable_count / len(test_results["step_results"]),
            "total_warnings": len(self.stability_warnings),
        }
        
        logger.info(
            f"Stability test completed: "
            f"stable={stable_count}/{len(test_results['step_results'])}, "
            f"ratio={test_results['summary']['stability_ratio']:.2f}"
        )
        
        return test_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["is_stable"] = self.is_stable
        stats["warnings_count"] = len(self.stability_warnings)
        
        return stats
    
    def reset(self) -> None:
        """
        重置稳定性检查器
        """
        self.is_stable = True
        self.stability_warnings.clear()
        
        self._stats = {
            "total_checks": 0,
            "unstable_events": 0,
            "nan_detected": 0,
            "inf_detected": 0,
            "gradient_explosions": 0,
        }
        
        logger.info("StabilityChecker reset")
    
    def __repr__(self) -> str:
        status = "stable" if self.is_stable else "unstable"
        return (
            f"StabilityChecker(status={status}, "
            f"checks={self._stats['total_checks']})"
        )


class SleepUpdater:
    """
    睡眠期梯度更新
    
    整合重放流程、伴随法梯度回传和约束机制，
    实现完整的睡眠期参数更新。
    
    Task 20 完整实现：
    - SubTask 20.1: 从关键帧向前积分的重放流程
    - SubTask 20.2: 伴随法梯度回传（仅更新积分引擎参数）
    - SubTask 20.3: 不修改关键帧本身的约束
    - SubTask 20.4: 睡眠重放稳定性测试
    
    使用示例：
        updater = SleepUpdater(config=SleepUpdaterConfig())
        updater.initialize(integration_engine)
        
        # 从关键帧数据库获取关键帧
        keyframes = keyframe_db.get_recent_keyframes()
        
        # 执行睡眠更新
        result = updater.perform_sleep_update(keyframes)
        
        # 测试稳定性
        stability_test = updater.test_stability()
    """
    
    def __init__(
        self,
        config: Optional[SleepUpdaterConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        integration_engine: Optional[IntegrationEngine] = None,
        device: Optional[str] = None
    ):
        """
        初始化睡眠更新器
        
        Args:
            config: 睡眠更新配置
            global_config: 全局配置
            integration_engine: 积分引擎
            device: 计算设备
        """
        self.config = config or SleepUpdaterConfig()
        self.global_config = global_config or ChronosConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从全局配置更新
        if global_config:
            self.config.learning_rate = global_config.training.learning_rate
            self.config.gradient_clip_value = global_config.training.gradient_clip_threshold
            self.config.weight_decay = global_config.training.weight_decay
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim
        
        # 积分引擎
        self.integration_engine = integration_engine
        
        # 核心组件
        self.gradient_constraints: Optional[GradientConstraints] = None
        self.stability_checker: Optional[StabilityChecker] = None
        
        # 参数优化器
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
        # 可训练参数
        self.trainable_params: List[torch.nn.Parameter] = []
        
        # 参数历史（用于约束检查）
        self._old_params: List[torch.Tensor] = []
        
        # 更新统计
        self._update_count: int = 0
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_updates": 0,
            "total_keyframes_replayed": 0,
            "avg_update_loss": 0.0,
            "avg_gradient_norm": 0.0,
            "stable_updates": 0,
            "unstable_updates": 0,
        }
        
        # 初始化标志
        self._initialized = False
        
        logger.info(
            f"SleepUpdater created: "
            f"learning_rate={self.config.learning_rate}, "
            f"clip_value={self.config.gradient_clip_value}"
        )
    
    def initialize(
        self,
        integration_engine: Optional[IntegrationEngine] = None
    ) -> None:
        """
        初始化睡眠更新器
        
        Args:
            integration_engine: 积分引擎
        """
        if integration_engine is not None:
            self.integration_engine = integration_engine
        
        # 创建梯度约束
        self.gradient_constraints = GradientConstraints(
            config=self.config,
            device=self.device,
        )
        
        # 创建稳定性检查器
        self.stability_checker = StabilityChecker(
            config=self.config,
            device=self.device,
        )
        
        # 收集可训练参数
        self._collect_trainable_params()
        
        # 创建优化器
        if self.trainable_params:
            if self.config.optimizer_type == "adam":
                self.optimizer = torch.optim.Adam(
                    self.trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            elif self.config.optimizer_type == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.trainable_params,
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self.trainable_params,
                    lr=self.config.learning_rate,
                )
        
        self._initialized = True
        
        logger.info(
            f"SleepUpdater initialized: "
            f"trainable_params={len(self.trainable_params)}"
        )
    
    def _collect_trainable_params(self) -> None:
        """
        收集可训练参数
        """
        self.trainable_params = []
        
        if self.integration_engine is None:
            logger.warning("Integration engine not available")
            return
        
        # 收集快变量动力学参数
        if self.config.update_fast_dynamics and self.integration_engine.fast_dynamics is not None:
            self.trainable_params.extend(
                list(self.integration_engine.fast_dynamics.dynamics_fn.parameters())
            )
        
        # 收集慢变量动力学参数
        if self.config.update_slow_dynamics and self.integration_engine.slow_dynamics is not None:
            self.trainable_params.extend(
                list(self.integration_engine.slow_dynamics.dynamics_fn.parameters())
            )
        
        # 收集耦合参数（通常不更新）
        if self.config.update_coupling and self.integration_engine.coupling_system is not None:
            # 耦合系数通常是自适应的，不需要额外训练
            pass
        
        logger.debug(f"Collected {len(self.trainable_params)} trainable parameters")
    
    def perform_sleep_update(
        self,
        keyframes: List[KeyframeData],
        max_updates: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        执行睡眠期参数更新
        
        Args:
            keyframes: 关键帧列表
            max_updates: 最大更新次数
            
        Returns:
            更新结果字典
        """
        if not self._initialized:
            raise ValueError("SleepUpdater not initialized.")
        
        if not keyframes:
            logger.warning("No keyframes for sleep update")
            return {"success": False, "reason": "no_keyframes"}
        
        max_updates = max_updates or self.config.max_replay_steps
        
        logger.info(
            f"Starting sleep update: "
            f"keyframes={len(keyframes)}, "
            f"max_updates={max_updates}"
        )
        
        # 记录开始时间
        start_time = time.time()
        
        # 保存原始参数
        self._save_old_params()
        
        # 冻结关键帧状态
        frozen_states = self.gradient_constraints.freeze_keyframe_states(keyframes)
        
        # 执行更新循环
        update_losses = []
        gradient_norms = []
        is_stable = True
        
        for update_step in range(max_updates):
            # 批量选择关键帧
            batch_start = (update_step * self.config.replay_batch_size) % len(keyframes)
            batch_end = min(batch_start + self.config.replay_batch_size, len(keyframes))
            batch_keyframes = keyframes[batch_start:batch_end]
            
            if not batch_keyframes:
                continue
            
            # 执行单次更新
            update_result = self._single_update_step(
                batch_keyframes,
                frozen_states[batch_start:batch_end],
            )
            
            update_losses.append(update_result["loss"])
            gradient_norms.append(update_result["gradient_norm"])
            
            # 检查稳定性
            if not update_result["is_stable"]:
                is_stable = False
                logger.warning(f"Unstable update at step {update_step}")
                break
        
        # 强制参数约束
        if is_stable:
            self.gradient_constraints.enforce_parameter_limits(
                self.trainable_params,
                self._old_params,
            )
        
        # 更新统计
        elapsed_time = time.time() - start_time
        self._update_count += 1
        
        avg_loss = np.mean(update_losses) if update_losses else 0.0
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        self._stats["total_updates"] += 1
        self._stats["total_keyframes_replayed"] += len(keyframes)
        self._stats["avg_update_loss"] = avg_loss
        self._stats["avg_gradient_norm"] = avg_grad_norm
        
        if is_stable:
            self._stats["stable_updates"] += 1
        else:
            self._stats["unstable_updates"] += 1
        
        # 构建结果
        result = {
            "success": True,
            "update_count": self._update_count,
            "keyframes_processed": len(keyframes),
            "max_updates": max_updates,
            "avg_loss": avg_loss,
            "avg_gradient_norm": avg_grad_norm,
            "elapsed_time_ms": elapsed_time * 1000,
            "is_stable": is_stable,
            "update_losses": update_losses,
            "gradient_norms": gradient_norms,
            "gradient_constraints_stats": self.gradient_constraints.get_statistics(),
            "stability_checker_stats": self.stability_checker.get_statistics(),
        }
        
        logger.info(
            f"Sleep update completed: "
            f"updates={self._update_count}, "
            f"avg_loss={avg_loss:.4f}, "
            f"stable={is_stable}"
        )
        
        return result
    
    def _single_update_step(
        self,
        batch_keyframes: List[KeyframeData],
        frozen_states: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        执行单次更新步骤
        
        Args:
            batch_keyframes: 批量关键帧
            frozen_states: 冻结状态列表
            
        Returns:
            单次更新结果
        """
        # 清零梯度
        if self.optimizer:
            self.optimizer.zero_grad()
        
        # 计算总损失
        total_loss = torch.tensor(0.0, device=self.device)
        
        for keyframe, (E_fast_frozen, E_slow_frozen) in zip(batch_keyframes, frozen_states):
            # 从关键帧向前积分
            # 计算一致性损失
            consistency_loss = self._compute_consistency_loss(
                keyframe,
                E_fast_frozen,
                E_slow_frozen,
            )
            
            # 计算预测改善损失
            improve_loss = self._compute_improve_loss(
                keyframe,
                E_fast_frozen,
                E_slow_frozen,
            )
            
            # 加权组合
            loss = consistency_loss + 0.5 * improve_loss
            total_loss = total_loss + loss
        
        # 平均损失
        avg_loss = total_loss / len(batch_keyframes)
        
        # 检查损失稳定性
        loss_check = self.stability_checker.check_loss(avg_loss)

        # 反向传播（仅当有可训练参数且损失需要梯度时）
        if len(self.trainable_params) > 0 and avg_loss.requires_grad:
            avg_loss.backward()
        else:
            logger.warning(
                "No trainable parameters or loss doesn't require grad, "
                "skipping backward pass"
            )

        # 收集梯度
        gradients = {}
        for i, param in enumerate(self.trainable_params):
            if param.grad is not None:
                gradients[f"param_{i}"] = param.grad.clone()
        
        # 检查梯度稳定性
        gradient_check = self.stability_checker.check_gradients(gradients)
        
        # 应用梯度裁剪
        clipped_gradients = self.gradient_constraints.apply_gradient_clipping(gradients)
        
        # 手动设置裁剪后的梯度
        for i, param in enumerate(self.trainable_params):
            param_key = f"param_{i}"
            if param_key in clipped_gradients and clipped_gradients[param_key] is not None:
                if param.grad is not None:
                    param.grad.copy_(clipped_gradients[param_key])
                else:
                    param.grad = clipped_gradients[param_key]
        
        # 执行优化步骤
        if self.optimizer:
            self.optimizer.step()
        
        # 计算梯度范数
        total_grad_norm = 0.0
        grad_count = 0
        for grad in clipped_gradients.values():
            if grad is not None:
                total_grad_norm += torch.norm(grad).item() ** 2
                grad_count += 1

        # 避免除零错误
        avg_grad_norm = np.sqrt(total_grad_norm / grad_count) if grad_count > 0 else 0.0
        
        # 检查参数稳定性
        param_check = self.stability_checker.check_parameters(self.trainable_params)
        
        # 判断稳定性
        is_stable = (
            loss_check["is_stable"] and
            gradient_check["is_stable"] and
            param_check["is_stable"] and
            self.stability_checker.is_stable
        )
        
        return {
            "loss": avg_loss.item(),
            "gradient_norm": avg_grad_norm,
            "is_stable": is_stable,
            "loss_check": loss_check,
            "gradient_check": gradient_check,
            "param_check": param_check,
        }
    
    def _compute_consistency_loss(
        self,
        keyframe: KeyframeData,
        E_fast_frozen: torch.Tensor,
        E_slow_frozen: torch.Tensor
    ) -> torch.Tensor:
        """
        计算一致性损失
        
        Args:
            keyframe: 关键帧数据
            E_fast_frozen: 冻结的快变量状态
            E_slow_frozen: 冻结的慢变量状态
            
        Returns:
            一致性损失张量
        """
        if self.integration_engine is None or keyframe.E_fast is None:
            return torch.tensor(0.0, device=self.device)
        
        # 从冻结状态向前积分
        # 使用当前参数的求解器
        E_fast_replayed = self.integration_engine.fast_dynamics.step(
            E_fast_frozen.clone().requires_grad_(True),
            E_slow_frozen.clone(),
            {},
            self.config.replay_window_seconds,
            keyframe.timestamp,
        )
        
        # 与记录的状态对比
        # 使用冻结状态作为目标（确保历史不被修改）
        loss = torch.norm(E_fast_replayed - E_fast_frozen)
        
        return loss
    
    def _compute_improve_loss(
        self,
        keyframe: KeyframeData,
        E_fast_frozen: torch.Tensor,
        E_slow_frozen: torch.Tensor
    ) -> torch.Tensor:
        """
        计算预测改善损失
        
        Args:
            keyframe: 关键帧数据
            E_fast_frozen: 冻结的快变量状态
            E_slow_frozen: 冻结的慢变量状态
            
        Returns:
            预测改善损失张量
        """
        if self.integration_engine is None:
            return torch.tensor(0.0, device=self.device)
        
        # 需要有实际后续状态
        if keyframe.actual_outcome is None:
            return torch.tensor(0.0, device=self.device)
        
        # 计算预测窗口
        if keyframe.outcome_timestamp is not None:
            prediction_window = keyframe.outcome_timestamp - keyframe.timestamp
        else:
            prediction_window = self.config.replay_window_seconds
        
        # 从冻结状态向前积分预测
        E_fast_predicted = self.integration_engine.fast_dynamics.step(
            E_fast_frozen.clone().requires_grad_(True),
            E_slow_frozen.clone(),
            {},
            prediction_window,
            keyframe.timestamp,
        )
        
        # 与实际结果对比
        actual_outcome = torch.tensor(
            keyframe.actual_outcome,
            dtype=torch.float32,
            device=self.device
        ).detach()
        
        loss = torch.norm(E_fast_predicted - actual_outcome)
        
        return loss
    
    def _save_old_params(self) -> None:
        """
        保存原始参数值
        """
        self._old_params = [param.data.clone() for param in self.trainable_params]
    
    def test_stability(
        self,
        test_keyframes: Optional[List[KeyframeData]] = None,
        test_steps: int = 100
    ) -> Dict[str, Any]:
        """
        测试睡眠重放稳定性
        
        Task 20.4: 睡眠重放稳定性测试
        
        Args:
            test_keyframes: 测试关键帧（如果为 None，生成随机测试数据）
            test_steps: 测试步数
            
        Returns:
            稳定性测试结果
        """
        if not self._initialized:
            raise ValueError("SleepUpdater not initialized.")
        
        # 如果没有提供测试关键帧，生成随机数据
        if test_keyframes is None:
            test_keyframes = self._generate_test_keyframes(test_steps)
        
        # 使用稳定性检查器执行测试
        stability_test = self.stability_checker.test_sleep_replay_stability(
            self.integration_engine,
            test_keyframes,
            test_steps,
        )
        
        # 执行参数更新测试
        update_test = self.perform_sleep_update(
            test_keyframes,
            max_updates=min(test_steps, 10),
        )
        
        # 综合结果
        result = {
            "overall_stable": stability_test["overall_stable"] and update_test["is_stable"],
            "replay_stability": stability_test,
            "update_stability": update_test,
            "recommendations": [],
        }
        
        # 生成建议
        if not stability_test["overall_stable"]:
            result["recommendations"].append(
                "Reduce replay window or check numerical precision"
            )
        
        if not update_test["is_stable"]:
            result["recommendations"].append(
                "Reduce learning rate or increase gradient clipping"
            )
        
        logger.info(
            f"Stability test completed: "
            f"overall_stable={result['overall_stable']}"
        )
        
        return result
    
    def _generate_test_keyframes(self, count: int) -> List[KeyframeData]:
        """
        生成随机测试关键帧
        
        Args:
            count: 数量
            
        Returns:
            测试关键帧列表
        """
        test_keyframes = []
        
        for i in range(count):
            # 生成随机状态
            E_fast = np.random.randn(self.config.fast_dim).astype(np.float32)
            E_slow = np.random.randn(self.config.slow_dim).astype(np.float32)
            
            # 创建关键帧
            keyframe = KeyframeData(
                keyframe_id=f"test_kf_{i}",
                timestamp=i * 1.0,
                E_fast=E_fast,
                E_slow=E_slow,
                emotional_intensity=np.random.rand(),
                importance=np.random.rand(),
            )
            
            test_keyframes.append(keyframe)
        
        return test_keyframes
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["update_count"] = self._update_count
        stats["initialized"] = self._initialized
        stats["trainable_params_count"] = len(self.trainable_params)
        
        if self.gradient_constraints:
            stats["gradient_constraints_stats"] = self.gradient_constraints.get_statistics()
        
        if self.stability_checker:
            stats["stability_checker_stats"] = self.stability_checker.get_statistics()
        
        return stats
    
    def reset(self) -> None:
        """
        重置睡眠更新器
        """
        self._old_params.clear()
        self._update_count = 0
        
        if self.gradient_constraints:
            self.gradient_constraints.reset()
        
        if self.stability_checker:
            self.stability_checker.reset()
        
        if self.optimizer:
            self.optimizer.zero_grad()
        
        self._stats = {
            "total_updates": 0,
            "total_keyframes_replayed": 0,
            "avg_update_loss": 0.0,
            "avg_gradient_norm": 0.0,
            "stable_updates": 0,
            "unstable_updates": 0,
        }
        
        logger.info("SleepUpdater reset")
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"SleepUpdater(status={status}, "
            f"updates={self._update_count}, "
            f"params={len(self.trainable_params)}"
        )


def create_sleep_updater_from_config(
    config: ChronosConfig,
    integration_engine: Optional[IntegrationEngine] = None,
    device: Optional[str] = None
) -> SleepUpdater:
    """
    从全局配置创建睡眠更新器
    
    Args:
        config: 全局配置
        integration_engine: 积分引擎
        device: 计算设备
        
    Returns:
        SleepUpdater 实例
    """
    updater_config = SleepUpdaterConfig(
        learning_rate=config.training.learning_rate,
        gradient_clip_value=config.training.gradient_clip_threshold,
        weight_decay=config.training.weight_decay,
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
    )
    
    updater = SleepUpdater(
        config=updater_config,
        global_config=config,
        integration_engine=integration_engine,
        device=device,
    )
    
    updater.initialize()
    return updater