"""
动力学对齐训练系统
================

实现 Chronos-Self 项目的动力学对齐训练，包括：
- 多步长一致性损失
- 半群性质正则损失
- 长时序开环损失（72小时）
- 周期性验证机制

核心功能：
- L_consistency = Σ_{i≠j} ||E^(i)(t+ΔT) - E^(j)(t+ΔT)||²
- L_ode = E_{t,Δt1,Δt2}[||f_θ(f_θ(E,Δt1),Δt2) - f_θ(E,Δt1+Δt2)||²]
- L_long = ||E_slow(T) - E_slow(0)||² + E_{t∈[0,T]}[||E(t) - Ē(t)||²]

使用示例：
    alignment_system = DynamicsAlignment(config=DynamicsAlignmentConfig())
    alignment_losses = alignment_system.compute_alignment_losses(
        state_sequence, integration_engine
    )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
import numpy as np

from chronos_core.utils.config import ChronosConfig, ValidationConfig, NeuralODEConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

logger = logging.getLogger(__name__)


@dataclass
class DynamicsAlignmentConfig:
    """动力学对齐训练配置"""

    # 多步长一致性配置
    multistep_sizes: List[int] = field(default_factory=lambda: [1, 10, 100])  # 不同步长大小
    multistep_weight: float = 0.1  # 多步长一致性权重

    # 半群性质正则化配置
    semigroup_enabled: bool = True
    semigroup_weight: float = 0.05  # 半群性质权重
    delta_t1_range: Tuple[float, float] = (0.01, 0.1)  # Δt1 范围
    delta_t2_range: Tuple[float, float] = (0.01, 0.1)  # Δt2 范围

    # 长时序开环验证配置
    long_sequence_hours: float = 72.0  # 72小时验证
    long_sequence_weight: float = 0.01  # 长时序权重
    baseline_drift_threshold: float = 0.1  # 基线漂移阈值
    self_prediction_error_threshold: float = 0.5  # 自预测误差阈值

    # 周期性验证配置
    validation_interval_epochs: int = 5  # 每 N 个 epoch 验证
    validation_frequency_steps: int = 1000  # 每 N 步验证
    save_best_model: bool = True  # 保存最佳模型
    early_stopping_threshold: float = 0.05  # 早停阈值

    # 数值稳定性
    max_sequence_length: int = 10000  # 最大序列长度
    gradient_clip_threshold: float = 10.0  # 梯度裁剪阈值
    numerical_stability_threshold: float = 1e6

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


class MultiStepConsistencyLoss(nn.Module):
    """
    多步长一致性损失

    计算不同步长积分结果的一致性：
        L_consistency = Σ_{i≠j} ||E^(i)(t+ΔT) - E^(j)(t+ΔT)||²

    物理意义：
    - 对同一状态序列使用不同步长分别积分
    - 要求终点一致（ODE的连续性）
    - 确保动力学的尺度不变性
    """

    def __init__(
        self,
        config: Optional[DynamicsAlignmentConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or DynamicsAlignmentConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 统计信息
        self._consistency_errors: List[float] = []
        self._call_count: int = 0

        logger.info(
            f"MultiStepConsistencyLoss created: "
            f"multistep_sizes={self.config.multistep_sizes}, "
            f"device={self.device}"
        )

    def forward(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState,
        total_time: float,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        计算多步长一致性损失

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态
            total_time: 总积分时间 ΔT
            num_samples: 样本数（可选）

        Returns:
            多步长一致性损失值
        """
        # 使用配置的步长大小
        step_sizes = self.config.multistep_sizes
        if num_samples is not None:
            step_sizes = step_sizes[:num_samples]

        # 使用不同步长积分到同一终点时间
        final_states: List[SelfState] = []

        for step_size in step_sizes:
            # 计算需要的步数
            num_steps = int(total_time * step_size)

            # 执行积分
            trajectory = integration_engine.integrate(
                initial_state=initial_state,
                num_steps=num_steps,
                record_trajectory=False
            )

            # 记录最终状态
            final_state = trajectory[-1]
            final_states.append(final_state)

        # 计算所有不同步长之间的终点一致性
        consistency_loss = torch.tensor(0.0, device=self.device)

        for i in range(len(final_states)):
            for j in range(i + 1, len(final_states)):
                # 计算状态差异
                state_i = final_states[i]
                state_j = final_states[j]

                # 快变量差异
                fast_diff = torch.norm(
                    state_i.E_fast.to(self.device) - state_j.E_fast.to(self.device),
                    p=2
                )

                # 慢变量差异
                slow_diff = torch.norm(
                    state_i.E_slow.to(self.device) - state_j.E_slow.to(self.device),
                    p=2
                )

                # 合并差异
                total_diff = fast_diff + slow_diff * 0.1

                # 累加一致性损失
                consistency_loss = consistency_loss + total_diff ** 2

        # 统计
        self._call_count += 1
        self._consistency_errors.append(consistency_loss.item())

        if len(self._consistency_errors) > 1000:
            self._consistency_errors = self._consistency_errors[-1000:]

        logger.debug(
            f"MultiStepConsistencyLoss computed: "
            f"step_sizes={step_sizes}, "
            f"loss={consistency_loss.item():.4f}"
        )

        return consistency_loss

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._consistency_errors),
            "mean_error": np.mean(self._consistency_errors) if self._consistency_errors else 0.0,
            "std_error": np.std(self._consistency_errors) if self._consistency_errors else 0.0,
            "multistep_sizes": self.config.multistep_sizes,
        }
        return stats


class SemigroupRegularizationLoss(nn.Module):
    """
    半群性质正则损失

    计算差分映射的半群性质：
        L_ode = E_{t,Δt1,Δt2}[||f_θ(f_θ(E,Δt1),Δt2) - f_θ(E,Δt1+Δt2)||²]

    物理意义：
    - 要求差分映射满足近似半群性质
    - 确保ODE动力学的正确性
    - f(E, Δt1+Δt2) ≈ f(f(E, Δt1), Δt2)
    """

    def __init__(
        self,
        config: Optional[DynamicsAlignmentConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or DynamicsAlignmentConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 统计信息
        self._semigroup_errors: List[float] = []
        self._call_count: int = 0

        logger.info(
            f"SemigroupRegularizationLoss created: "
            f"weight={self.config.semigroup_weight}, "
            f"device={self.device}"
        )

    def forward(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState,
        num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        计算半群性质正则损失

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态
            num_samples: 样本数（可选）

        Returns:
            半群性质正则损失值
        """
        if not self.config.semigroup_enabled:
            return torch.tensor(0.0, device=self.device)

        # 随机生成 Δt1 和 Δt2
        delta_t1_min, delta_t1_max = self.config.delta_t1_range
        delta_t2_min, delta_t2_max = self.config.delta_t2_range

        # 默认样本数
        if num_samples is None:
            num_samples = 10

        semigroup_loss = torch.tensor(0.0, device=self.device)

        for _ in range(num_samples):
            # 随机采样 Δt1 和 Δt2
            delta_t1 = np.random.uniform(delta_t1_min, delta_t1_max)
            delta_t2 = np.random.uniform(delta_t2_min, delta_t2_max)

            # 方法1: 分两步积分（f(f(E, Δt1), Δt2)）
            # 第一步：积分 Δt1
            trajectory1 = integration_engine.integrate(
                initial_state=initial_state,
                num_steps=int(delta_t1 * 100),  # 假设 dt=0.01
                record_trajectory=False
            )
            intermediate_state = trajectory1[-1]

            # 第二步：从中间状态积分 Δt2
            trajectory2 = integration_engine.integrate(
                initial_state=intermediate_state,
                num_steps=int(delta_t2 * 100),
                record_trajectory=False
            )
            final_state_two_steps = trajectory2[-1]

            # 方法2: 直接积分 Δt1+Δt2（f(E, Δt1+Δt2)）
            total_time = delta_t1 + delta_t2
            trajectory_direct = integration_engine.integrate(
                initial_state=initial_state,
                num_steps=int(total_time * 100),
                record_trajectory=False
            )
            final_state_direct = trajectory_direct[-1]

            # 计算差异
            fast_diff = torch.norm(
                final_state_two_steps.E_fast.to(self.device) -
                final_state_direct.E_fast.to(self.device),
                p=2
            )

            slow_diff = torch.norm(
                final_state_two_steps.E_slow.to(self.device) -
                final_state_direct.E_slow.to(self.device),
                p=2
            )

            # 合并差异
            total_diff = fast_diff + slow_diff * 0.1

            # 累加半群损失
            semigroup_loss = semigroup_loss + total_diff ** 2

        # 平均损失
        semigroup_loss = semigroup_loss / num_samples

        # 统计
        self._call_count += 1
        self._semigroup_errors.append(semigroup_loss.item())

        if len(self._semigroup_errors) > 1000:
            self._semigroup_errors = self._semigroup_errors[-1000:]

        logger.debug(
            f"SemigroupRegularizationLoss computed: "
            f"num_samples={num_samples}, "
            f"loss={semigroup_loss.item():.4f}"
        )

        return semigroup_loss

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._semigroup_errors),
            "mean_error": np.mean(self._semigroup_errors) if self._semigroup_errors else 0.0,
            "std_error": np.std(self._semigroup_errors) if self._semigroup_errors else 0.0,
            "enabled": self.config.semigroup_enabled,
        }
        return stats


class LongSequenceOpenLoopLoss(nn.Module):
    """
    长时序开环损失（72小时验证）

    计算长时间尺度的开环验证：
        L_long = ||E_slow(T) - E_slow(0)||² + E_{t∈[0,T]}[||E(t) - Ē(t)||²]

    物理意义：
    - 周期性进行无梯度截断的长时序积分
    - 计算基线漂移和自预测误差增长
    - 这是核心的P0级验证指标
    """

    def __init__(
        self,
        config: Optional[DynamicsAlignmentConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or DynamicsAlignmentConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 验证历史记录
        self._validation_history: List[Dict[str, float]] = []
        self._call_count: int = 0

        # 最佳模型记录
        self._best_loss: float = float('inf')
        self._best_model_state: Optional[Dict[str, Any]] = None

        logger.info(
            f"LongSequenceOpenLoopLoss created: "
            f"hours={self.config.long_sequence_hours}, "
            f"device={self.device}"
        )

    def forward(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState,
        duration_hours: Optional[float] = None,
        record_trajectory: bool = True
    ) -> Dict[str, Any]:
        """
        执行长时序开环验证

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态
            duration_hours: 验证时长（小时）
            record_trajectory: 是否记录轨迹

        Returns:
            验证结果字典：{
                'baseline_drift': 基线漂移,
                'self_prediction_error': 自预测误差,
                'final_state': 最终状态,
                'trajectory': 轨迹（可选）,
                'is_valid': 是否通过验证
            }
        """
        # 使用配置时长或指定时长
        duration_hours = duration_hours or self.config.long_sequence_hours

        # 记录初始基线
        initial_slow_norm = torch.norm(initial_state.E_slow.to(self.device)).item()
        initial_fast_norm = torch.norm(initial_state.E_fast.to(self.device)).item()

        # 执行长时序积分（无梯度截断）
        start_time = time.time()

        # 计算总步数（假设 dt=0.01s）
        total_steps = int(duration_hours * 3600 / 0.01)

        # 运行连续积分
        trajectory = integration_engine.integrate(
            initial_state=initial_state,
            num_steps=total_steps,
            record_trajectory=record_trajectory
        )

        # 获取最终状态
        final_state = trajectory[-1]

        # 计算基线漂移
        final_slow_norm = torch.norm(final_state.E_slow.to(self.device)).item()
        baseline_drift = abs(final_slow_norm - initial_slow_norm)

        # 计算自预测误差（简化：使用快变量的变化）
        final_fast_norm = torch.norm(final_state.E_fast.to(self.device)).item()
        self_prediction_error = abs(final_fast_norm - initial_fast_norm)

        # 检查是否通过验证
        is_valid = (
            baseline_drift < self.config.baseline_drift_threshold and
            self_prediction_error < self.config.self_prediction_error_threshold
        )

        # 计算总损失
        long_sequence_loss = torch.tensor(
            baseline_drift ** 2 + self_prediction_error ** 2,
            device=self.device
        )

        # 更新最佳模型
        if self.config.save_best_model and long_sequence_loss.item() < self._best_loss:
            self._best_loss = long_sequence_loss.item()
            # 记录最佳状态（简化，不实际保存完整模型）
            self._best_model_state = {
                "loss": self._best_loss,
                "baseline_drift": baseline_drift,
                "self_prediction_error": self_prediction_error,
                "timestamp": time.time(),
            }

        # 统计
        self._call_count += 1
        validation_result = {
            "loss": long_sequence_loss.item(),
            "baseline_drift": baseline_drift,
            "self_prediction_error": self_prediction_error,
            "duration_hours": duration_hours,
            "total_steps": total_steps,
            "elapsed_time": time.time() - start_time,
            "is_valid": is_valid,
        }
        self._validation_history.append(validation_result)

        if len(self._validation_history) > 100:
            self._validation_history = self._validation_history[-100:]

        logger.info(
            f"LongSequenceOpenLoopLoss computed: "
            f"duration={duration_hours}h, "
            f"baseline_drift={baseline_drift:.4f}, "
            f"self_prediction_error={self_prediction_error:.4f}, "
            f"is_valid={is_valid}"
        )

        # 返回完整验证结果
        result = {
            "loss": long_sequence_loss,
            "baseline_drift": baseline_drift,
            "self_prediction_error": self_prediction_error,
            "final_state": final_state,
            "is_valid": is_valid,
            "elapsed_time": time.time() - start_time,
        }

        if record_trajectory:
            result["trajectory"] = trajectory

        return result

    def get_best_model_state(self) -> Optional[Dict[str, Any]]:
        """获取最佳模型状态"""
        return self._best_model_state

    def get_validation_history(self) -> List[Dict[str, float]]:
        """获取验证历史"""
        return self._validation_history

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._validation_history),
            "best_loss": self._best_loss,
            "best_model_state": self._best_model_state,
            "mean_baseline_drift": np.mean([h["baseline_drift"] for h in self._validation_history]) if self._validation_history else 0.0,
            "mean_self_prediction_error": np.mean([h["self_prediction_error"] for h in self._validation_history]) if self._validation_history else 0.0,
            "pass_rate": np.mean([h["is_valid"] for h in self._validation_history]) if self._validation_history else 0.0,
        }
        return stats


class PeriodicValidation(nn.Module):
    """
    周期性验证机制

    定期执行验证，监测训练进展：
    - 每 N 个 epoch 进行一次验证
    - 记录验证结果和趋势
    - 保存最佳模型
    - 支持早停机制
    """

    def __init__(
        self,
        config: Optional[DynamicsAlignmentConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or DynamicsAlignmentConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 验证状态
        self._current_epoch: int = 0
        self._current_step: int = 0
        self._last_validation_epoch: int = 0
        self._last_validation_step: int = 0

        # 验证历史
        self._validation_records: List[Dict[str, Any]] = []

        # 早停状态
        self._best_validation_loss: float = float('inf')
        self._patience_counter: int = 0

        logger.info(
            f"PeriodicValidation created: "
            f"interval_epochs={self.config.validation_interval_epochs}, "
            f"frequency_steps={self.config.validation_frequency_steps}"
        )

    def forward(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        force_validation: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        执行周期性验证

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态
            epoch: 当前 epoch
            step: 当前步数
            force_validation: 强制验证

        Returns:
            验证结果（如果执行验证）或 None
        """
        # 更新当前状态
        if epoch is not None:
            self._current_epoch = epoch
        if step is not None:
            self._current_step = step

        # 检查是否应该验证
        should_validate = force_validation or self._should_validate()

        if not should_validate:
            return None

        # 执行验证
        logger.info(
            f"Periodic validation triggered: "
            f"epoch={self._current_epoch}, "
            f"step={self._current_step}"
        )

        # 创建长时序验证模块
        long_sequence_validator = LongSequenceOpenLoopLoss(
            config=self.config,
            device=self.device
        )

        # 执行验证（使用较短时长以节省时间）
        validation_result = long_sequence_validator.forward(
            integration_engine=integration_engine,
            initial_state=initial_state,
            duration_hours=1.0,  # 使用1小时作为验证时长
            record_trajectory=False
        )

        # 记录验证结果
        validation_record = {
            "epoch": self._current_epoch,
            "step": self._current_step,
            "timestamp": time.time(),
            "result": validation_result,
        }
        self._validation_records.append(validation_record)

        # 更新验证时间
        self._last_validation_epoch = self._current_epoch
        self._last_validation_step = self._current_step

        # 检查早停
        early_stop = self._check_early_stopping(validation_result)

        if early_stop:
            logger.info(
                f"Early stopping triggered: "
                f"best_loss={self._best_validation_loss:.4f}, "
                f"patience={self._patience_counter}"
            )
            validation_record["early_stop"] = True

        return validation_record

    def _should_validate(self) -> bool:
        """判断是否应该验证"""
        # 基于 epoch 验证
        if self._current_epoch > 0 and self._last_validation_epoch > 0:
            if self._current_epoch - self._last_validation_epoch >= self.config.validation_interval_epochs:
                return True

        # 基于 step 验证
        if self._current_step > 0 and self._last_validation_step > 0:
            if self._current_step - self._last_validation_step >= self.config.validation_frequency_steps:
                return True

        # 初始验证
        if self._last_validation_epoch == 0 and self._current_epoch > 0:
            return True

        return False

    def _check_early_stopping(self, validation_result: Dict[str, Any]) -> bool:
        """检查早停"""
        current_loss = validation_result["loss"].item()

        # 更新最佳损失
        if current_loss < self._best_validation_loss:
            self._best_validation_loss = current_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1

        # 检查早停条件
        if self._patience_counter > 10:  # 超过10次没有改进
            return True

        # 检查阈值条件
        if current_loss > self.config.early_stopping_threshold:
            return True

        return False

    def get_validation_records(self) -> List[Dict[str, Any]]:
        """获取验证记录"""
        return self._validation_records

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "current_epoch": self._current_epoch,
            "current_step": self._current_step,
            "last_validation_epoch": self._last_validation_epoch,
            "last_validation_step": self._last_validation_step,
            "validation_count": len(self._validation_records),
            "best_validation_loss": self._best_validation_loss,
            "patience_counter": self._patience_counter,
        }
        return stats

    def reset(self) -> None:
        """重置验证状态"""
        self._current_epoch = 0
        self._current_step = 0
        self._last_validation_epoch = 0
        self._last_validation_step = 0
        self._validation_records.clear()
        self._best_validation_loss = float('inf')
        self._patience_counter = 0

        logger.info("PeriodicValidation reset")


class DynamicsAlignment(nn.Module):
    """
    完整动力学对齐训练系统

    整合所有动力学对齐损失：
    - 多步长一致性损失
    - 半群性质正则损失
    - 长时序开环损失
    - 周期性验证机制

    完整损失：
        L_alignment = λ_consistency·L_consistency +
                      λ_semigroup·L_semigroup +
                      λ_long·L_long
    """

    def __init__(
        self,
        config: Optional[DynamicsAlignmentConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or DynamicsAlignmentConfig()

        # 从全局配置更新
        if global_config:
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim
            self.config.long_sequence_hours = global_config.validation.p0_open_loop_hours
            self.config.baseline_drift_threshold = global_config.validation.p0_max_baseline_drift

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建各个对齐损失模块
        self.multistep_consistency = MultiStepConsistencyLoss(
            config=self.config,
            device=self.device
        )

        self.semigroup_regularization = SemigroupRegularizationLoss(
            config=self.config,
            device=self.device
        )

        self.long_sequence_validation = LongSequenceOpenLoopLoss(
            config=self.config,
            device=self.device
        )

        self.periodic_validation = PeriodicValidation(
            config=self.config,
            device=self.device
        )

        # 统计信息
        self._alignment_history: List[Dict[str, float]] = []
        self._call_count: int = 0

        logger.info(
            f"DynamicsAlignment created: "
            f"multistep_weight={self.config.multistep_weight}, "
            f"semigroup_weight={self.config.semigroup_weight}, "
            f"long_sequence_weight={self.config.long_sequence_weight}, "
            f"device={self.device}"
        )

    def forward(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        include_long_sequence: bool = False,
        force_validation: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有动力学对齐损失

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态
            epoch: 当前 epoch
            step: 当前步数
            include_long_sequence: 是否包含长时序验证
            force_validation: 强制验证

        Returns:
            对齐损失字典：{
                'total': 总对齐损失,
                'consistency': 多步长一致性损失,
                'semigroup': 半群性质损失,
                'long_sequence': 长时序损失（可选）,
                'validation_result': 验证结果（可选）,
                'weights': 权重字典
            }
        """
        # 计算多步长一致性损失
        L_consistency = self.multistep_consistency.forward(
            integration_engine=integration_engine,
            initial_state=initial_state,
            total_time=1.0  # 使用1秒作为测试时长
        )

        # 计算半群性质正则损失
        L_semigroup = self.semigroup_regularization.forward(
            integration_engine=integration_engine,
            initial_state=initial_state
        )

        # 初始化损失字典
        losses = {
            'consistency': L_consistency,
            'semigroup': L_semigroup,
            'long_sequence': torch.tensor(0.0, device=self.device),
        }

        # 计算长时序验证（可选）
        validation_result = None
        if include_long_sequence:
            L_long_result = self.long_sequence_validation.forward(
                integration_engine=integration_engine,
                initial_state=initial_state,
                duration_hours=self.config.long_sequence_hours,
                record_trajectory=False
            )
            losses['long_sequence'] = L_long_result['loss']

        # 执行周期性验证
        validation_result = self.periodic_validation.forward(
            integration_engine=integration_engine,
            initial_state=initial_state,
            epoch=epoch,
            step=step,
            force_validation=force_validation
        )

        # 计算总对齐损失
        total_alignment_loss = (
            self.config.multistep_weight * losses['consistency'] +
            self.config.semigroup_weight * losses['semigroup'] +
            self.config.long_sequence_weight * losses['long_sequence']
        )

        # 添加总损失和权重到字典
        losses['total'] = total_alignment_loss
        losses['weights'] = {
            'consistency': self.config.multistep_weight,
            'semigroup': self.config.semigroup_weight,
            'long_sequence': self.config.long_sequence_weight,
        }

        # 添加验证结果（如果有）
        if validation_result is not None:
            losses['validation_result'] = validation_result

        # 统计
        self._call_count += 1
        self._alignment_history.append({
            'total': total_alignment_loss.item(),
            'consistency': losses['consistency'].item(),
            'semigroup': losses['semigroup'].item(),
            'long_sequence': losses['long_sequence'].item(),
        })

        if len(self._alignment_history) > 1000:
            self._alignment_history = self._alignment_history[-1000:]

        logger.debug(
            f"DynamicsAlignment computed: "
            f"total={total_alignment_loss.item():.4f}, "
            f"consistency={losses['consistency'].item():.4f}, "
            f"semigroup={losses['semigroup'].item():.4f}"
        )

        return losses

    def compute_alignment_losses(
        self,
        integration_engine: IntegrationEngine,
        initial_state: SelfState
    ) -> Dict[str, torch.Tensor]:
        """
        计算对齐损失（简化接口）

        Args:
            integration_engine: 积分引擎
            initial_state: 初始状态

        Returns:
            对齐损失字典
        """
        return self.forward(
            integration_engine=integration_engine,
            initial_state=initial_state
        )

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """获取验证历史"""
        return self.periodic_validation.get_validation_records()

    def get_best_model_state(self) -> Optional[Dict[str, Any]]:
        """获取最佳模型状态"""
        return self.long_sequence_validation.get_best_model_state()

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 各个子损失统计
        consistency_stats = self.multistep_consistency.get_statistics()
        semigroup_stats = self.semigroup_regularization.get_statistics()
        long_sequence_stats = self.long_sequence_validation.get_statistics()
        validation_stats = self.periodic_validation.get_statistics()

        # 总对齐统计
        if self._alignment_history:
            total_losses = [h['total'] for h in self._alignment_history]
            consistency_losses = [h['consistency'] for h in self._alignment_history]
            semigroup_losses = [h['semigroup'] for h in self._alignment_history]

            stats = {
                "call_count": self._call_count,
                "history_length": len(self._alignment_history),
                "weights": {
                    "consistency": self.config.multistep_weight,
                    "semigroup": self.config.semigroup_weight,
                    "long_sequence": self.config.long_sequence_weight,
                },
                "total_loss": {
                    "mean": np.mean(total_losses),
                    "std": np.std(total_losses),
                    "min": np.min(total_losses),
                    "max": np.max(total_losses),
                },
                "consistency_loss": {
                    "mean": np.mean(consistency_losses),
                    "std": np.std(consistency_losses),
                },
                "semigroup_loss": {
                    "mean": np.mean(semigroup_losses),
                    "std": np.std(semigroup_losses),
                },
                "consistency_stats": consistency_stats,
                "semigroup_stats": semigroup_stats,
                "long_sequence_stats": long_sequence_stats,
                "validation_stats": validation_stats,
            }
        else:
            stats = {
                "call_count": self._call_count,
                "history_length": 0,
                "weights": {
                    "consistency": self.config.multistep_weight,
                    "semigroup": self.config.semigroup_weight,
                    "long_sequence": self.config.long_sequence_weight,
                },
            }

        return stats

    def reset(self) -> None:
        """重置所有状态"""
        self._alignment_history.clear()
        self._call_count = 0

        # 重置子模块
        self.multistep_consistency._consistency_errors.clear()
        self.multistep_consistency._call_count = 0

        self.semigroup_regularization._semigroup_errors.clear()
        self.semigroup_regularization._call_count = 0

        self.long_sequence_validation._validation_history.clear()
        self.long_sequence_validation._call_count = 0

        self.periodic_validation.reset()

        logger.info("DynamicsAlignment reset")

    def __repr__(self) -> str:
        return (
            f"DynamicsAlignment("
            f"calls={self._call_count}, "
            f"validation_count={len(self.periodic_validation.get_validation_records())})"
        )


def create_dynamics_alignment_from_config(
    global_config: ChronosConfig,
    device: Optional[str] = None
) -> DynamicsAlignment:
    """
    从全局配置创建动力学对齐系统

    Args:
        global_config: 全局配置
        device: 计算设备

    Returns:
        DynamicsAlignment 实例
    """
    alignment_config = DynamicsAlignmentConfig(
        fast_dim=global_config.dim.fast_variable_dim,
        slow_dim=global_config.dim.slow_variable_dim,
        long_sequence_hours=global_config.validation.p0_open_loop_hours,
        baseline_drift_threshold=global_config.validation.p0_max_baseline_drift,
        validation_interval_epochs=global_config.training.validation_frequency,
    )

    alignment_system = DynamicsAlignment(
        config=alignment_config,
        global_config=global_config,
        device=device
    )

    return alignment_system