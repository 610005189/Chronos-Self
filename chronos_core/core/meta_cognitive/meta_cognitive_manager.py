"""
元认知管理器 - Meta-Cognitive Manager
======================================

Task 15: 实现元认知管理器，负责扰动训练、部分依赖机制和消融测试。

核心功能：
- 随机噪声扰动：迫使 L1 不能完全依赖 L2 信号
- 部分依赖机制：L2 仅作为"建议"而非"指令"
- L2 消融测试：移除 L2 调控信号，测试 L1 功能维持率
- 独立性验证：确保自指截断机制有效

关键特性：
- 扰动训练防止 L1 过度依赖 L2
- 部分依赖权重 ∈ [0.3, 0.7]
- 消融测试验证 L1 独立性（维持率 > 0.4）
- 维持率 = 1 表示自指截断失效
- 维持率 ≈ 0 表示 L2 无贡献
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import time
from collections import deque

from chronos_core.core.meta_cognitive.meta_cognitive_layer import (
    MetaCognitiveLayer,
    MetaCognitiveLayerConfig,
)
from chronos_core.core.meta_cognitive.self_state_layer import (
    SelfStateLayer,
    SelfStateLayerConfig,
)
from chronos_core.utils.config import (
    DimensionalityConfig,
    MetaCognitiveConfig,
    ChronosConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaCognitiveManagerConfig:
    """元认知管理器配置"""
    
    # 扰动参数
    perturbation_noise_sigma: float = 0.1       # 扰动噪声标准差 σ_C
    perturbation_enabled: bool = True           # 是否启用扰动
    
    # 部分依赖参数
    dependency_weight_min: float = 0.3          # 最小依赖权重
    dependency_weight_max: float = 0.7          # 最大依赖权重
    dependency_weight_default: float = 0.5      # 默认依赖权重
    dependency_adaptation_rate: float = 0.01    # 依赖权重自适应速率
    
    # 消融测试参数
    ablation_enabled: bool = True               # 是否启用消融测试
    ablation_threshold: float = 0.4             # 功能维持率阈值
    ablation_test_interval: int = 100           # 消融测试间隔步数
    ablation_window_size: int = 50              # 消融测试窗口大小
    
    # 性能记录参数
    performance_history_size: int = 100         # 性能历史大小
    
    # 设备参数
    device: str = "cpu"


class L2PerturbationTrainer:
    """
    L2 扰动训练器
    
    实现随机噪声扰动：
    - 调控信号扰动：C_actual(t) = C(t) + ε_C
    - ε_C ~ N(0, σ_C²)，σ_C 可配置
    
    目的：
    - 防止 L1 过度依赖 L2 信号
    - 增强系统鲁棒性
    - 防止 L2 被内化
    
    Attributes:
        noise_sigma: 扰动噪声标准差
        perturbation_enabled: 是否启用扰动
    """
    
    def __init__(
        self,
        noise_sigma: float = 0.1,
        perturbation_enabled: bool = True,
        control_signal_dim: int = 128,
        device: str = "cpu"
    ):
        """
        初始化 L2 扰动训练器
        
        Args:
            noise_sigma: 扰动噪声标准差 σ_C
            perturbation_enabled: 是否启用扰动
            control_signal_dim: 调控信号维度
            device: 计算设备
        """
        self.noise_sigma = noise_sigma
        self.perturbation_enabled = perturbation_enabled
        self.control_signal_dim = control_signal_dim
        self.device = device
        
        # 扰动历史记录
        self._perturbation_history: deque = deque(maxlen=100)
        
        # 统计信息
        self._stats = {
            "total_perturbations": 0,
            "mean_perturbation_norm": 0.0,
            "max_perturbation_norm": 0.0,
        }
        
        logger.info(
            f"L2PerturbationTrainer initialized: "
            f"noise_sigma={noise_sigma}, enabled={perturbation_enabled}"
        )
    
    def perturb_control_signal(
        self,
        control_signal: torch.Tensor
    ) -> torch.Tensor:
        """
        对调控信号添加扰动
        
        Args:
            control_signal: L2 调控信号
        
        Returns:
            扰动后的调控信号 C_actual(t) = C(t) + ε_C
        """
        control_signal = control_signal.to(self.device)
        
        if not self.perturbation_enabled:
            return control_signal
        
        # 生成扰动噪声：ε_C ~ N(0, σ_C²)
        epsilon = torch.randn_like(control_signal) * self.noise_sigma
        
        # 添加扰动
        perturbed_signal = control_signal + epsilon
        
        # 记录扰动历史
        perturbation_norm = torch.norm(epsilon).item()
        self._perturbation_history.append(perturbation_norm)
        
        # 更新统计
        self._stats["total_perturbations"] += 1
        if len(self._perturbation_history) > 0:
            self._stats["mean_perturbation_norm"] = np.mean(list(self._perturbation_history))
            self._stats["max_perturbation_norm"] = max(self._perturbation_history)
        
        logger.debug(
            f"Perturbed control signal: "
            f"original_norm={torch.norm(control_signal).item():.4f}, "
            f"perturbation_norm={perturbation_norm:.4f}"
        )
        
        return perturbed_signal
    
    def compute_perturbation_strength(
        self,
        control_signal: torch.Tensor,
        perturbed_signal: torch.Tensor
    ) -> float:
        """
        计算扰动强度
        
        Args:
            control_signal: 原始调控信号
            perturbed_signal: 扰动后的调控信号
        
        Returns:
            扰动强度比率
        """
        original_norm = torch.norm(control_signal).item()
        perturbation_norm = torch.norm(perturbed_signal - control_signal).item()
        
        if original_norm > 0:
            strength_ratio = perturbation_norm / original_norm
        else:
            strength_ratio = perturbation_norm
        
        return strength_ratio
    
    def set_noise_sigma(self, sigma: float):
        """
        设置扰动噪声标准差
        
        Args:
            sigma: 新的噪声标准差
        """
        self.noise_sigma = sigma
        logger.info(f"Updated perturbation noise sigma to {sigma}")
    
    def enable_perturbation(self, enabled: bool):
        """
        启用或禁用扰动
        
        Args:
            enabled: 是否启用扰动
        """
        self.perturbation_enabled = enabled
        logger.info(f"Perturbation {'enabled' if enabled else 'disabled'}")
    
    def reset(self):
        """重置扰动训练器"""
        self._perturbation_history.clear()
        self._stats = {
            "total_perturbations": 0,
            "mean_perturbation_norm": 0.0,
            "max_perturbation_norm": 0.0,
        }
        logger.debug("L2PerturbationTrainer reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["noise_sigma"] = self.noise_sigma
        stats["perturbation_enabled"] = self.perturbation_enabled
        stats["perturbation_history_length"] = len(self._perturbation_history)
        
        if len(self._perturbation_history) > 0:
            stats["recent_perturbation_norms"] = list(self._perturbation_history)[-10:]
        
        return stats


class L2AblationTester:
    """
    L2 消融测试器
    
    实现 L2 消融测试：
    - 移除 L2 调控信号
    - 记录消融前后的性能差异
    - 测试 L1 的功能维持率
    
    独立性验证：
    - 维持率 > 0.4：自指截断有效
    - 维持率 = 1：自指截断失效（L2 已被内化）
    - 维持率 ≈ 0：L2 无贡献
    
    Attributes:
        ablation_threshold: 功能维持率阈值
        ablation_window_size: 消融测试窗口大小
    """
    
    def __init__(
        self,
        ablation_threshold: float = 0.4,
        ablation_window_size: int = 50,
        device: str = "cpu"
    ):
        """
        初始化 L2 消融测试器
        
        Args:
            ablation_threshold: 功能维持率阈值
            ablation_window_size: 消融测试窗口大小
            device: 计算设备
        """
        self.ablation_threshold = ablation_threshold
        self.ablation_window_size = ablation_window_size
        self.device = device
        
        # 消融状态
        self._ablation_active: bool = False
        
        # 性能历史（消融前后）
        self._pre_ablation_performance: deque = deque(maxlen=ablation_window_size)
        self._post_ablation_performance: deque = deque(maxlen=ablation_window_size)
        
        # 消融测试结果
        self._ablation_results: List[Dict[str, Any]] = []
        
        # 统计信息
        self._stats = {
            "total_ablation_tests": 0,
            "average_retention_rate": 0.0,
            "min_retention_rate": 1.0,
            "max_retention_rate": 0.0,
        }
        
        logger.info(
            f"L2AblationTester initialized: "
            f"threshold={ablation_threshold}, window_size={ablation_window_size}"
        )
    
    def start_ablation(self):
        """
        开始消融测试
        
        移除 L2 调控信号
        """
        self._ablation_active = True
        self._pre_ablation_performance.clear()
        logger.info("L2 ablation test started")
    
    def end_ablation(self):
        """
        结束消融测试
        
        恢复 L2 调控信号
        """
        self._ablation_active = False
        self._post_ablation_performance.clear()
        logger.info("L2 ablation test ended")
    
    def is_ablation_active(self) -> bool:
        """
        检查消融是否激活
        
        Returns:
            是否处于消融状态
        """
        return self._ablation_active
    
    def record_performance(
        self,
        performance_metric: float,
        is_pre_ablation: bool = True
    ):
        """
        记录性能指标
        
        Args:
            performance_metric: 性能指标值
            is_pre_ablation: 是否为消融前性能
        """
        if is_pre_ablation:
            self._pre_ablation_performance.append(performance_metric)
        else:
            self._post_ablation_performance.append(performance_metric)
        
        logger.debug(
            f"Recorded performance: "
            f"metric={performance_metric:.4f}, "
            f"phase={'pre-ablation' if is_pre_ablation else 'post-ablation'}"
        )
    
    def compute_retention_rate(self) -> float:
        """
        计算功能维持率
        
        维持率 = post_ablation_performance / pre_ablation_performance
        
        Returns:
            功能维持率
        """
        if len(self._pre_ablation_performance) == 0 or \
           len(self._post_ablation_performance) == 0:
            logger.warning("Insufficient performance data for retention rate calculation")
            return 0.0
        
        # 计算平均性能
        pre_mean = np.mean(list(self._pre_ablation_performance))
        post_mean = np.mean(list(self._post_ablation_performance))
        
        # 计算维持率
        if pre_mean > 0:
            retention_rate = post_mean / pre_mean
        else:
            retention_rate = 0.0
        
        logger.info(
            f"Computed retention rate: "
            f"pre_mean={pre_mean:.4f}, post_mean={post_mean:.4f}, "
            f"retention_rate={retention_rate:.4f}"
        )
        
        return retention_rate
    
    def validate_independence(self) -> Tuple[bool, Dict[str, Any]]:
        """
        验证独立性
        
        检查 L1 的功能维持率是否满足要求
        
        Returns:
            (is_valid, statistics): 是否独立以及统计信息
        """
        retention_rate = self.compute_retention_rate()
        
        # 验证独立性
        is_valid = retention_rate > self.ablation_threshold
        
        # 检查特殊情况
        if retention_rate >= 1.0:
            logger.warning(
                "Retention rate = 1.0: L2 may have been internalized! "
                "Self-reference truncation mechanism failed."
            )
        
        if retention_rate <= 0.0:
            logger.warning(
                "Retention rate ≈ 0: L2 has no contribution!"
            )
        
        # 记录结果
        result = {
            "retention_rate": retention_rate,
            "threshold": self.ablation_threshold,
            "is_valid": is_valid,
            "pre_ablation_mean": np.mean(list(self._pre_ablation_performance)) if self._pre_ablation_performance else 0.0,
            "post_ablation_mean": np.mean(list(self._post_ablation_performance)) if self._post_ablation_performance else 0.0,
            "timestamp": time.time(),
        }
        self._ablation_results.append(result)
        
        # 更新统计
        self._stats["total_ablation_tests"] += 1
        self._stats["average_retention_rate"] = np.mean(
            [r["retention_rate"] for r in self._ablation_results]
        )
        self._stats["min_retention_rate"] = min(
            [r["retention_rate"] for r in self._ablation_results]
        )
        self._stats["max_retention_rate"] = max(
            [r["retention_rate"] for r in self._ablation_results]
        )
        
        logger.info(
            f"Independence validation: "
            f"retention_rate={retention_rate:.4f}, "
            f"threshold={self.ablation_threshold}, "
            f"is_valid={is_valid}"
        )
        
        return is_valid, result
    
    def get_ablation_results(self) -> List[Dict[str, Any]]:
        """
        获取消融测试结果
        
        Returns:
            消融测试结果列表
        """
        return self._ablation_results.copy()
    
    def get_latest_retention_rate(self) -> float:
        """
        获取最新的功能维持率
        
        Returns:
            最新功能维持率
        """
        if self._ablation_results:
            return self._ablation_results[-1]["retention_rate"]
        return 0.0
    
    def set_threshold(self, threshold: float):
        """
        设置功能维持率阈值
        
        Args:
            threshold: 新的阈值
        """
        self.ablation_threshold = threshold
        logger.info(f"Updated ablation threshold to {threshold}")
    
    def reset(self):
        """重置消融测试器"""
        self._ablation_active = False
        self._pre_ablation_performance.clear()
        self._post_ablation_performance.clear()
        self._ablation_results.clear()
        self._stats = {
            "total_ablation_tests": 0,
            "average_retention_rate": 0.0,
            "min_retention_rate": 1.0,
            "max_retention_rate": 0.0,
        }
        logger.debug("L2AblationTester reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["ablation_threshold"] = self.ablation_threshold
        stats["ablation_active"] = self._ablation_active
        stats["pre_ablation_history_length"] = len(self._pre_ablation_performance)
        stats["post_ablation_history_length"] = len(self._post_ablation_performance)
        stats["ablation_results_count"] = len(self._ablation_results)
        
        if self._ablation_results:
            stats["latest_retention_rate"] = self._ablation_results[-1]["retention_rate"]
        
        return stats


class MetaCognitiveManager(nn.Module):
    """
    元认知管理器
    
    Task 15 完整实现
    
    功能：
    - 随机噪声扰动（L2PerturbationTrainer）
    - 部分依赖机制
    - L2 消融测试（L2AblationTester）
    - 独立性验证
    
    特性：
    - 扰动训练防止 L1 过度依赖 L2
    - 部分依赖权重 ∈ [0.3, 0.7]
    - 消融测试验证 L1 独立性
    - 自指截断机制验证
    """
    
    def __init__(
        self,
        config: Optional[MetaCognitiveManagerConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        control_signal_dim: int = 128,
        device: Optional[str] = None
    ):
        """
        初始化元认知管理器
        
        Args:
            config: 管理器配置
            meta_config: 元认知配置
            control_signal_dim: 调控信号维度
            device: 计算设备
        """
        super().__init__()
        
        # 合并配置
        self.config = config or MetaCognitiveManagerConfig()
        
        if meta_config:
            self.config.perturbation_noise_sigma = meta_config.l2_perturbation_noise
            self.config.ablation_threshold = meta_config.l2_ablation_threshold
        
        self.control_signal_dim = control_signal_dim
        self.device = device or self.config.device
        
        # 创建扰动训练器
        self.perturbation_trainer = L2PerturbationTrainer(
            noise_sigma=self.config.perturbation_noise_sigma,
            perturbation_enabled=self.config.perturbation_enabled,
            control_signal_dim=self.control_signal_dim,
            device=self.device
        )
        
        # 创建消融测试器
        self.ablation_tester = L2AblationTester(
            ablation_threshold=self.config.ablation_threshold,
            ablation_window_size=self.config.ablation_window_size,
            device=self.device
        )
        
        # 部分依赖权重
        self._dependency_weight: float = self.config.dependency_weight_default
        
        # 依赖权重历史
        self._dependency_history: deque = deque(maxlen=self.config.performance_history_size)
        
        # 统计信息
        self._stats = {
            "total_control_signals_processed": 0,
            "total_dependency_adjustments": 0,
            "average_dependency_weight": self._dependency_weight,
        }
        
        self.to(self.device)
        
        logger.info(
            f"MetaCognitiveManager initialized: "
            f"perturbation_sigma={self.config.perturbation_noise_sigma}, "
            f"dependency_range=[{self.config.dependency_weight_min}, {self.config.dependency_weight_max}], "
            f"ablation_threshold={self.config.ablation_threshold}"
        )
    
    def process_control_signal(
        self,
        control_signal: torch.Tensor,
        apply_perturbation: bool = True
    ) -> Tuple[torch.Tensor, float]:
        """
        处理调控信号
        
        流程：
        1. 扰动（可选）
        2. 应用依赖权重
        
        Args:
            control_signal: L2 调控信号
            apply_perturbation: 是否应用扰动
        
        Returns:
            (processed_signal, dependency_weight): 处理后的信号和依赖权重
        """
        control_signal = control_signal.to(self.device)
        
        # 1. 扰动
        if apply_perturbation and self.perturbation_trainer.perturbation_enabled:
            perturbed_signal = self.perturbation_trainer.perturb_control_signal(
                control_signal
            )
        else:
            perturbed_signal = control_signal
        
        # 2. 应用依赖权重
        # processed_signal = perturbed_signal * dependency_weight
        processed_signal = perturbed_signal * self._dependency_weight
        
        # 记录依赖历史
        self._dependency_history.append(self._dependency_weight)
        
        # 更新统计
        self._stats["total_control_signals_processed"] += 1
        self._stats["average_dependency_weight"] = np.mean(list(self._dependency_history))
        
        logger.debug(
            f"Processed control signal: "
            f"original_norm={torch.norm(control_signal).item():.4f}, "
            f"processed_norm={torch.norm(processed_signal).item():.4f}, "
            f"dependency_weight={self._dependency_weight:.4f}"
        )
        
        return processed_signal, self._dependency_weight
    
    def adjust_dependency_weight(
        self,
        performance_metric: float,
        adaptation_rate: Optional[float] = None
    ):
        """
        自适应调整依赖权重
        
        基于性能指标调整 L1 对 L2 的依赖度
        
        Args:
            performance_metric: 性能指标（越高越好）
            adaptation_rate: 自适应速率（可选）
        """
        adaptation_rate = adaptation_rate or self.config.dependency_adaptation_rate
        
        # 自适应调整
        # 性能高 -> 降低依赖（L1 更独立）
        # 性能低 -> 增加依赖（需要更多 L2 调控）
        
        # 计算调整方向
        # 使用 sigmoid 映射性能指标到调整因子
        # numpy 没有 sigmoid，手动实现
        adjustment_factor = 1.0 / (1.0 + np.exp(-performance_metric))
        
        # 计算新依赖权重
        target_weight = self.config.dependency_weight_min + \
                       adjustment_factor * (self.config.dependency_weight_max - self.config.dependency_weight_min)
        
        # 平滑更新
        self._dependency_weight = (
            adaptation_rate * target_weight +
            (1 - adaptation_rate) * self._dependency_weight
        )
        
        # 确保在范围内
        self._dependency_weight = np.clip(
            self._dependency_weight,
            self.config.dependency_weight_min,
            self.config.dependency_weight_max
        )
        
        # 更新统计
        self._stats["total_dependency_adjustments"] += 1
        self._stats["average_dependency_weight"] = np.mean(list(self._dependency_history))
        
        logger.info(
            f"Adjusted dependency weight: "
            f"performance={performance_metric:.4f}, "
            f"new_weight={self._dependency_weight:.4f}"
        )
    
    def set_dependency_weight(self, weight: float):
        """
        手动设置依赖权重
        
        Args:
            weight: 新的依赖权重
        """
        # 确保在范围内
        self._dependency_weight = np.clip(
            weight,
            self.config.dependency_weight_min,
            self.config.dependency_weight_max
        )
        
        logger.info(f"Set dependency weight to {self._dependency_weight:.4f}")
    
    def get_dependency_weight(self) -> float:
        """
        获取当前依赖权重
        
        Returns:
            当前依赖权重
        """
        return self._dependency_weight
    
    def start_ablation_test(self):
        """
        开始消融测试
        
        移除 L2 调控信号
        """
        self.ablation_tester.start_ablation()
    
    def end_ablation_test(self):
        """
        结束消融测试
        
        恢复 L2 调控信号
        """
        self.ablation_tester.end_ablation()
    
    def is_ablation_active(self) -> bool:
        """
        检查消融是否激活
        
        Returns:
            是否处于消融状态
        """
        return self.ablation_tester.is_ablation_active()
    
    def record_performance(
        self,
        performance_metric: float,
        is_pre_ablation: bool = True
    ):
        """
        记录性能指标
        
        Args:
            performance_metric: 性能指标值
            is_pre_ablation: 是否为消融前性能
        """
        self.ablation_tester.record_performance(performance_metric, is_pre_ablation)
    
    def validate_independence(self) -> Tuple[bool, Dict[str, Any]]:
        """
        验证独立性
        
        Returns:
            (is_valid, statistics): 是否独立以及统计信息
        """
        return self.ablation_tester.validate_independence()
    
    def get_retention_rate(self) -> float:
        """
        获取功能维持率
        
        Returns:
            功能维持率
        """
        return self.ablation_tester.get_latest_retention_rate()
    
    def enable_perturbation(self, enabled: bool):
        """
        启用或禁用扰动
        
        Args:
            enabled: 是否启用扰动
        """
        self.perturbation_trainer.enable_perturbation(enabled)
    
    def set_perturbation_sigma(self, sigma: float):
        """
        设置扰动噪声标准差
        
        Args:
            sigma: 新的噪声标准差
        """
        self.perturbation_trainer.set_noise_sigma(sigma)
        self.config.perturbation_noise_sigma = sigma
    
    def get_perturbation_statistics(self) -> Dict[str, Any]:
        """
        获取扰动统计信息
        
        Returns:
            扰动统计信息
        """
        return self.perturbation_trainer.get_statistics()
    
    def get_ablation_statistics(self) -> Dict[str, Any]:
        """
        获取消融统计信息
        
        Returns:
            消融统计信息
        """
        return self.ablation_tester.get_statistics()
    
    def reset(self):
        """重置元认知管理器"""
        # 重置扰动训练器
        self.perturbation_trainer.reset()
        
        # 重置消融测试器
        self.ablation_tester.reset()
        
        # 重置依赖权重
        self._dependency_weight = self.config.dependency_weight_default
        self._dependency_history.clear()
        
        # 重置统计
        self._stats = {
            "total_control_signals_processed": 0,
            "total_dependency_adjustments": 0,
            "average_dependency_weight": self._dependency_weight,
        }
        
        logger.info("MetaCognitiveManager reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        # 添加扰动统计
        stats["perturbation_stats"] = self.perturbation_trainer.get_statistics()
        
        # 添加消融统计
        stats["ablation_stats"] = self.ablation_tester.get_statistics()
        
        # 添加依赖信息
        stats["dependency_info"] = {
            "current_weight": self._dependency_weight,
            "weight_range": [
                self.config.dependency_weight_min,
                self.config.dependency_weight_max,
            ],
            "history_length": len(self._dependency_history),
        }
        
        # 添加配置信息
        stats["config"] = {
            "perturbation_noise_sigma": self.config.perturbation_noise_sigma,
            "perturbation_enabled": self.config.perturbation_enabled,
            "ablation_threshold": self.config.ablation_threshold,
            "ablation_enabled": self.config.ablation_enabled,
        }
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证元认知管理器
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证扰动参数
        if self.config.perturbation_noise_sigma < 0:
            errors.append(f"Invalid perturbation noise sigma: {self.config.perturbation_noise_sigma}")
        
        # 验证依赖权重范围
        if not 0 < self.config.dependency_weight_min < self.config.dependency_weight_max < 1:
            errors.append(
                f"Invalid dependency weight range: "
                f"[{self.config.dependency_weight_min}, {self.config.dependency_weight_max}]"
            )
        
        # 验证当前依赖权重
        if not self.config.dependency_weight_min <= self._dependency_weight <= self.config.dependency_weight_max:
            errors.append(
                f"Dependency weight {self._dependency_weight} out of range "
                f"[{self.config.dependency_weight_min}, {self.config.dependency_weight_max}]"
            )
        
        # 验证消融阈值
        if not 0 < self.config.ablation_threshold < 1:
            errors.append(f"Invalid ablation threshold: {self.config.ablation_threshold}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __repr__(self) -> str:
        return (
            f"MetaCognitiveManager("
            f"perturbation_sigma={self.config.perturbation_noise_sigma:.4f}, "
            f"dependency_weight={self._dependency_weight:.4f}, "
            f"ablation_threshold={self.config.ablation_threshold:.4f})"
        )


def create_meta_cognitive_manager_from_config(
    meta_config: MetaCognitiveConfig,
    control_signal_dim: int = 128,
    device: Optional[str] = None
) -> MetaCognitiveManager:
    """
    从配置创建元认知管理器
    
    Args:
        meta_config: 元认知配置
        control_signal_dim: 调控信号维度
        device: 计算设备
    
    Returns:
        MetaCognitiveManager 实例
    """
    config = MetaCognitiveManagerConfig(
        perturbation_noise_sigma=meta_config.l2_perturbation_noise,
        ablation_threshold=meta_config.l2_ablation_threshold,
    )
    
    manager = MetaCognitiveManager(
        config=config,
        meta_config=meta_config,
        control_signal_dim=control_signal_dim,
        device=device
    )
    
    return manager