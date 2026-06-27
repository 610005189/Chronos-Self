"""
非对称耦合与稳定性约束机制
==========================

实现快变量与慢变量之间的非对称耦合，以及系统稳定性监测与保障机制。

核心功能：
- 自适应耦合系数计算（基于快变量方差）
- 耦合强度上限裁剪（clip机制）
- 稳定性监测系统（防止发散）
- 边缘混沌稳态维持
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, List, TYPE_CHECKING
import logging
from dataclasses import dataclass, field
import math

from chronos_core.utils.config import CouplingStabilityConfig, DimensionalityConfig

if TYPE_CHECKING:
    from chronos_core.core.state_manager import StateManager

logger = logging.getLogger(__name__)


@dataclass
class CouplingConfig:
    """耦合机制配置"""

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512

    # 自适应耦合参数
    base_coupling_coeff: float = 0.01  # α_0 基础耦合系数
    adaptation_coeff: float = 1.0  # β 耦合适应性系数

    # 耦合强度限制
    coupling_upper_bound: float = 10.0  # 耦合系数上限
    coupling_lower_bound: float = 0.001  # 耦合系数下限
    slow_coupling_limit: float = 0.1  # 慢变量耦合系数上限（相对于慢变量时间尺度 τ_slow）

    # 稳定性参数
    stability_threshold: float = 1e6  # 状态范数阈值
    variance_threshold: float = 100.0  # 快变量方差阈值
    drift_threshold: float = 10.0  # 慢变量漂移阈值

    # Lyapunov指数阈值（边缘混沌）
    lyapunov_positive_threshold: float = 0.1  # 最大 Lyapunov 指数上限（防止发散）
    lyapunov_negative_threshold: float = -0.1  # 最小 Lyapunov 指数下限（防止冻结）

    # 监控间隔
    monitoring_interval: int = 100  # 每 N 步检查一次稳定性
    variance_window: int = 50  # 方差计算窗口大小
    drift_window: int = 100  # 漂移计算窗口大小

    # 稳定性保障措施
    auto_coupling_reduction: bool = True  # 自动降低耦合系数
    coupling_reduction_factor: float = 0.5  # 降低因子
    state_reset_threshold: float = 1e8  # 状态重置阈值


class AdaptiveCouplingCoefficients:
    """
    自适应耦合系数计算器

    实现基于快变量方差的自适应耦合系数计算：
        α(t) = α_0 · 1/(1 + β · Var(E_fast(t)))

    当快变量方差增大时，减小耦合系数以避免过度扰动慢变量。
    """

    def __init__(
        self,
        config: Optional[CouplingConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化自适应耦合系数计算器

        Args:
            config: 耦合配置
            coupling_config: 耦合稳定性配置（来自全局配置）
            dim_config: 维度配置
            device: 计算设备
        """
        self.config = config or CouplingConfig()

        if coupling_config:
            self.config.base_coupling_coeff = coupling_config.coupling_adaptation_coeff
            self.config.adaptation_coeff = coupling_config.coupling_adaptation_coeff
            self.config.stability_threshold = coupling_config.stability_threshold
            self.config.coupling_upper_bound = coupling_config.coupling_upper_bound

        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 当前耦合系数
        self.current_coupling = self.config.base_coupling_coeff

        # 方差历史（用于计算平均方差）
        self.variance_history: List[float] = []

        # 统计信息
        self.update_count = 0

        logger.info(
            f"AdaptiveCouplingCoefficients initialized: "
            f"base_coeff={self.config.base_coupling_coeff}, "
            f"adaptation={self.config.adaptation_coeff}"
        )

    def compute(
        self,
        E_fast: torch.Tensor,
        clip_upper_bound: bool = True
    ) -> float:
        """
        计算自适应耦合系数

        Args:
            E_fast: 快变量状态
            clip_upper_bound: 是否裁剪上限

        Returns:
            耦合系数 α(t)
        """
        # 计算快变量方差
        variance = torch.var(E_fast).item()

        # 记录方差历史
        self.variance_history.append(variance)
        if len(self.variance_history) > self.config.variance_window:
            self.variance_history = self.variance_history[-self.config.variance_window:]

        # 计算平均方差
        avg_variance = np.mean(self.variance_history) if self.variance_history else variance

        # 自适应耦合系数公式
        # α(t) = α_0 · 1/(1 + β · Var(E_fast(t)))
        coupling = self.config.base_coupling_coeff / (
            1 + self.config.adaptation_coeff * avg_variance
        )

        # 裁剪下限（防止过小）
        coupling = max(self.config.coupling_lower_bound, coupling)

        # 裁剪上限（防止过大）
        if clip_upper_bound:
            coupling = min(self.config.coupling_upper_bound, coupling)

        # 特殊裁剪：相对于慢变量时间尺度
        # coupling <= 1/τ_slow（τ_slow = slow_update_frequency）
        # 这里使用 slow_coupling_limit 作为上限
        coupling = min(self.config.slow_coupling_limit, coupling)

        # 更新当前耦合系数
        self.current_coupling = coupling

        # 统计
        self.update_count += 1

        logger.debug(
            f"Coupling coefficient computed: variance={variance:.4f}, "
            f"avg_variance={avg_variance:.4f}, coupling={coupling:.4f}"
        )

        return coupling

    def compute_with_clip(self, E_fast: torch.Tensor) -> Tuple[float, bool]:
        """
        计算耦合系数并检查是否需要裁剪

        Args:
            E_fast: 快变量状态

        Returns:
            (耦合系数, 是否被裁剪)
        """
        # 不裁剪的计算
        coupling_unclipped = self.config.base_coupling_coeff / (
            1 + self.config.adaptation_coeff * torch.var(E_fast).item()
        )

        # 裁剪后的计算
        coupling_clipped = self.compute(E_fast, clip_upper_bound=True)

        # 是否被裁剪
        is_clipped = coupling_unclipped > coupling_clipped

        return coupling_clipped, is_clipped

    def get_current_coupling(self) -> float:
        """获取当前耦合系数"""
        return self.current_coupling

    def set_coupling(self, value: float) -> None:
        """
        手动设置耦合系数

        Args:
            value: 新的耦合系数值
        """
        # 应用裁剪
        clipped_value = max(
            self.config.coupling_lower_bound,
            min(self.config.coupling_upper_bound, value)
        )
        self.current_coupling = clipped_value

        logger.info(f"Coupling coefficient manually set to: {clipped_value}")

    def reduce_coupling(self, factor: float = 0.5) -> None:
        """
        降低耦合系数（用于稳定性保障）

        Args:
            factor: 降低因子
        """
        new_coupling = self.current_coupling * factor
        self.set_coupling(new_coupling)

        logger.warning(
            f"Coupling coefficient reduced: "
            f"{self.current_coupling/factor:.4f} -> {new_coupling:.4f}"
        )

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        avg_variance = np.mean(self.variance_history) if self.variance_history else 0.0

        return {
            "current_coupling": self.current_coupling,
            "base_coupling": self.config.base_coupling_coeff,
            "adaptation_coeff": self.config.adaptation_coeff,
            "average_variance": avg_variance,
            "update_count": self.update_count,
            "variance_history_length": len(self.variance_history)
        }

    def reset(self) -> None:
        """重置耦合系数"""
        self.current_coupling = self.config.base_coupling_coeff
        self.variance_history.clear()
        self.update_count = 0

        logger.info("AdaptiveCouplingCoefficients reset")

    def __repr__(self) -> str:
        return f"AdaptiveCouplingCoefficients(current={self.current_coupling:.4f})"


class StabilityMonitor:
    """
    稳定性监测系统

    监测系统状态，防止发散，维持边缘混沌稳态。

    主要监测内容：
    1. 快变量方差是否过大
    2. 慢变量漂移是否过大
    3. 状态范数是否超出阈值
    4. Lyapunov 指数估计（边缘混沌）
    5. 自动采取稳定性保障措施
    """

    def __init__(
        self,
        config: Optional[CouplingConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化稳定性监测系统

        Args:
            config: 耦合配置
            coupling_config: 耦合稳定性配置
            device: 计算设备
        """
        self.config = config or CouplingConfig()

        if coupling_config:
            self.config.stability_threshold = coupling_config.stability_threshold
            self.config.variance_threshold = coupling_config.stability_threshold / 100
            self.config.lyapunov_positive_threshold = coupling_config.lyapunov_threshold

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 系统稳定性状态
        self.is_stable = True
        self.is_edge_of_chaos = True  # 边缘混沌状态

        # 监测历史
        self.fast_norm_history: List[float] = []
        self.slow_norm_history: List[float] = []
        self.fast_variance_history: List[float] = []
        self.slow_drift_history: List[float] = []

        # Lyapunov 指数估计
        self.lyapunov_estimate: Optional[float] = None

        # 监测计数器
        self.monitoring_count = 0
        self.warning_count = 0
        self.critical_count = 0

        # 最近的状态轨迹（用于 Lyapunov 计算）
        self.state_trajectory: List[torch.Tensor] = []

        logger.info(
            f"StabilityMonitor initialized: "
            f"threshold={self.config.stability_threshold}, "
            f"variance_threshold={self.config.variance_threshold}"
        )

    def monitor(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        E_fast_prev: Optional[torch.Tensor] = None,
        E_slow_prev: Optional[torch.Tensor] = None,
        step_count: int = 0
    ) -> Dict:
        """
        执行稳定性监测

        Args:
            E_fast: 当前快变量状态
            E_slow: 当前慢变量状态
            E_fast_prev: 上一时刻快变量（可选）
            E_slow_prev: 上一时刻慢变量（可选）
            step_count: 当前步数

        Returns:
            监测结果字典
        """
        # 确保输入在设备上
        E_fast = E_fast.to(self.device)
        E_slow = E_slow.to(self.device)

        # 检查频率（不需要每步都监测）
        if step_count % self.config.monitoring_interval != 0:
            return {"skip": True}

        # 初始化监测结果
        result = {
            "is_stable": True,
            "warnings": [],
            "critical_issues": [],
            "actions_taken": []
        }

        # 1. 检查快变量范数和方差
        fast_norm = torch.norm(E_fast).item()
        fast_variance = torch.var(E_fast).item()

        self.fast_norm_history.append(fast_norm)
        self.fast_variance_history.append(fast_variance)

        if len(self.fast_norm_history) > self.config.variance_window:
            self.fast_norm_history = self.fast_norm_history[-self.config.variance_window:]
            self.fast_variance_history = self.fast_variance_history[-self.config.variance_window:]

        # 快变量稳定性检查
        if fast_norm > self.config.stability_threshold:
            result["warnings"].append(f"Fast variable norm too large: {fast_norm:.4e}")
            result["is_stable"] = False
            self.warning_count += 1

        if fast_variance > self.config.variance_threshold:
            result["warnings"].append(f"Fast variable variance too large: {fast_variance:.4e}")
            # 建议降低耦合系数
            result["actions_taken"].append("reduce_coupling")

        # 2. 检查慢变量范数和漂移
        slow_norm = torch.norm(E_slow).item()
        self.slow_norm_history.append(slow_norm)

        if len(self.slow_norm_history) > self.config.drift_window:
            self.slow_norm_history = self.slow_norm_history[-self.config.drift_window:]

        # 计算慢变量漂移（如果有历史数据）
        if E_slow_prev is not None:
            slow_drift = torch.norm(E_slow - E_slow_prev.to(self.device)).item()
            self.slow_drift_history.append(slow_drift)

            if len(self.slow_drift_history) > self.config.drift_window:
                self.slow_drift_history = self.slow_drift_history[-self.config.drift_window:]

            if slow_drift > self.config.drift_threshold:
                result["warnings"].append(f"Slow variable drift too large: {slow_drift:.4e}")
                result["is_stable"] = False

        # 3. 检查 NaN 和 Inf
        if torch.isnan(E_fast).any() or torch.isinf(E_fast).any():
            result["critical_issues"].append("NaN or Inf in fast variable")
            result["is_stable"] = False
            result["actions_taken"].append("reset_state")
            self.critical_count += 1

        if torch.isnan(E_slow).any() or torch.isinf(E_slow).any():
            result["critical_issues"].append("NaN or Inf in slow variable")
            result["is_stable"] = False
            result["actions_taken"].append("reset_state")
            self.critical_count += 1

        # 4. Lyapunov 指数估计（边缘混沌检查）
        if E_fast_prev is not None:
            # 记录状态轨迹
            self.state_trajectory.append(E_fast.clone())

            if len(self.state_trajectory) > self.config.variance_window:
                self.state_trajectory = self.state_trajectory[-self.config.variance_window:]

            # 计算 Lyapunov 指数估计
            lyapunov = self._estimate_lyapunov_exponent()

            if lyapunov is not None:
                self.lyapunov_estimate = lyapunov

                # 边缘混沌检查
                if lyapunov > self.config.lyapunov_positive_threshold:
                    result["warnings"].append(f"System too chaotic: λ={lyapunov:.4f}")
                    result["actions_taken"].append("reduce_coupling")
                    self.is_edge_of_chaos = False

                elif lyapunov < self.config.lyapunov_negative_threshold:
                    result["warnings"].append(f"System too stable (frozen): λ={lyapunov:.4f}")
                    result["actions_taken"].append("increase_coupling")
                    self.is_edge_of_chaos = False

                else:
                    # 在边缘混沌区间内
                    self.is_edge_of_chaos = True

        # 更新稳定性状态
        self.is_stable = result["is_stable"]
        self.monitoring_count += 1

        # 记录日志
        if result["warnings"]:
            logger.warning(f"Stability warnings: {result['warnings']}")
        if result["critical_issues"]:
            logger.error(f"Critical issues: {result['critical_issues']}")

        return result

    def _estimate_lyapunov_exponent(self) -> Optional[float]:
        """
        估计最大 Lyapunov 指数

        使用简单的距离增长方法：
            λ ≈ (1/Δt) · log(||ΔE(t)|| / ||ΔE(0)||)

        Args:
            trajectory: 状态轨迹

        Returns:
            Lyapunov 指数估计值（如果计算失败返回 None）
        """
        if len(self.state_trajectory) < 10:
            return None

        try:
            # 计算相邻状态的距离变化
            distances = []
            for i in range(1, len(self.state_trajectory)):
                delta = torch.norm(
                    self.state_trajectory[i] - self.state_trajectory[i-1]
                ).item()
                distances.append(delta)

            if len(distances) < 2 or distances[0] == 0:
                return None

            # 使用平均增长率估计 Lyapunov 指数
            # λ ≈ mean(log(d_{i+1} / d_i)) / Δt
            growth_rates = []
            for i in range(1, len(distances)):
                if distances[i-1] > 0 and distances[i] > 0:
                    rate = math.log(distances[i] / distances[i-1])
                    growth_rates.append(rate)

            if not growth_rates:
                return None

            # 平均增长率（假设时间步长为 1）
            lyapunov = np.mean(growth_rates)

            return lyapunov

        except Exception as e:
            logger.debug(f"Lyapunov estimation failed: {e}")
            return None

    def check_edge_of_chaos(self) -> bool:
        """
        检查系统是否处于边缘混沌稳态

        Returns:
            是否处于边缘混沌
        """
        return self.is_edge_of_chaos

    def get_stability_status(self) -> Dict:
        """
        获取完整的稳定性状态报告

        Returns:
            状态报告字典
        """
        avg_fast_norm = np.mean(self.fast_norm_history) if self.fast_norm_history else 0.0
        avg_slow_norm = np.mean(self.slow_norm_history) if self.slow_norm_history else 0.0
        avg_variance = np.mean(self.fast_variance_history) if self.fast_variance_history else 0.0
        avg_drift = np.mean(self.slow_drift_history) if self.slow_drift_history else 0.0

        return {
            "is_stable": self.is_stable,
            "is_edge_of_chaos": self.is_edge_of_chaos,
            "monitoring_count": self.monitoring_count,
            "warning_count": self.warning_count,
            "critical_count": self.critical_count,
            "lyapunov_estimate": self.lyapunov_estimate,
            "average_fast_norm": avg_fast_norm,
            "average_slow_norm": avg_slow_norm,
            "average_variance": avg_variance,
            "average_drift": avg_drift,
            "current_fast_norm": self.fast_norm_history[-1] if self.fast_norm_history else 0.0,
            "current_slow_norm": self.slow_norm_history[-1] if self.slow_norm_history else 0.0
        }

    def reset(self) -> None:
        """重置监测系统"""
        self.is_stable = True
        self.is_edge_of_chaos = True
        self.fast_norm_history.clear()
        self.slow_norm_history.clear()
        self.fast_variance_history.clear()
        self.slow_drift_history.clear()
        self.state_trajectory.clear()
        self.lyapunov_estimate = None
        self.monitoring_count = 0
        self.warning_count = 0
        self.critical_count = 0

        logger.info("StabilityMonitor reset")

    def __repr__(self) -> str:
        status = "stable" if self.is_stable else "unstable"
        chaos = "edge-of-chaos" if self.is_edge_of_chaos else "off-edge"
        return f"StabilityMonitor(status={status}, chaos={chaos})"


class CouplingAndStabilitySystem:
    """
    非对称耦合与稳定性保障系统

    整合自适应耦合系数计算和稳定性监测，
    提供完整的耦合与稳定性保障机制。

    主要功能：
    1. 自适应耦合系数计算
    2. 耦合强度裁剪
    3. 稳定性监测
    4. 自动稳定性保障措施
    5. 边缘混沌稳态维持

    使用示例：
        system = CouplingAndStabilitySystem(config=CouplingConfig())
        system.initialize()

        # 每步调用
        coupling = system.update_coupling(E_fast)
        stability_report = system.monitor_stability(E_fast, E_slow, step_count)

        # 根据报告采取行动
        if stability_report["actions_taken"]:
            system.apply_actions(stability_report["actions_taken"])
    """

    def __init__(
        self,
        config: Optional[CouplingConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化耦合与稳定性系统

        Args:
            config: 耦合配置
            coupling_config: 耦合稳定性配置
            dim_config: 维度配置
            device: 计算设备
        """
        self.config = config or CouplingConfig()

        if coupling_config:
            self.config.base_coupling_coeff = coupling_config.coupling_adaptation_coeff
            self.config.adaptation_coeff = coupling_config.coupling_adaptation_coeff
            self.config.stability_threshold = coupling_config.stability_threshold
            self.config.coupling_upper_bound = coupling_config.coupling_upper_bound
            self.config.lyapunov_positive_threshold = coupling_config.lyapunov_threshold

        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 核心组件
        self.coupling_calculator: Optional[AdaptiveCouplingCoefficients] = None
        self.stability_monitor: Optional[StabilityMonitor] = None

        # 初始化标志
        self._initialized = False

        logger.info(
            f"CouplingAndStabilitySystem created: "
            f"device={self.device}"
        )

    def initialize(self) -> None:
        """初始化系统"""
        # 创建自适应耦合系数计算器
        self.coupling_calculator = AdaptiveCouplingCoefficients(
            config=self.config,
            device=self.device
        )

        # 创建稳定性监测器
        self.stability_monitor = StabilityMonitor(
            config=self.config,
            device=self.device
        )

        self._initialized = True
        logger.info("CouplingAndStabilitySystem initialized")

    def update_coupling(self, E_fast: torch.Tensor) -> float:
        """
        更新耦合系数

        Args:
            E_fast: 快变量状态

        Returns:
            新的耦合系数
        """
        if not self._initialized:
            raise ValueError("System not initialized.")

        return self.coupling_calculator.compute(E_fast)

    def monitor_stability(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        E_fast_prev: Optional[torch.Tensor] = None,
        E_slow_prev: Optional[torch.Tensor] = None,
        step_count: int = 0
    ) -> Dict:
        """
        监测系统稳定性

        Args:
            E_fast: 当前快变量
            E_slow: 当前慢变量
            E_fast_prev: 上一时刻快变量
            E_slow_prev: 上一时刻慢变量
            step_count: 当前步数

        Returns:
            监测报告
        """
        if not self._initialized:
            raise ValueError("System not initialized.")

        return self.stability_monitor.monitor(
            E_fast, E_slow,
            E_fast_prev, E_slow_prev,
            step_count
        )

    def apply_stability_actions(
        self,
        actions: List[str],
        state_manager: Optional["StateManager"] = None
    ) -> None:
        """
        应用稳定性保障措施

        Args:
            actions: 需要采取的行动列表
            state_manager: 状态管理器（用于重置状态）
        """
        for action in actions:
            if action == "reduce_coupling":
                self.coupling_calculator.reduce_coupling(
                    self.config.coupling_reduction_factor
                )

            elif action == "increase_coupling":
                # 适当增加耦合系数
                current = self.coupling_calculator.get_current_coupling()
                new_coupling = min(
                    current * 1.2,
                    self.config.coupling_upper_bound
                )
                self.coupling_calculator.set_coupling(new_coupling)

            elif action == "reset_state":
                logger.critical("System state reset triggered!")
                if state_manager is not None and hasattr(state_manager, 'reset'):
                    state_manager.reset()

    def check_edge_of_chaos(self) -> bool:
        """
        检查边缘混沌稳态

        Returns:
            是否处于边缘混沌
        """
        if not self._initialized:
            return True

        return self.stability_monitor.check_edge_of_chaos()

    def get_current_coupling(self) -> float:
        """获取当前耦合系数"""
        if not self._initialized:
            return self.config.base_coupling_coeff

        return self.coupling_calculator.get_current_coupling()

    def get_stability_report(self) -> Dict:
        """获取稳定性报告"""
        if not self._initialized:
            return {"status": "not_initialized"}

        report = {
            "coupling": self.coupling_calculator.get_statistics(),
            "stability": self.stability_monitor.get_stability_status()
        }

        return report

    def reset(self) -> None:
        """重置系统"""
        if self.coupling_calculator:
            self.coupling_calculator.reset()
        if self.stability_monitor:
            self.stability_monitor.reset()

        logger.info("CouplingAndStabilitySystem reset")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        coupling = self.get_current_coupling()
        return (
            f"CouplingAndStabilitySystem(status={status}, "
            f"coupling={coupling:.4f})"
        )


def create_coupling_system_from_config(
    coupling_config: CouplingStabilityConfig,
    dim_config: DimensionalityConfig,
    device: Optional[str] = None
) -> CouplingAndStabilitySystem:
    """
    从全局配置创建耦合与稳定性系统

    Args:
        coupling_config: 耦合稳定性配置
        dim_config: 维度配置
        device: 计算设备

    Returns:
        CouplingAndStabilitySystem 实例
    """
    config = CouplingConfig(
        fast_dim=dim_config.fast_variable_dim,
        slow_dim=dim_config.slow_variable_dim,
        base_coupling_coeff=coupling_config.coupling_adaptation_coeff,
        adaptation_coeff=coupling_config.coupling_adaptation_coeff,
        coupling_upper_bound=coupling_config.coupling_upper_bound,
        stability_threshold=coupling_config.stability_threshold,
        lyapunov_positive_threshold=coupling_config.lyapunov_threshold
    )

    system = CouplingAndStabilitySystem(
        config=config,
        coupling_config=coupling_config,
        dim_config=dim_config,
        device=device
    )

    system.initialize()
    return system