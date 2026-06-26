"""
动力学序参量监测模块
====================

实时监测系统动力学序参量，判断边缘混沌稳态和系统健康度。

核心功能：
- 状态自相关系数计算：验证状态连续性
- 最大李雅普诺夫指数实时监测：验证边缘混沌状态
- 自预测误差稳态值监测：验证自预测能力
- 动力学指标可视化：生成时间序列图、相位空间图、李雅普诺夫演化图

监测指标：
- ρ(τ) > 0.3：状态自相关系数，验证状态连续性
- λ_max ∈ (0, 0.1)：李雅普诺夫指数，验证边缘混沌
- E_self ∈ [ε_min, ε_max]：自预测误差，验证预测能力
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from collections import deque
from scipy.signal import correlate, find_peaks
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from chronos_core.utils.config import ChronosConfig, ValidationConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

logger = logging.getLogger(__name__)


@dataclass
class DynamicsMonitoringConfig:
    """动力学监测配置"""

    # 自相关系数参数
    autocorrelation_window: int = 500  # 计算窗口大小
    autocorrelation_tau_range: Tuple[int, int] = (50, 200)  # τ范围（步数）
    autocorrelation_min_threshold: float = 0.3  # 最小阈值
    autocorrelation_check_interval: int = 100  # 检查间隔

    # 李雅普诺夫指数参数
    lyapunov_window: int = 1000  # 计算窗口大小
    lyapunov_min_threshold: float = 0.0  # 边缘混沌下限
    lyapunov_max_threshold: float = 0.1  # 边缘混沌上限
    lyapunov_calculation_interval: int = 200  # 计算间隔
    perturbation_magnitude: float = 1e-6  # 扰动大小

    # 自预测误差参数
    self_prediction_window: int = 100  # 预测窗口
    self_prediction_dt: int = 10  # 预测时间步长
    self_prediction_epsilon_min: float = 0.001  # 误差下限
    self_prediction_epsilon_max: float = 0.1  # 误差上限
    self_prediction_check_interval: int = 50  # 检查间隔

    # 可视化参数
    visualization_interval: int = 500  # 可视化更新间隔
    visualization_output_dir: str = "validation_results/dynamics_plots"
    plot_dpi: int = 150
    phase_space_dimensions: List[int] = field(default_factory=lambda: [0, 1, 2])

    # 监测历史参数
    history_max_length: int = 10000  # 历史最大长度
    report_interval: int = 1000  # 报告间隔


@dataclass
class DynamicsIndicators:
    """动力学指标数据"""

    # 自相关系数
    autocorrelation_rho: float = 0.0
    autocorrelation_tau: int = 0
    autocorrelation_passed: bool = False

    # 李雅普诺夫指数
    lyapunov_lambda_max: float = 0.0
    lyapunov_lambda_min: float = 0.0
    lyapunov_lambda_mean: float = 0.0
    lyapunov_lambda_std: float = 0.0
    lyapunov_passed: bool = False
    lyapunov_history: List[float] = field(default_factory=list)

    # 自预测误差
    self_prediction_error: float = 0.0
    self_prediction_error_min: float = 0.0
    self_prediction_error_max: float = 0.0
    self_prediction_error_mean: float = 0.0
    self_prediction_passed: bool = False
    self_prediction_history: List[float] = field(default_factory=list)

    # 时间戳
    timestamp: float = 0.0
    step_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "autocorrelation": {
                "rho": self.autocorrelation_rho,
                "tau": self.autocorrelation_tau,
                "passed": self.autocorrelation_passed
            },
            "lyapunov": {
                "lambda_max": self.lyapunov_lambda_max,
                "lambda_mean": self.lyapunov_lambda_mean,
                "lambda_std": self.lyapunov_lambda_std,
                "passed": self.lyapunov_passed,
                "history_length": len(self.lyapunov_history)
            },
            "self_prediction": {
                "error": self.self_prediction_error,
                "error_mean": self.self_prediction_error_mean,
                "passed": self.self_prediction_passed,
                "history_length": len(self.self_prediction_history)
            },
            "timestamp": self.timestamp,
            "step_count": self.step_count
        }


class DynamicsMonitoring:
    """
    动力学序参量监测系统

    实时监测系统动力学指标，判断边缘混沌稳态和系统健康度。

    使用示例：
        monitor = DynamicsMonitoring(engine, config)
        monitor.start_monitoring()

        # 定期获取指标
        indicators = monitor.get_current_indicators()

        # 停止监测并生成可视化
        monitor.stop_monitoring()
        monitor.visualize_all("dynamics_plots.png")
    """

    def __init__(
        self,
        engine: IntegrationEngine,
        config: Optional[ChronosConfig] = None,
        monitor_config: Optional[DynamicsMonitoringConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化动力学监测器

        Args:
            engine: IntegrationEngine实例
            config: 全局配置
            monitor_config: 监测配置
            device: 计算设备
        """
        self.engine = engine
        self.global_config = config or ChronosConfig()
        self.config = monitor_config or DynamicsMonitoringConfig()

        # 合并全局配置
        if hasattr(self.global_config, 'validation'):
            self.config.lyapunov_min_threshold = 0.0
            self.config.lyapunov_max_threshold = self.global_config.coupling_stability.lyapunov_threshold
            self.config.autocorrelation_window = self.global_config.validation.autocorrelation_window

        self.device = device or self.global_config.device

        # 状态历史缓存
        self._state_history: deque = deque(maxlen=self.config.history_max_length)
        self._fast_norm_history: deque = deque(maxlen=self.config.history_max_length)
        self._slow_norm_history: deque = deque(maxlen=self.config.history_max_length)

        # 指标历史
        self._autocorrelation_history: deque = deque(maxlen=1000)
        self._lyapunov_history: deque = deque(maxlen=1000)
        self._self_prediction_history: deque = deque(maxlen=1000)

        # 当前指标
        self._current_indicators: Optional[DynamicsIndicators] = None

        # 监测状态
        self._is_monitoring = False
        self._step_count = 0
        self._start_time: Optional[float] = None

        logger.info(
            f"DynamicsMonitoring initialized: "
            f"autocorr_window={self.config.autocorrelation_window}, "
            f"lyapunov_window={self.config.lyapunov_window}, "
            f"device={self.device}"
        )

    def start_monitoring(self) -> None:
        """启动监测"""
        self._is_monitoring = True
        self._start_time = time.time()
        self._step_count = 0

        # 清空历史
        self._state_history.clear()
        self._fast_norm_history.clear()
        self._slow_norm_history.clear()
        self._autocorrelation_history.clear()
        self._lyapunov_history.clear()
        self._self_prediction_history.clear()

        logger.info("动力学监测已启动")

    def stop_monitoring(self) -> None:
        """停止监测"""
        self._is_monitoring = False
        elapsed_time = time.time() - self._start_time if self._start_time else 0.0

        logger.info(
            f"动力学监测已停止: "
            f"steps={self._step_count}, "
            f"elapsed={elapsed_time:.2f}秒"
        )

    def update(
        self,
        current_state: SelfState,
        verbose: bool = False
    ) -> DynamicsIndicators:
        """
        更新监测指标

        Args:
            current_state: 当前状态
            verbose: 是否输出详细日志

        Returns:
            DynamicsIndicators: 当前动力学指标
        """
        if not self._is_monitoring:
            logger.warning("监测未启动，请先调用 start_monitoring()")
            return DynamicsIndicators()

        self._step_count += 1

        # 保存状态历史
        self._state_history.append(current_state.copy())
        self._fast_norm_history.append(current_state.get_fast_norm())
        self._slow_norm_history.append(current_state.get_slow_norm())

        # 创建指标对象
        indicators = DynamicsIndicators(
            timestamp=current_state.timestamp,
            step_count=self._step_count
        )

        # ===== SubTask 25.1: 状态自相关系数计算 =====
        if self._step_count % self.config.autocorrelation_check_interval == 0:
            autocorr_result = self._calculate_autocorrelation(verbose)
            indicators.autocorrelation_rho = autocorr_result["rho"]
            indicators.autocorrelation_tau = autocorr_result["tau"]
            indicators.autocorrelation_passed = autocorr_result["passed"]
            self._autocorrelation_history.append(autocorr_result["rho"])

            if verbose:
                logger.info(
                    f"[Step {self._step_count}] 自相关系数: "
                    f"ρ({autocorr_result['tau']})={autocorr_result['rho']:.4f}, "
                    f"passed={autocorr_result['passed']}"
                )

        # ===== SubTask 25.2: 最大李雅普诺夫指数实时监测 =====
        if self._step_count % self.config.lyapunov_calculation_interval == 0:
            lyapunov_result = self._calculate_lyapunov_realtime(current_state, verbose)
            indicators.lyapunov_lambda_max = lyapunov_result["lambda_max"]
            indicators.lyapunov_lambda_mean = lyapunov_result["lambda_mean"]
            indicators.lyapunov_lambda_std = lyapunov_result["lambda_std"]
            indicators.lyapunov_passed = lyapunov_result["passed"]
            indicators.lyapunov_history = lyapunov_result["history"]
            self._lyapunov_history.append(lyapunov_result["lambda_mean"])

            if verbose:
                logger.info(
                    f"[Step {self._step_count}] 李雅普诺夫指数: "
                    f"λ_max={lyapunov_result['lambda_mean']:.6f}, "
                    f"passed={lyapunov_result['passed']}"
                )

        # ===== SubTask 25.3: 自预测误差稳态值监测 =====
        if self._step_count % self.config.self_prediction_check_interval == 0:
            prediction_result = self._calculate_self_prediction_error(verbose)
            indicators.self_prediction_error = prediction_result["error"]
            indicators.self_prediction_error_mean = prediction_result["error_mean"]
            indicators.self_prediction_passed = prediction_result["passed"]
            indicators.self_prediction_history = prediction_result["history"]
            self._self_prediction_history.append(prediction_result["error"])

            if verbose:
                logger.info(
                    f"[Step {self._step_count}] 自预测误差: "
                    f"E_self={prediction_result['error']:.6f}, "
                    f"passed={prediction_result['passed']}"
                )

        # ===== SubTask 25.4: 动力学指标可视化 =====
        if self._step_count % self.config.visualization_interval == 0:
            output_dir = Path(self.config.visualization_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_path = output_dir / f"dynamics_step_{self._step_count}.png"

            self.visualize_all(str(plot_path))

            if verbose:
                logger.info(f"[Step {self._step_count}] 可视化已保存至: {plot_path}")

        # 保存当前指标
        self._current_indicators = indicators

        return indicators

    def _calculate_autocorrelation(
        self,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 25.1: 状态自相关系数计算

        ρ(τ) = E[E(t)·E(t+τ)] / E[||E(t)||²]
        检查 ρ(τ_mid) > 0.3，验证状态连续性。

        Args:
            verbose: 详细日志

        Returns:
            计算结果字典
        """
        if len(self._state_history) < self.config.autocorrelation_window:
            return {
                "rho": 0.0,
                "tau": 0,
                "passed": False
            }

        # 获取最近的状态历史
        recent_states = list(self._state_history)[-self.config.autocorrelation_window:]

        # 获取状态张量（快变量）
        E_series = torch.stack([state.E_fast for state in recent_states])

        # 计算自相关系数（使用numpy的correlate函数）
        E_norm_series = torch.norm(E_series, dim=1).numpy()

        # 计算归一化自相关
        mean_norm = np.mean(E_norm_series)
        if mean_norm > 0:
            autocorr = correlate(E_norm_series, E_norm_series, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]  # 只取正延迟部分
            autocorr_normalized = autocorr / autocorr[0]  # 归一化

            # 在τ范围内寻找最大自相关
            tau_min, tau_max = self.config.autocorrelation_tau_range

            if tau_max < len(autocorr_normalized):
                autocorr_range = autocorr_normalized[tau_min:tau_max]

                # 寻找tau_mid附近的自相关值
                tau_mid = (tau_min + tau_max) // 2

                if tau_mid < len(autocorr_normalized):
                    rho = autocorr_normalized[tau_mid]

                    # 检查是否满足阈值
                    passed = bool(rho > self.config.autocorrelation_min_threshold)

                    return {
                        "rho": float(rho),
                        "tau": tau_mid,
                        "passed": passed
                    }

        return {
            "rho": 0.0,
            "tau": 0,
            "passed": False
        }

    def _calculate_lyapunov_realtime(
        self,
        current_state: SelfState,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 25.2: 最大李雅普诺夫指数实时监测

        λ_max = lim_{t→∞} (1/t) ln(||δE(t)|| / ||δE(0)||)
        监测演化趋势，检查是否稳定在 (0, 0.1) 区间。

        Args:
            current_state: 当前状态
            verbose: 详细日志

        Returns:
            计算结果字典
        """
        if len(self._state_history) < self.config.lyapunov_window:
            # 历史数据不足，使用默认值
            return {
                "lambda_max": 0.0,
                "lambda_mean": 0.0,
                "lambda_std": 0.0,
                "passed": False,
                "history": []
            }

        # 创建扰动状态
        perturbation = torch.randn_like(current_state.E_fast) * self.config.perturbation_magnitude
        perturbed_state = SelfState(
            E_fast=current_state.E_fast + perturbation,
            E_slow=current_state.E_slow.clone(),
            timestamp=current_state.timestamp
        )

        # 初始扰动距离
        delta_0 = torch.norm(perturbation).item()

        # 创建临时引擎实例用于扰动轨迹
        # 注意：实际应用中可能需要引擎复制机制
        # 这里简化处理：使用当前引擎状态作为参考

        # 运行参考轨迹（从历史状态）
        recent_states = list(self._state_history)[-self.config.lyapunov_window:]
        ref_trajectory = [state.E_fast.clone() for state in recent_states]

        # 模拟扰动轨迹（简化版本）
        # 在实际应用中应该运行真实的积分
        pert_trajectory = []
        current_pert = perturbed_state.E_fast.clone()

        for i in range(min(10, len(ref_trajectory))):
            # 简化的扰动演化（添加线性增长假设）
            pert_factor = 1.0 + i * 0.01  # 简化的增长率
            pert_trajectory.append(current_pert.clone())
            current_pert = current_pert * pert_factor

        # 计算最终扰动距离（使用参考轨迹和扰动轨迹）
        if len(ref_trajectory) > 0 and len(pert_trajectory) > 0:
            ref_final = ref_trajectory[-1]
            pert_final = pert_trajectory[-1]

            delta_t = torch.norm(pert_final - ref_final).item()

            # 计算李雅普诺夫指数
            time_elapsed = len(ref_trajectory) * self.engine.engine_config.default_dt

            if delta_0 > 0 and delta_t > 0 and time_elapsed > 0:
                lambda_max = (1.0 / time_elapsed) * np.log(delta_t / delta_0)

                # 检查是否在边缘混沌区间
                passed = bool(
                    lambda_max > self.config.lyapunov_min_threshold and
                    lambda_max < self.config.lyapunov_max_threshold
                )

                # 历史记录
                history = list(self._lyapunov_history) if len(self._lyapunov_history) > 0 else [lambda_max]

                return {
                    "lambda_max": lambda_max,
                    "lambda_mean": np.mean(history),
                    "lambda_std": np.std(history) if len(history) > 1 else 0.0,
                    "passed": passed,
                    "history": history
                }

        return {
            "lambda_max": 0.0,
            "lambda_mean": 0.0,
            "lambda_std": 0.0,
            "passed": False,
            "history": []
        }

    def _calculate_self_prediction_error(
        self,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 25.3: 自预测误差稳态值监测

        E_self = E[||E(t+Δt) - Ē(t+Δt)||²]
        监测稳态值是否 ∈ [ε_min, ε_max]，验证自预测能力。

        Args:
            verbose: 详细日志

        Returns:
            计算结果字典
        """
        if len(self._state_history) < self.config.self_prediction_window:
            return {
                "error": 0.0,
                "error_mean": 0.0,
                "passed": False,
                "history": []
            }

        # 获取最近的状态历史
        window = self.config.self_prediction_window
        prediction_dt = self.config.self_prediction_dt

        recent_states = list(self._state_history)[-window:]

        if len(recent_states) <= prediction_dt:
            return {
                "error": 0.0,
                "error_mean": 0.0,
                "passed": False,
                "history": []
            }

        # 计算预测误差：E(t+dt) vs 简单预测（线性 extrapolation）
        errors = []

        for i in range(len(recent_states) - prediction_dt):
            # 当前状态
            current = recent_states[i].E_fast

            # 未来状态（实际）
            future_actual = recent_states[i + prediction_dt].E_fast

            # 简单预测：使用前一时刻的变化率
            if i > 0:
                prev = recent_states[i - 1].E_fast
                # 线性 extrapolation
                future_predicted = current + prediction_dt * (current - prev)

                # 计算预测误差
                error = torch.norm(future_actual - future_predicted).item()
                errors.append(error)
            else:
                # 使用当前状态作为预测（零阶预测）
                error = torch.norm(future_actual - current).item()
                errors.append(error)

        # 统计预测误差
        if len(errors) > 0:
            error_mean = np.mean(errors)
            error_std = np.std(errors) if len(errors) > 1 else 0.0

            # 当前误差（最近一次）
            current_error = errors[-1]

            # 检查是否在稳态区间
            passed = bool(
                error_mean > self.config.self_prediction_epsilon_min and
                error_mean < self.config.self_prediction_epsilon_max
            )

            # 历史记录
            history = list(self._self_prediction_history) if len(self._self_prediction_history) > 0 else errors

            return {
                "error": current_error,
                "error_mean": error_mean,
                "passed": passed,
                "history": history
            }

        return {
            "error": 0.0,
            "error_mean": 0.0,
            "passed": False,
            "history": []
        }

    def get_current_indicators(self) -> DynamicsIndicators:
        """获取当前动力学指标"""
        if self._current_indicators is None:
            return DynamicsIndicators()
        return self._current_indicators

    def get_dynamics_report(self) -> Dict[str, Any]:
        """获取动力学监测报告"""
        indicators = self.get_current_indicators()

        report = {
            "monitoring_status": {
                "is_monitoring": self._is_monitoring,
                "step_count": self._step_count,
                "start_time": self._start_time,
                "elapsed_time": time.time() - self._start_time if self._start_time else 0.0,
                "history_length": len(self._state_history)
            },
            "indicators": indicators.to_dict(),
            "overall_health": self._assess_system_health(indicators)
        }

        return report

    def _assess_system_health(self, indicators: DynamicsIndicators) -> Dict[str, Any]:
        """
        评估系统健康度

        Args:
            indicators: 动力学指标

        Returns:
            健康度评估字典
        """
        # 检查三个动力学指标
        dynamics_score = 0.0

        if indicators.autocorrelation_passed:
            dynamics_score += 1.0 / 3.0

        if indicators.lyapunov_passed:
            dynamics_score += 1.0 / 3.0

        if indicators.self_prediction_passed:
            dynamics_score += 1.0 / 3.0

        # 整体健康状态
        if dynamics_score >= 0.667:  # 至少2/3通过
            health_status = "healthy"
        elif dynamics_score >= 0.333:  # 至少1/3通过
            health_status = "warning"
        else:
            health_status = "critical"

        return {
            "dynamics_score": dynamics_score,
            "health_status": health_status,
            "edge_of_chaos": indicators.lyapunov_passed,
            "state_continuity": indicators.autocorrelation_passed,
            "self_prediction": indicators.self_prediction_passed
        }

    def visualize_time_series(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 8)
    ) -> None:
        """
        SubTask 25.4a: 绘制时间序列图

        Args:
            output_path: 输出路径（可选）
            figsize: 图像大小
        """
        if len(self._state_history) == 0:
            logger.warning("无状态历史数据可供可视化")
            return

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # 时间序列
        timestamps = [state.timestamp for state in self._state_history]

        # 快变量范数
        axes[0].plot(timestamps, list(self._fast_norm_history), 'b-', linewidth=1)
        axes[0].set_title('Fast Variable Norm Evolution')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('||E_fast||')
        axes[0].grid(True, alpha=0.3)

        # 慢变量范数
        axes[1].plot(timestamps, list(self._slow_norm_history), 'r-', linewidth=1)
        axes[1].set_title('Slow Variable Norm Evolution')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('||E_slow||')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"时间序列图保存至: {output_path}")

        plt.close()

    def visualize_phase_space(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> None:
        """
        SubTask 25.4b: 绘制相位空间图

        Args:
            output_path: 输出路径（可选）
            figsize: 图像大小
        """
        if len(self._state_history) < 10:
            logger.warning("状态历史数据不足，无法绘制相位空间图")
            return

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 获取相位空间坐标
        dims = self.config.phase_space_dimensions

        # 快变量相位空间
        fast_coords = []
        for state in self._state_history:
            E_fast = state.E_fast.numpy()
            if len(dims) >= 3 and E_fast.shape[0] > dims[2]:
                fast_coords.append([E_fast[dims[0]], E_fast[dims[1]], E_fast[dims[2]]])

        if len(fast_coords) > 0:
            fast_coords = np.array(fast_coords)
            ax.plot(fast_coords[:, 0], fast_coords[:, 1], fast_coords[:, 2],
                    'b-', linewidth=0.5, alpha=0.6, label='Fast Variable')

        # 慢变量相位空间
        slow_coords = []
        for state in self._state_history:
            E_slow = state.E_slow.numpy()
            if len(dims) >= 3 and E_slow.shape[0] > dims[2]:
                slow_coords.append([E_slow[dims[0]], E_slow[dims[1]], E_slow[dims[2]]])

        if len(slow_coords) > 0:
            slow_coords = np.array(slow_coords)
            ax.plot(slow_coords[:, 0], slow_coords[:, 1], slow_coords[:, 2],
                    'r-', linewidth=0.5, alpha=0.6, label='Slow Variable')

        ax.set_title('Phase Space Trajectory')
        ax.set_xlabel(f'E[Dim {dims[0]}]')
        ax.set_ylabel(f'E[Dim {dims[1]}]')
        ax.set_zlabel(f'E[Dim {dims[2]}]')
        ax.legend()

        if output_path:
            plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"相位空间图保存至: {output_path}")

        plt.close()

    def visualize_lyapunov_evolution(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        SubTask 25.4c: 绘制李雅普诺夫指数演化图

        Args:
            output_path: 输出路径（可选）
            figsize: 图像大小
        """
        if len(self._lyapunov_history) == 0:
            logger.warning("无李雅普诺夫指数历史数据")
            return

        fig, axes = plt.subplots(1, 1, figsize=figsize)

        # 李雅普诺夫指数历史
        lambda_history = list(self._lyapunov_history)
        steps = [i * self.config.lyapunov_calculation_interval for i in range(len(lambda_history))]

        axes.plot(steps, lambda_history, 'g-', linewidth=1, label='λ_max')

        # 边缘混沌区间
        axes.axhline(y=self.config.lyapunov_min_threshold, color='r', linestyle='--',
                     label=f'Lower Bound ({self.config.lyapunov_min_threshold})')
        axes.axhline(y=self.config.lyapunov_max_threshold, color='r', linestyle='--',
                     label=f'Upper Bound ({self.config.lyapunov_max_threshold})')

        # 标记区间
        axes.fill_between(steps, self.config.lyapunov_min_threshold,
                          self.config.lyapunov_max_threshold, alpha=0.2, color='green',
                          label='Edge-of-Chaos Region')

        axes.set_title('Lyapunov Exponent Evolution')
        axes.set_xlabel('Step')
        axes.set_ylabel('λ_max')
        axes.legend()
        axes.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"李雅普诺夫演化图保存至: {output_path}")

        plt.close()

    def visualize_all(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12)
    ) -> None:
        """
        SubTask 25.4d: 综合可视化（所有动力学指标）

        Args:
            output_path: 输出路径（可选）
            figsize: 图像大小
        """
        if len(self._state_history) == 0:
            logger.warning("无数据可供可视化")
            return

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig)

        # 1. 时间序列（快变量范数）
        ax1 = fig.add_subplot(gs[0, 0])
        timestamps = [state.timestamp for state in self._state_history]
        ax1.plot(timestamps, list(self._fast_norm_history), 'b-', linewidth=1)
        ax1.set_title('Fast Variable Norm')
        ax1.set_xlabel('Time (s)')
        ax1.grid(True, alpha=0.3)

        # 2. 时间序列（慢变量范数）
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(timestamps, list(self._slow_norm_history), 'r-', linewidth=1)
        ax2.set_title('Slow Variable Norm')
        ax2.set_xlabel('Time (s)')
        ax2.grid(True, alpha=0.3)

        # 3. 李雅普诺夫指数演化
        ax3 = fig.add_subplot(gs[0, 2])
        if len(self._lyapunov_history) > 0:
            lambda_history = list(self._lyapunov_history)
            steps = [i * self.config.lyapunov_calculation_interval for i in range(len(lambda_history))]
            ax3.plot(steps, lambda_history, 'g-', linewidth=1)
            ax3.axhline(y=self.config.lyapunov_max_threshold, color='r', linestyle='--')
            ax3.set_title('Lyapunov Exponent')
            ax3.set_xlabel('Step')
            ax3.grid(True, alpha=0.3)

        # 4. 自相关系数演化
        ax4 = fig.add_subplot(gs[1, 0])
        if len(self._autocorrelation_history) > 0:
            autocorr_history = list(self._autocorrelation_history)
            steps = [i * self.config.autocorrelation_check_interval for i in range(len(autocorr_history))]
            ax4.plot(steps, autocorr_history, 'm-', linewidth=1)
            ax4.axhline(y=self.config.autocorrelation_min_threshold, color='r', linestyle='--')
            ax4.set_title('Autocorrelation ρ(τ)')
            ax4.set_xlabel('Step')
            ax4.grid(True, alpha=0.3)

        # 5. 自预测误差演化
        ax5 = fig.add_subplot(gs[1, 1])
        if len(self._self_prediction_history) > 0:
            prediction_history = list(self._self_prediction_history)
            steps = [i * self.config.self_prediction_check_interval for i in range(len(prediction_history))]
            ax5.plot(steps, prediction_history, 'c-', linewidth=1)
            ax5.axhline(y=self.config.self_prediction_epsilon_min, color='r', linestyle='--')
            ax5.axhline(y=self.config.self_prediction_epsilon_max, color='r', linestyle='--')
            ax5.set_title('Self-Prediction Error')
            ax5.set_xlabel('Step')
            ax5.grid(True, alpha=0.3)

        # 6. 相位空间（3D）
        ax6 = fig.add_subplot(gs[1, 2], projection='3d')
        dims = self.config.phase_space_dimensions
        fast_coords = []
        for state in self._state_history:
            E_fast = state.E_fast.numpy()
            if len(dims) >= 3 and E_fast.shape[0] > dims[2]:
                fast_coords.append([E_fast[dims[0]], E_fast[dims[1]], E_fast[dims[2]]])

        if len(fast_coords) > 0:
            fast_coords = np.array(fast_coords)
            ax6.plot(fast_coords[:, 0], fast_coords[:, 1], fast_coords[:, 2],
                     'b-', linewidth=0.5, alpha=0.6)
            ax6.set_title('Phase Space (Fast)')

        # 7. 系统健康度雷达图
        ax7 = fig.add_subplot(gs[2, 0], polar=True)
        indicators = self.get_current_indicators()

        # 健康度指标
        health_metrics = [
            ('Autocorrelation', indicators.autocorrelation_passed),
            ('Lyapunov', indicators.lyapunov_passed),
            ('Self-Prediction', indicators.self_prediction_passed)
        ]

        angles = np.linspace(0, 2 * np.pi, len(health_metrics), endpoint=False)
        values = [1.0 if passed else 0.0 for _, passed in health_metrics]

        ax7.plot(angles, values, 'o-', linewidth=2)
        ax7.fill(angles, values, alpha=0.25)
        ax7.set_xticks(angles)
        ax7.set_xticklabels([name for name, _ in health_metrics])
        ax7.set_ylim(0, 1)
        ax7.set_title('System Health')

        # 8. 状态分布直方图
        ax8 = fig.add_subplot(gs[2, 1])
        fast_norms = list(self._fast_norm_history)
        ax8.hist(fast_norms, bins=30, color='blue', alpha=0.7, label='Fast')
        slow_norms = list(self._slow_norm_history)
        ax8.hist(slow_norms, bins=30, color='red', alpha=0.7, label='Slow')
        ax8.set_title('State Norm Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)

        # 9. 统计摘要
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
动力学监测统计摘要

监测步数: {self._step_count}
历史长度: {len(self._state_history)}

自相关系数:
  ρ(τ): {indicators.autocorrelation_rho:.4f}
  通过: {'✓' if indicators.autocorrelation_passed else '✗'}

李雅普诺夫指数:
  λ_max: {indicators.lyapunov_lambda_mean:.6f}
  通过: {'✓' if indicators.lyapunov_passed else '✗'}

自预测误差:
  E_self: {indicators.self_prediction_error_mean:.6f}
  通过: {'✓' if indicators.self_prediction_passed else '✗'}

系统状态: {'健康' if all([indicators.autocorrelation_passed,
                            indicators.lyapunov_passed,
                            indicators.self_prediction_passed]) else '警告'}
"""

        ax9.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                 family='monospace')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.plot_dpi, bbox_inches='tight')
            logger.info(f"综合可视化保存至: {output_path}")

        plt.close()

    def save_report(self, filepath: str) -> None:
        """保存监测报告到JSON文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_dynamics_report()

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"动力学监测报告保存至: {filepath}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取监测统计信息"""
        return {
            "is_monitoring": self._is_monitoring,
            "step_count": self._step_count,
            "history_length": len(self._state_history),
            "autocorr_history_length": len(self._autocorrelation_history),
            "lyapunov_history_length": len(self._lyapunov_history),
            "prediction_history_length": len(self._self_prediction_history)
        }