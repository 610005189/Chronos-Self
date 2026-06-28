"""
P0级核心验证模块
================

实现最高优先级的核心动力学验证，验证系统的基本稳定性和边缘混沌状态。

核心功能：
- 72小时无输入开环运行测试：验证系统在无外部输入下的稳定性
- 慢变量基线漂移率监测：验证抗寂灭能力
- 最大李雅普诺夫指数计算：验证边缘混沌状态
- 动力学对齐验证：验证ODE连续性

验证通过标准：
- 开环运行稳定：72小时内系统未崩溃，维持边缘混沌稳态
- 漂移率达标：drift_rate < threshold（抗寂灭能力）
- 李雅普诺夫达标：λ_max ∈ (0, 0.1)（边缘混沌）
- 对齐验证通过：多步长轨迹终点误差可控（ODE连续性）
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from scipy.integrate import odeint
from scipy.signal import correlate
import matplotlib.pyplot as plt

from chronos_core.utils.config import ChronosConfig, ValidationConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

logger = logging.getLogger(__name__)


@dataclass
class P0ValidationConfig:
    """P0级验证配置"""

    # 72小时开环运行参数
    open_loop_hours: float = 72.0
    open_loop_dt: float = 0.01  # 时间步长（秒）
    stability_check_interval: int = 1000  # 稳定性检查间隔（步数）
    max_stability_warnings: int = 20  # 最大稳定性警告次数

    # 慢变量漂移率参数
    drift_calculation_window: int = 100  # 漂移率计算窗口
    max_baseline_drift_rate: float = 0.1  # 最大基线漂移率阈值
    drift_monitoring_interval: int = 500  # 漂移监测间隔

    # 李雅普诺夫指数参数
    lyapunov_calculation_steps: int = 1000  # 计算步数
    lyapunov_min_threshold: float = 0.0  # 最小阈值（边缘混沌下限）
    lyapunov_max_threshold: float = 0.1  # 最大阈值（边缘混沌上限）
    perturbation_magnitude: float = 1e-6  # 扰动大小
    lyapunov_recalculation_interval: int = 2000  # 重计算间隔

    # 动力学对齐验证参数
    alignment_test_steps: List[int] = field(default_factory=lambda: [10, 100, 1000])
    alignment_max_error_threshold: float = 0.05  # 最大终点误差阈值
    alignment_num_tests: int = 5  # 测试次数

    # 验证报告参数
    report_output_dir: str = "validation_results"
    save_trajectory: bool = True
    trajectory_sample_interval: int = 100


@dataclass
class P0ValidationResult:
    """P0级验证结果"""

    # 总体结果
    is_passed: bool = False
    overall_score: float = 0.0

    # 开环运行测试结果
    open_loop_passed: bool = False
    open_loop_stable: bool = False
    open_loop_edge_of_chaos: bool = False
    open_loop_duration_hours: float = 0.0
    open_loop_steps_completed: int = 0
    open_loop_stability_warnings: int = 0
    open_loop_final_state: Optional[SelfState] = None

    # 漂移率测试结果
    drift_passed: bool = False
    drift_rate: float = 0.0
    drift_baseline_norm_initial: float = 0.0
    drift_baseline_norm_final: float = 0.0
    drift_time_elapsed: float = 0.0

    # 李雅普诺夫指数测试结果
    lyapunov_passed: bool = False
    lyapunov_max: float = 0.0
    lyapunov_min: float = 0.0
    lyapunov_mean: float = 0.0
    lyapunov_std: float = 0.0
    lyapunov_history: List[float] = field(default_factory=list)

    # 动力学对齐测试结果
    alignment_passed: bool = False
    alignment_errors: Dict[int, float] = field(default_factory=dict)  # step_size -> error
    alignment_max_error: float = 0.0
    alignment_avg_error: float = 0.0

    # 元意识引擎指标（可选）
    meta_consciousness_enabled: bool = False
    m_pre_history: Optional[List[float]] = None
    lambda_history: Optional[List[int]] = None
    awareness_gradient_history: Optional[List[List[float]]] = None

    # 统计信息
    validation_time: float = 0.0
    device: str = "cpu"
    timing_breakdown: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "is_passed": self.is_passed,
            "overall_score": self.overall_score,
            "open_loop": {
                "passed": self.open_loop_passed,
                "stable": self.open_loop_stable,
                "edge_of_chaos": self.open_loop_edge_of_chaos,
                "duration_hours": self.open_loop_duration_hours,
                "steps_completed": self.open_loop_steps_completed,
                "stability_warnings": self.open_loop_stability_warnings
            },
            "drift": {
                "passed": self.drift_passed,
                "rate": self.drift_rate,
                "baseline_norm_initial": self.drift_baseline_norm_initial,
                "baseline_norm_final": self.drift_baseline_norm_final,
                "time_elapsed": self.drift_time_elapsed
            },
            "lyapunov": {
                "passed": self.lyapunov_passed,
                "max": self.lyapunov_max,
                "min": self.lyapunov_min,
                "mean": self.lyapunov_mean,
                "std": self.lyapunov_std,
                "history": self.lyapunov_history
            },
            "alignment": {
                "passed": self.alignment_passed,
                "errors": self.alignment_errors,
                "max_error": self.alignment_max_error,
                "avg_error": self.alignment_avg_error
            },
            "stats": {
                "validation_time": self.validation_time,
                "device": self.device,
                "timing_breakdown": self.timing_breakdown
            }
        }
        if self.meta_consciousness_enabled:
            result["meta_consciousness"] = {
                "enabled": True,
                "m_pre_history": self.m_pre_history,
                "lambda_history": self.lambda_history,
                "awareness_gradient_history": self.awareness_gradient_history
            }
        return result


class P0Validation:
    """
    P0级核心验证系统

    实现最高优先级的核心动力学验证，验证系统基本稳定性和边缘混沌状态。

    使用示例：
        validator = P0Validation(engine, config)
        result = validator.run_full_validation()
        validator.save_report(result, "p0_report.json")
    """

    def __init__(
        self,
        engine: IntegrationEngine,
        config: Optional[ChronosConfig] = None,
        p0_config: Optional[P0ValidationConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化P0验证器

        Args:
            engine: IntegrationEngine实例
            config: 全局配置
            p0_config: P0验证配置
            device: 计算设备
        """
        self.engine = engine
        self.global_config = config or ChronosConfig()
        self.config = p0_config or P0ValidationConfig()

        # 合并全局配置（仅在用户未显式传入 p0_config 时覆盖）
        if p0_config is None and hasattr(self.global_config, 'validation'):
            self.config.open_loop_hours = self.global_config.validation.p0_open_loop_hours
            self.config.max_baseline_drift_rate = self.global_config.validation.p0_max_baseline_drift

        self.device = device or self.global_config.device

        # 验证状态
        self._trajectory: List[SelfState] = []
        self._baseline_initial: Optional[torch.Tensor] = None
        self._baseline_history: List[torch.Tensor] = []

        logger.info(
            f"P0Validation initialized: "
            f"open_loop_hours={self.config.open_loop_hours}, "
            f"device={self.device}"
        )

    def run_full_validation(
        self,
        initial_state: Optional[SelfState] = None,
        verbose: bool = True
    ) -> P0ValidationResult:
        """
        执行完整的P0级验证

        Args:
            initial_state: 初始状态（可选）
            verbose: 是否输出详细日志

        Returns:
            P0ValidationResult: 验证结果
        """
        start_time = time.time()

        if verbose:
            logger.info("=" * 80)
            logger.info("开始 P0级核心验证")
            logger.info("=" * 80)

        # 创建结果对象
        result = P0ValidationResult(device=self.device)

        # 初始化状态（如果未提供）
        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(self.engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(self.engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        # 保存初始慢变量基线
        self._baseline_initial = initial_state.E_slow.clone()

        # ===== SubTask 24.1: 72小时无输入开环运行测试 =====
        if verbose:
            logger.info("\n[SubTask 24.1] 72小时无输入开环运行测试...")

        open_loop_result = self._test_open_loop_run(initial_state, verbose)
        result.open_loop_passed = open_loop_result["passed"]
        result.open_loop_stable = open_loop_result["stable"]
        result.open_loop_edge_of_chaos = open_loop_result["edge_of_chaos"]
        result.open_loop_duration_hours = open_loop_result["duration_hours"]
        result.open_loop_steps_completed = open_loop_result["steps_completed"]
        result.open_loop_stability_warnings = open_loop_result["stability_warnings"]
        result.open_loop_final_state = open_loop_result["final_state"]

        # 元意识引擎指标
        result.meta_consciousness_enabled = open_loop_result.get("meta_consciousness_enabled", False)
        if result.meta_consciousness_enabled:
            result.m_pre_history = open_loop_result.get("m_pre_history")
            result.lambda_history = open_loop_result.get("lambda_history")
            result.awareness_gradient_history = open_loop_result.get("awareness_gradient_history")

        if not result.open_loop_passed:
            logger.warning("❌ 72小时开环运行测试失败")
            # 如果开环运行失败，后续测试可能无法进行
            result.is_passed = False
            result.overall_score = 0.0
            result.validation_time = time.time() - start_time
            result.timing_breakdown = {
                "open_loop_run": open_loop_result.get("timing", {}).get("total_time_s", 0.0),
                "open_loop_steps_per_sec": open_loop_result.get("timing", {}).get("steps_per_sec", 0.0),
            }
            return result

        if verbose:
            logger.info("✓ 72小时开环运行测试通过")

        # ===== SubTask 24.2: 慢变量基线漂移率监测 =====
        if verbose:
            logger.info("\n[SubTask 24.2] 慢变量基线漂移率监测...")

        drift_result = self._calculate_baseline_drift_rate(verbose)
        result.drift_passed = drift_result["passed"]
        result.drift_rate = drift_result["rate"]
        result.drift_baseline_norm_initial = drift_result["baseline_norm_initial"]
        result.drift_baseline_norm_final = drift_result["baseline_norm_final"]
        result.drift_time_elapsed = drift_result["time_elapsed"]

        if verbose:
            if result.drift_passed:
                logger.info(f"✓ 漂移率监测通过: drift_rate={result.drift_rate:.6f}")
            else:
                logger.warning(f"❌ 漂移率监测失败: drift_rate={result.drift_rate:.6f}")

        # ===== SubTask 24.3: 最大李雅普诺夫指数计算 =====
        if verbose:
            logger.info("\n[SubTask 24.3] 最大李雅普诺夫指数计算...")

        lyapunov_result = self._calculate_lyapunov_exponent(
            result.open_loop_final_state,
            verbose
        )
        result.lyapunov_passed = lyapunov_result["passed"]
        result.lyapunov_max = lyapunov_result["max"]
        result.lyapunov_min = lyapunov_result["min"]
        result.lyapunov_mean = lyapunov_result["mean"]
        result.lyapunov_std = lyapunov_result["std"]
        result.lyapunov_history = lyapunov_result["history"]

        if verbose:
            if result.lyapunov_passed:
                logger.info(
                    f"✓ 李雅普诺夫指数通过: "
                    f"λ_max={result.lyapunov_mean:.6f} ∈ (0, 0.1)"
                )
            else:
                logger.warning(
                    f"❌ 李雅普诺夫指数失败: "
                    f"λ_max={result.lyapunov_mean:.6f}"
                )

        # ===== SubTask 24.4: 动力学对齐验证 =====
        if verbose:
            logger.info("\n[SubTask 24.4] 动力学对齐验证（多步长轨迹终点误差）...")

        alignment_result = self._test_dynamics_alignment(
            initial_state,
            verbose
        )
        result.alignment_passed = alignment_result["passed"]
        result.alignment_errors = alignment_result["errors"]
        result.alignment_max_error = alignment_result["max_error"]
        result.alignment_avg_error = alignment_result["avg_error"]

        if verbose:
            if result.alignment_passed:
                logger.info(
                    f"✓ 动力学对齐验证通过: "
                    f"max_error={result.alignment_max_error:.6f}"
                )
            else:
                logger.warning(
                    f"❌ 动力学对齐验证失败: "
                    f"max_error={result.alignment_max_error:.6f}"
                )

        # ===== 综合判定 =====
        result.is_passed = (
            result.open_loop_passed and
            result.drift_passed and
            result.lyapunov_passed and
            result.alignment_passed
        )

        # 计算总体得分（加权平均）
        scores = {
            "open_loop": 0.4 if result.open_loop_passed else 0.0,
            "drift": 0.2 if result.drift_passed else 0.0,
            "lyapunov": 0.2 if result.lyapunov_passed else 0.0,
            "alignment": 0.2 if result.alignment_passed else 0.0
        }
        result.overall_score = sum(scores.values())

        # 收集各子步骤计时数据
        result.timing_breakdown = {
            "open_loop_run": open_loop_result.get("timing", {}).get("total_time_s", 0.0),
            "open_loop_steps_per_sec": open_loop_result.get("timing", {}).get("steps_per_sec", 0.0),
            "baseline_drift_calc": drift_result.get("timing", {}).get("calc_time_s", 0.0),
            "lyapunov_calc": lyapunov_result.get("timing", {}).get("calc_time_s", 0.0),
        }

        result.validation_time = time.time() - start_time

        if verbose:
            logger.info("\n" + "=" * 80)
            logger.info(f"P0级验证完成: {'✓ 通过' if result.is_passed else '❌ 失败'}")
            logger.info(f"总体得分: {result.overall_score:.2f}")
            logger.info(f"验证时间: {result.validation_time:.2f}秒")
            logger.info("=" * 80)

        return result

    def _test_open_loop_run(
        self,
        initial_state: SelfState,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 24.1: 72小时无输入开环运行测试

        系统无外部输入连续运行72小时（模拟时间），监测稳定性。

        Args:
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            测试结果字典
        """
        # 计算总步数（dt=0.01s，72小时=259200秒）
        dt = self.config.open_loop_dt
        total_steps = int(self.config.open_loop_hours * 3600 / dt)

        if verbose:
            logger.info(f"  开始72小时开环运行（模拟时间）...")
            logger.info(f"  总步数: {total_steps}")
            logger.info(f"  时间步长: {dt}秒")

        # 重置引擎
        self.engine.reset()
        self.engine.initialize()

        # 运行统计
        stability_warnings = 0
        steps_completed = 0
        is_stable = True
        is_edge_of_chaos = True
        loop_start_time = time.time()

        # 状态轨迹采样
        trajectory_samples = []

        # 检测元意识引擎是否启用
        meta_consciousness_enabled = False
        meta_field = None
        meta_depth = None
        m_pre_history = []
        lambda_history = []
        awareness_gradient_history = []

        meta_system = getattr(self.engine, 'meta_cognitive_system', None)
        if meta_system is not None:
            meta_config = getattr(self.global_config, 'meta_cognitive', None)
            if meta_config is not None and getattr(meta_config, 'enable_meta_consciousness', False):
                meta_consciousness_enabled = True
                dynamic_layers = getattr(meta_system, 'dynamic_layers', None)
                if dynamic_layers is not None:
                    if hasattr(dynamic_layers, 'meta_field'):
                        meta_field = dynamic_layers.meta_field
                    if hasattr(dynamic_layers, 'self_ref_depth'):
                        meta_depth = dynamic_layers.self_ref_depth

        # 运行循环
        current_state = initial_state

        for step_idx in range(total_steps):
            # 执行单步（无外部输入）
            try:
                current_state = self.engine.step(current_state, inputs=None, dt=dt)
                steps_completed += 1

                # 稳定性检查
                if step_idx % self.config.stability_check_interval == 0:
                    stability_report = self.engine.coupling_system.get_stability_report()

                    if not stability_report.get("is_stable", True):
                        stability_warnings += 1

                    if not stability_report.get("is_edge_of_chaos", True):
                        is_edge_of_chaos = False

                    # 检查状态有效性
                    is_valid, errors = current_state.validate()
                    if not is_valid:
                        is_stable = False
                        logger.warning(f"  状态无效: {errors}")
                        break

                    # 记录轨迹样本
                    if self.config.save_trajectory:
                        if step_idx % self.config.trajectory_sample_interval == 0:
                            trajectory_samples.append(current_state.copy())

                    # 记录元意识指标
                    if meta_consciousness_enabled:
                        if step_idx % self.config.trajectory_sample_interval == 0:
                            if meta_field is not None:
                                m_pre_history.append(meta_field.get_value())
                            if meta_depth is not None:
                                lambda_history.append(meta_depth.get_depth())
                            if meta_system is not None and hasattr(meta_system, 'get_awareness_gradients'):
                                try:
                                    grads = meta_system.get_awareness_gradients()
                                    awareness_gradient_history.append(grads)
                                except Exception:
                                    pass

                    # 日志：每1000步平均步时
                    elapsed = time.time() - loop_start_time
                    if elapsed > 0 and verbose:
                        avg_step_ms = (elapsed / (step_idx + 1)) * 1000.0
                        step_rate = (step_idx + 1) / elapsed
                        logger.debug(
                            f"  步数 {step_idx}: "
                            f"平均步时 {avg_step_ms:.4f} ms, "
                            f"速率 {step_rate:.1f} 步/s"
                        )

                    # 严重不稳定时停止
                    if stability_warnings > self.config.max_stability_warnings:
                        is_stable = False
                        logger.warning(
                            f"  过多稳定性警告 ({stability_warnings}), "
                            f"终止运行于步骤 {step_idx}"
                        )
                        break

                # 进度报告
                if verbose and step_idx % 10000 == 0:
                    progress = step_idx / total_steps * 100
                    logger.info(
                        f"  进度: {progress:.1f}% "
                        f"({step_idx}/{total_steps}步), "
                        f"E_fast_norm={current_state.get_fast_norm():.4f}, "
                        f"E_slow_norm={current_state.get_slow_norm():.4f}"
                    )

            except Exception as e:
                logger.error(f"  运行错误于步骤 {step_idx}: {e}")
                is_stable = False
                break

        # 计算实际运行时长（模拟时间）
        duration_hours = steps_completed * dt / 3600

        # 保存轨迹
        self._trajectory = trajectory_samples

        # 保存慢变量历史
        self._baseline_history = [
            state.E_slow.clone() for state in trajectory_samples
        ]

        result = {
            "passed": (
                is_stable and
                is_edge_of_chaos and
                duration_hours >= self.config.open_loop_hours * 0.95  # 至少完成95%
            ),
            "stable": is_stable,
            "edge_of_chaos": is_edge_of_chaos,
            "duration_hours": duration_hours,
            "steps_completed": steps_completed,
            "stability_warnings": stability_warnings,
            "final_state": current_state,
            "meta_consciousness_enabled": meta_consciousness_enabled,
            "m_pre_history": m_pre_history if meta_consciousness_enabled else None,
            "lambda_history": lambda_history if meta_consciousness_enabled else None,
            "awareness_gradient_history": awareness_gradient_history if meta_consciousness_enabled else None,
            "timing": {
                "total_time_s": time.time() - loop_start_time,
                "steps_per_sec": steps_completed / (time.time() - loop_start_time) if (time.time() - loop_start_time) > 0 else 0.0
            }
        }

        if verbose:
            logger.info(
                f"  开环运行完成: "
                f"duration={duration_hours:.2f}小时, "
                f"stable={is_stable}, "
                f"edge_of_chaos={is_edge_of_chaos}, "
                f"warnings={stability_warnings}"
            )

        return result

    def _calculate_baseline_drift_rate(
        self,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 24.2: 慢变量基线漂移率监测

        计算：drift_rate = ||E_slow(T) - E_slow(0)|| / T
        检查漂移率是否低于阈值，验证抗寂灭能力。

        Args:
            verbose: 详细日志

        Returns:
            测试结果字典
        """
        drift_start_time = time.time()

        if len(self._baseline_history) == 0:
            logger.warning("  无基线历史数据，使用轨迹数据")
            if len(self._trajectory) > 0:
                baseline_initial = self._trajectory[0].E_slow
                baseline_final = self._trajectory[-1].E_slow
            else:
                logger.error("  无轨迹数据，无法计算漂移率")
                return {
                    "passed": False,
                    "rate": float('inf'),
                    "baseline_norm_initial": 0.0,
                    "baseline_norm_final": 0.0,
                    "time_elapsed": 0.0,
                    "timing": {"calc_time_s": time.time() - drift_start_time}
                }
        else:
            baseline_initial = self._baseline_history[0]
            baseline_final = self._baseline_history[-1]

        # 计算漂移率
        drift_norm = torch.norm(baseline_final - baseline_initial).item()
        time_elapsed = len(self._trajectory) * self.config.trajectory_sample_interval * self.config.open_loop_dt

        if time_elapsed > 0:
            drift_rate = drift_norm / time_elapsed
        else:
            drift_rate = float('inf')

        baseline_norm_initial = torch.norm(baseline_initial).item()
        baseline_norm_final = torch.norm(baseline_final).item()

        result = {
            "passed": drift_rate < self.config.max_baseline_drift_rate,
            "rate": drift_rate,
            "baseline_norm_initial": baseline_norm_initial,
            "baseline_norm_final": baseline_norm_final,
            "time_elapsed": time_elapsed,
            "timing": {"calc_time_s": time.time() - drift_start_time}
        }

        if verbose:
            logger.info(
                f"  漂移率: {drift_rate:.6f} "
                f"(阈值: {self.config.max_baseline_drift_rate})"
            )
            logger.info(
                f"  基线范数变化: "
                f"{baseline_norm_initial:.4f} -> {baseline_norm_final:.4f}"
            )

        return result

    def _calculate_lyapunov_exponent(
        self,
        final_state: SelfState,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 24.3: 最大李雅普诺夫指数计算

        λ_max = lim_{t→∞} (1/t) ln(||δE(t)|| / ||δE(0)||)
        使用数值方法计算，检查 λ_max ∈ (0, 0.1)，验证边缘混沌状态。

        Args:
            final_state: 72小时运行后的最终状态
            verbose: 详细日志

        Returns:
            测试结果字典
        """
        if verbose:
            logger.info(f"  计算李雅普诺夫指数...")

        lyapunov_start_time = time.time()

        lyapunov_history = []

        # 重置引擎到最终状态附近
        self.engine.reset()
        self.engine.initialize()

        # 创建参考轨迹和扰动轨迹
        reference_state = final_state.copy()

        # 计算步数
        calc_steps = self.config.lyapunov_calculation_steps

        # 执行多次计算取平均
        num_trials = 5

        for trial in range(num_trials):
            # 创建扰动状态（随机扰动）
            perturbation = torch.randn_like(reference_state.E_fast) * self.config.perturbation_magnitude
            perturbed_state = SelfState(
                E_fast=reference_state.E_fast + perturbation,
                E_slow=reference_state.E_slow.clone(),
                timestamp=reference_state.timestamp
            )

            # 初始扰动距离
            delta_0 = torch.norm(perturbation).item()

            # 运行参考轨迹和扰动轨迹
            ref_state = reference_state.copy()
            pert_state = perturbed_state.copy()

            dt = self.config.open_loop_dt

            for step in range(calc_steps):
                # 更新参考轨迹
                ref_state = self.engine.step(ref_state, inputs=None, dt=dt)

                # 保存引擎状态（用于恢复）
                # 注意：这里需要重新初始化引擎来运行扰动轨迹
                # 实际应用中可以使用两个独立的引擎实例

            # 重新初始化引擎运行扰动轨迹
            self.engine.reset()
            self.engine.initialize()

            pert_state = perturbed_state.copy()
            for step in range(calc_steps):
                pert_state = self.engine.step(pert_state, inputs=None, dt=dt)

            # 计算最终扰动距离
            delta_t = torch.norm(pert_state.E_fast - ref_state.E_fast).item()

            # 计算李雅普诺夫指数
            time_elapsed = calc_steps * dt

            if delta_0 > 0 and delta_t > 0:
                lyapunov = (1.0 / time_elapsed) * np.log(delta_t / delta_0)
                lyapunov_history.append(lyapunov)

            if verbose and trial % 2 == 0:
                logger.info(
                    f"  试验 {trial+1}/{num_trials}: "
                    f"δ_0={delta_0:.6e}, δ_t={delta_t:.6e}, "
                    f"λ={lyapunov:.6f}"
                )

        # 统计李雅普诺夫指数
        if len(lyapunov_history) > 0:
            lyapunov_max = max(lyapunov_history)
            lyapunov_min = min(lyapunov_history)
            lyapunov_mean = np.mean(lyapunov_history)
            lyapunov_std = np.std(lyapunov_history)
        else:
            lyapunov_max = lyapunov_min = lyapunov_mean = lyapunov_std = 0.0

        # 检查是否在边缘混沌区间 (0, 0.1)
        passed = bool(
            lyapunov_mean > self.config.lyapunov_min_threshold and
            lyapunov_mean < self.config.lyapunov_max_threshold
        )

        result = {
            "passed": passed,
            "max": lyapunov_max,
            "min": lyapunov_min,
            "mean": lyapunov_mean,
            "std": lyapunov_std,
            "history": lyapunov_history,
            "timing": {"calc_time_s": time.time() - lyapunov_start_time}
        }

        if verbose:
            logger.info(
                f"  李雅普诺夫指数统计: "
                f"mean={lyapunov_mean:.6f}, std={lyapunov_std:.6f}, "
                f"范围=[{lyapunov_min:.6f}, {lyapunov_max:.6f}]"
            )
            logger.info(
                f"  边缘混沌区间: "
                f"(0, {self.config.lyapunov_max_threshold})"
            )

        return result

    def _test_dynamics_alignment(
        self,
        initial_state: SelfState,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        SubTask 24.4: 动力学对齐验证（多步长轨迹终点误差）

        使用不同步长积分同一状态序列，计算终点误差，验证ODE连续性。

        Args:
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            测试结果字典
        """
        if verbose:
            logger.info(f"  测试动力学对齐（多步长积分）...")

        alignment_errors = {}

        # 参考步长（最细步长）
        reference_dt = self.config.open_loop_dt

        # 测试不同步长
        for test_steps in self.config.alignment_test_steps:
            if verbose:
                logger.info(f"  测试步数: {test_steps}")

            # 使用参考步长积分（精细积分）
            self.engine.reset()
            self.engine.initialize()

            ref_state = initial_state.copy()
            for step in range(test_steps):
                ref_state = self.engine.step(ref_state, inputs=None, dt=reference_dt)

            # 使用不同步长积分（粗粒积分）
            for coarse_steps in [1, 10, 100]:
                if coarse_steps > test_steps:
                    continue

                coarse_dt = test_steps * reference_dt / coarse_steps

                self.engine.reset()
                self.engine.initialize()

                coarse_state = initial_state.copy()
                for step in range(coarse_steps):
                    coarse_state = self.engine.step(coarse_state, inputs=None, dt=coarse_dt)

                # 计算终点误差
                error_fast = torch.norm(coarse_state.E_fast - ref_state.E_fast).item()
                error_slow = torch.norm(coarse_state.E_slow - ref_state.E_slow).item()
                total_error = (error_fast + error_slow) / 2.0

                alignment_errors[coarse_steps] = total_error

                if verbose:
                    logger.info(
                        f"    粗步数={coarse_steps}, "
                        f"dt={coarse_dt:.4f}, "
                        f"error={total_error:.6f}"
                    )

        # 计算统计
        if len(alignment_errors) > 0:
            max_error = max(alignment_errors.values())
            avg_error = np.mean(list(alignment_errors.values()))
        else:
            max_error = avg_error = 0.0

        # 判断是否通过（误差应小于阈值）
        passed = max_error < self.config.alignment_max_error_threshold

        result = {
            "passed": passed,
            "errors": alignment_errors,
            "max_error": max_error,
            "avg_error": avg_error
        }

        if verbose:
            logger.info(
                f"  对齐验证结果: "
                f"max_error={max_error:.6f}, "
                f"avg_error={avg_error:.6f}, "
                f"阈值={self.config.alignment_max_error_threshold}"
            )

        return result

    def save_report(
        self,
        result: P0ValidationResult,
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        保存验证报告

        Args:
            result: 验证结果
            filepath: 文件路径
            format: 格式（json或markdown）
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        elif format == "markdown":
            report_md = self._generate_markdown_report(result)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_md)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"P0验证报告保存至: {filepath}")

    def _generate_markdown_report(self, result: P0ValidationResult) -> str:
        """生成Markdown格式的验证报告"""
        report = f"""# P0级核心验证报告

## 总体结果

- **状态**: {'✓ 通过' if result.is_passed else '❌ 失败'}
- **总体得分**: {result.overall_score:.2f}
- **验证时间**: {result.validation_time:.2f}秒
- **计算设备**: {result.device}

## 详细结果

### 1. 72小时无输入开环运行测试 (SubTask 24.1)

- **状态**: {'✓ 通过' if result.open_loop_passed else '❌ 失败'}
- **稳定性**: {'✓ 稳定' if result.open_loop_stable else '❌ 不稳定'}
- **边缘混沌**: {'✓ 维持' if result.open_loop_edge_of_chaos else '❌ 未维持'}
- **运行时长**: {result.open_loop_duration_hours:.2f}小时
- **完成步数**: {result.open_loop_steps_completed}
- **稳定性警告**: {result.open_loop_stability_warnings}

### 2. 慢变量基线漂移率监测 (SubTask 24.2)

- **状态**: {'✓ 通过' if result.drift_passed else '❌ 失败'}
- **漂移率**: {result.drift_rate:.6f} (阈值: {self.config.max_baseline_drift_rate})
- **初始基线范数**: {result.drift_baseline_norm_initial:.4f}
- **最终基线范数**: {result.drift_baseline_norm_final:.4f}
- **时间跨度**: {result.drift_time_elapsed:.2f}秒

### 3. 最大李雅普诺夫指数计算 (SubTask 24.3)

- **状态**: {'✓ 通过' if result.lyapunov_passed else '❌ 失败'}
- **平均值**: {result.lyapunov_mean:.6f} (应在 [0, 0.1] 区间)
- **标准差**: {result.lyapunov_std:.6f}
- **范围**: [{result.lyapunov_min:.6f}, {result.lyapunov_max:.6f}]
- **历史记录**: {len(result.lyapunov_history)}次计算

### 4. 动力学对齐验证 (SubTask 24.4)

- **状态**: {'✓ 通过' if result.alignment_passed else '❌ 失败'}
- **最大误差**: {result.alignment_max_error:.6f} (阈值: {self.config.alignment_max_error_threshold})
- **平均误差**: {result.alignment_avg_error:.6f}

#### 步长测试误差详情

"""
        for steps, error in result.alignment_errors.items():
            report += f"- {steps}步: {error:.6f}\n"

        report += """
## 结论

"""
        if result.is_passed:
            report += """✓ **P0级验证全部通过**

系统表现出：
- 稳定的72小时开环运行能力
- 抗寂灭的慢变量基线维持
- 边缘混沌的动力学特征
- 连续的ODE动力学行为

系统已具备进行P1级验证的条件。
"""
        else:
            report += """❌ **P0级验证失败**

系统在以下方面存在问题：
"""
            if not result.open_loop_passed:
                report += "- 72小时开环运行不稳定或未完成\n"
            if not result.drift_passed:
                report += "- 慢变量基线漂移率超标\n"
            if not result.lyapunov_passed:
                report += "- 李雅普诺夫指数未处于边缘混沌区间\n"
            if not result.alignment_passed:
                report += "- 动力学对齐误差过大\n"

        return report

    def visualize_trajectory(
        self,
        output_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        可视化轨迹（如果有）

        Args:
            output_path: 输出路径（可选）
            figsize: 图像大小
        """
        if len(self._trajectory) == 0:
            logger.warning("无轨迹数据可供可视化")
            return

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 时间序列
        timestamps = [state.timestamp for state in self._trajectory]
        fast_norms = [state.get_fast_norm() for state in self._trajectory]
        slow_norms = [state.get_slow_norm() for state in self._trajectory]

        # 快变量范数
        axes[0, 0].plot(timestamps, fast_norms, 'b-', linewidth=1)
        axes[0, 0].set_title('Fast Variable Norm')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('||E_fast||')
        axes[0, 0].grid(True, alpha=0.3)

        # 慢变量范数
        axes[0, 1].plot(timestamps, slow_norms, 'r-', linewidth=1)
        axes[0, 1].set_title('Slow Variable Norm')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('||E_slow||')
        axes[0, 1].grid(True, alpha=0.3)

        # 状态演化（快变量前3维）
        fast_dims = [state.E_fast[:3].numpy() for state in self._trajectory]
        axes[1, 0].plot([dim[0] for dim in fast_dims], label='Dim 0')
        axes[1, 0].plot([dim[1] for dim in fast_dims], label='Dim 1')
        axes[1, 0].plot([dim[2] for dim in fast_dims], label='Dim 2')
        axes[1, 0].set_title('Fast Variable Evolution (First 3 Dimensions)')
        axes[1, 0].set_xlabel('Sample Index')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 状态演化（慢变量前3维）
        slow_dims = [state.E_slow[:3].numpy() for state in self._trajectory]
        axes[1, 1].plot([dim[0] for dim in slow_dims], label='Dim 0')
        axes[1, 1].plot([dim[1] for dim in slow_dims], label='Dim 1')
        axes[1, 1].plot([dim[2] for dim in slow_dims], label='Dim 2')
        axes[1, 1].set_title('Slow Variable Evolution (First 3 Dimensions)')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"轨迹可视化保存至: {output_path}")

        plt.close()

    def get_statistics(self) -> Dict[str, Any]:
        """获取验证统计信息"""
        return {
            "trajectory_length": len(self._trajectory),
            "baseline_history_length": len(self._baseline_history),
            "config": {
                "open_loop_hours": self.config.open_loop_hours,
                "max_baseline_drift_rate": self.config.max_baseline_drift_rate,
                "lyapunov_max_threshold": self.config.lyapunov_max_threshold,
                "alignment_max_error_threshold": self.config.alignment_max_error_threshold
            }
        }