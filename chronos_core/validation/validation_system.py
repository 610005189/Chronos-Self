"""
验证模块（Validation Module）
=============================

整合所有验证组件，实现完整的验证流程和报告生成。

验证层级：
- P0级验证（最高优先级）：核心动力学验证
- P1级验证：功能模块验证（DMN、工作记忆、L2独立性）
- P2级验证：模式检测（动力学指标+行为学指标）

验证模式：
- 快速验证：分钟级，关键指标测试
- 完整验证：小时级，所有指标测试
- 持续监测：长期运行，实时监测

验证流程：
1. P0级验证 → 模块基本稳定性
2. P1级验证 → 功能模块正确性
3. P2级验证 → 模式检测
4. 生成验证报告
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
from datetime import datetime, timezone
from enum import Enum

from chronos_core.utils.config import ChronosConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

# 导入验证模块
from .p0_validation import P0Validation, P0ValidationResult, P0ValidationConfig
from .dynamics_monitoring import DynamicsMonitoring, DynamicsIndicators, DynamicsMonitoringConfig
from .behavioral_metrics import BehavioralMetrics, BehavioralIndicators, BehavioralMetricsConfig
from .experiment_log import ExperimentLog, ExperimentRecord, generate_experiment_id

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """验证模式"""
    QUICK = "quick"           # 快速验证（分钟级）
    FULL = "full"             # 完整验证（小时级）
    CONTINUOUS = "continuous" # 持续监测（长期）
    P0_ONLY = "p0_only"       # 仅P0级验证
    PATTERN_DETECTION = "pattern_detection"   # 仅模式检测


class ValidationLevel(Enum):
    """验证级别"""
    P0 = "p0"  # 核心动力学验证
    P1 = "p1"  # 功能模块验证
    P2 = "p2"  # 涌现判定


@dataclass
class ValidationConfig:
    """验证模块配置"""

    # 验证模式
    default_mode: ValidationMode = ValidationMode.FULL

    # 快速验证参数
    quick_validation_steps: int = 1000

    # 完整验证参数
    full_validation_hours: float = 72.0

    # P0验证参数
    p0_config: P0ValidationConfig = field(default_factory=P0ValidationConfig)

    # 动力学监测参数
    dynamics_config: DynamicsMonitoringConfig = field(default_factory=DynamicsMonitoringConfig)

    # 行为学指标参数
    behavioral_config: BehavioralMetricsConfig = field(default_factory=BehavioralMetricsConfig)

    # 报告参数
    report_output_dir: str = "validation_results"
    report_format: str = "markdown"  # json 或 markdown
    save_plots: bool = True

    # 持续监测参数
    continuous_monitoring_interval: int = 1000  # 监测间隔（步数）
    continuous_report_interval: int = 10000  # 报告间隔（步数）


@dataclass
class ValidationResult:
    """完整验证结果"""

    # 验证模式
    validation_mode: ValidationMode = ValidationMode.FULL
    validation_time: float = 0.0

    # P0级验证结果
    p0_result: Optional[P0ValidationResult] = None
    p0_passed: bool = False

    # P1级验证结果（待实现）
    p1_result: Optional[Dict[str, Any]] = None
    p1_passed: bool = False

    # P2级验证结果（动力学监测 + 行为学指标）
    dynamics_result: Optional[DynamicsIndicators] = None
    behavioral_result: Optional[BehavioralIndicators] = None
    p2_passed: bool = False

    # 综合判定
    overall_passed: bool = False
    overall_score: float = 0.0
    emergence_detected: bool = False

    # 验证报告
    report_path: Optional[str] = None

    # 实验记录与性能分析
    experiment_id: str = ""
    profiling_data: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "validation_mode": self.validation_mode.value,
            "validation_time": self.validation_time,
            "p0": {
                "passed": self.p0_passed,
                "score": self.p0_result.overall_score if self.p0_result else 0.0,
                "details": self.p0_result.to_dict() if self.p0_result else None
            },
            "p1": {
                "passed": self.p1_passed,
                "details": self.p1_result
            },
            "p2": {
                "passed": self.p2_passed,
                "dynamics": self.dynamics_result.to_dict() if self.dynamics_result else None,
                "behavioral": self.behavioral_result.to_dict() if self.behavioral_result else None
            },
            "overall": {
                "passed": self.overall_passed,
                "score": self.overall_score,
                "emergence_detected": self.emergence_detected
            },
            "report_path": self.report_path,
            "experiment_id": self.experiment_id,
            "profiling_data": self.profiling_data
        }


class Validation:
    """
    验证模块

    整合所有验证组件，实现完整的验证流程和报告生成。

    使用示例：
        validation = Validation(config)
        result = validation.run_full_validation(engine)
        validation.save_final_report(result)
    """

    def __init__(
        self,
        config: Optional[ChronosConfig] = None,
        validation_config: Optional["ValidationConfig"] = None,
        device: Optional[str] = None
    ):
        """
        初始化验证模块

        Args:
            config: 全局配置
            validation_config: 验证模块配置
            device: 计算设备
        """
        self.global_config = config or ChronosConfig()
        self.config = validation_config or ValidationConfig()

        self.device = device or self.global_config.device

        # 验证组件
        self._p0_validator: Optional[P0Validation] = None
        self._dynamics_monitor: Optional[DynamicsMonitoring] = None
        self._behavioral_metrics: Optional[BehavioralMetrics] = None

        # 验证状态
        self._is_validating = False
        self._current_level: Optional[ValidationLevel] = None
        self._start_time: Optional[float] = None

        logger.info(
            f"Validation initialized: "
            f"default_mode={self.config.default_mode.value}, "
            f"device={self.device}"
        )

    def run_validation(
        self,
        engine: IntegrationEngine,
        mode: Optional[ValidationMode] = None,
        initial_state: Optional[SelfState] = None,
        verbose: bool = True
    ) -> ValidationResult:
        """
        执行验证

        Args:
            engine: IntegrationEngine实例
            mode: 验证模式（可选）
            initial_state: 初始状态（可选）
            verbose: 是否输出详细日志

        Returns:
            ValidationResult: 验证结果
        """
        mode = mode or self.config.default_mode

        self._is_validating = True
        self._start_time = time.time()

        if verbose:
            logger.info("=" * 80)
            logger.info(f"开始验证: mode={mode.value}")
            logger.info("=" * 80)

        # 创建结果对象
        result = ValidationResult(validation_mode=mode)
        result.experiment_id = generate_experiment_id()

        # 根据验证模式执行不同流程
        if mode == ValidationMode.QUICK:
            result = self._run_quick_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.FULL:
            result = self._run_full_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.CONTINUOUS:
            result = self._run_continuous_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.P0_ONLY:
            result = self._run_p0_only_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.PATTERN_DETECTION:
            result = self._run_pattern_detection_validation(engine, initial_state, verbose)

        # 记录验证时间
        result.validation_time = time.time() - self._start_time

        # 保存报告
        self.save_final_report(result, verbose)

        # 自动记录实验日志
        try:
            config = {
                "validation_mode": mode.value,
                "engine_config": {
                    "fast_dim": engine.engine_config.fast_dim,
                    "slow_dim": engine.engine_config.slow_dim,
                    "default_dt": engine.engine_config.default_dt,
                } if hasattr(engine, 'engine_config') else {},
            }
            metrics = {
                "overall_passed": result.overall_passed,
                "overall_score": result.overall_score,
                "p0_passed": result.p0_passed,
                "p1_passed": result.p1_passed,
                "p2_passed": result.p2_passed,
            }
            record = ExperimentRecord(
                experiment_id=result.experiment_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                duration_s=round(result.validation_time, 3),
                config=config,
                metrics=metrics,
                profilation_data=result.profiling_data,
                git_commit=None,
            )
            ExperimentLog().append(record)
        except Exception as exc:
            logger.warning("Failed to auto-log experiment: %s", exc)

        self._is_validating = False

        if verbose:
            logger.info("=" * 80)
            logger.info(
                f"验证完成: "
                f"mode={mode.value}, "
                f"time={result.validation_time:.2f}秒, "
                f"passed={result.overall_passed}"
            )
            logger.info("=" * 80)

        return result

    def _run_quick_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        快速验证（分钟级，关键指标测试）

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[快速验证模式] 分钟级关键指标测试...")

        result = ValidationResult(validation_mode=ValidationMode.QUICK)

        # 快速P0级验证（缩短时间）
        # 使用较少的步数进行快速测试
        self.config.p0_config.open_loop_hours = 0.1  # 6分钟
        self.config.p0_config.lyapunov_calculation_steps = 100
        self.config.p0_config.alignment_test_steps = [10, 50]

        self._p0_validator = P0Validation(
            engine,
            self.global_config,
            self.config.p0_config,
            self.device
        )

        # 初始化状态
        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        # 执行P0级验证
        result.p0_result = self._p0_validator.run_full_validation(
            initial_state,
            verbose=verbose
        )
        result.p0_passed = result.p0_result.is_passed

        # 快速动力学监测
        self._dynamics_monitor = DynamicsMonitoring(
            engine,
            self.global_config,
            self.config.dynamics_config,
            self.device
        )

        self._dynamics_monitor.start_monitoring()

        # 快速运行
        current_state = initial_state
        for step_idx in range(self.config.quick_validation_steps):
            current_state = engine.step(current_state)
            self._dynamics_monitor.update(current_state)

        result.dynamics_result = self._dynamics_monitor.get_current_indicators()
        self._dynamics_monitor.stop_monitoring()

        # 综合判定
        result.overall_passed = result.p0_passed
        result.overall_score = result.p0_result.overall_score if result.p0_result else 0.0

        return result

    def _run_full_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        完整验证（小时级，所有指标测试）

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[完整验证模式] 小时级全指标测试...")

        result = ValidationResult(validation_mode=ValidationMode.FULL)

        # ===== P0级验证 =====
        if verbose:
            logger.info("\n[P0级验证] 核心动力学验证...")

        self._p0_validator = P0Validation(
            engine,
            self.global_config,
            self.config.p0_config,
            self.device
        )

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        result.p0_result = self._p0_validator.run_full_validation(
            initial_state,
            verbose=verbose
        )
        result.p0_passed = result.p0_result.is_passed

        if not result.p0_passed:
            logger.warning("P0级验证失败，终止后续验证")
            result.overall_passed = False
            result.overall_score = 0.0
            return result

        # ===== P1级验证（待完整实现） =====
        if verbose:
            logger.info("\n[P1级验证] 功能模块验证...")

        # P1级验证包括：
        # - DMN功能验证
        # - 工作记忆验证
        # - L2独立性验证

        result.p1_result = self._run_p1_validation(engine, verbose)
        result.p1_passed = result.p1_result.get("passed", False)

        # ===== P2级验证（动力学监测 + 行为学指标） =====
        if verbose:
            logger.info("\n[P2级验证] 涌现判定...")

        # 动力学监测
        self._dynamics_monitor = DynamicsMonitoring(
            engine,
            self.global_config,
            self.config.dynamics_config,
            self.device
        )

        self._dynamics_monitor.start_monitoring()

        # 使用P0验证后的最终状态继续运行
        current_state = result.p0_result.open_loop_final_state

        # 运行一段时间进行动力学监测
        monitoring_steps = 10000
        for step_idx in range(monitoring_steps):
            current_state = engine.step(current_state)
            self._dynamics_monitor.update(current_state, verbose=(step_idx % 1000 == 0))

        result.dynamics_result = self._dynamics_monitor.get_current_indicators()
        self._dynamics_monitor.stop_monitoring()

        # 行为学指标判定
        self._behavioral_metrics = BehavioralMetrics(
            engine,
            self.global_config,
            self.config.behavioral_config,
            self.device
        )

        result.behavioral_result = self._behavioral_metrics.run_full_evaluation(
            current_state,
            verbose=verbose
        )

        # P2级判定
        result.p2_passed = result.behavioral_result.emergence_detected

        # 综合判定
        result.overall_passed = (
            result.p0_passed and
            result.p1_passed and
            result.p2_passed
        )

        # 计算总体得分
        scores = []
        if result.p0_result:
            scores.append(result.p0_result.overall_score)
        if result.p1_passed:
            scores.append(1.0)
        if result.behavioral_result:
            scores.append(result.behavioral_result.emergence_score)

        result.overall_score = np.mean(scores) if len(scores) > 0 else 0.0
        result.emergence_detected = result.behavioral_result.emergence_detected if result.behavioral_result else False

        return result

    def _run_continuous_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        持续监测（长期运行，实时监测）

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[持续监测模式] 长期实时监测...")

        result = ValidationResult(validation_mode=ValidationMode.CONTINUOUS)

        # 初始化监测器
        self._dynamics_monitor = DynamicsMonitoring(
            engine,
            self.global_config,
            self.config.dynamics_config,
            self.device
        )

        self._dynamics_monitor.start_monitoring()

        # 初始化状态
        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        current_state = initial_state

        # 持续运行（无限循环，直到手动停止）
        # 实际应用中应该有停止机制
        max_steps = 100000  # 限制步数用于演示

        for step_idx in range(max_steps):
            # 执行单步
            current_state = engine.step(current_state)

            # 更新监测指标
            self._dynamics_monitor.update(current_state)

            # 定期报告
            if step_idx % self.config.continuous_report_interval == 0:
                indicators = self._dynamics_monitor.get_current_indicators()

                # 保存报告
                report_path = Path(self.config.report_output_dir) / f"continuous_report_step_{step_idx}.json"
                self._dynamics_monitor.save_report(str(report_path))

                if verbose:
                    logger.debug(
                        f"[Step {step_idx}] "
                        f"ρ(τ)={indicators.autocorrelation_rho:.4f}, "
                        f"λ_max={indicators.lyapunov_lambda_mean:.6f}, "
                        f"E_self={indicators.self_prediction_error_mean:.6f}"
                    )

        result.dynamics_result = self._dynamics_monitor.get_current_indicators()
        self._dynamics_monitor.stop_monitoring()

        # 综合判定（基于最后一次指标）
        result.overall_passed = all([
            result.dynamics_result.autocorrelation_passed,
            result.dynamics_result.lyapunov_passed,
            result.dynamics_result.self_prediction_passed
        ])

        result.overall_score = result.dynamics_result.to_dict()["overall_health"]["dynamics_score"] if hasattr(result.dynamics_result, 'to_dict') else 0.0

        return result

    def _run_p0_only_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        仅P0级验证

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[P0_ONLY验证模式] 仅核心动力学验证...")

        result = ValidationResult(validation_mode=ValidationMode.P0_ONLY)

        self._p0_validator = P0Validation(
            engine,
            self.global_config,
            self.config.p0_config,
            self.device
        )

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        result.p0_result = self._p0_validator.run_full_validation(
            initial_state,
            verbose=verbose
        )
        result.p0_passed = result.p0_result.is_passed
        result.overall_passed = result.p0_passed
        result.overall_score = result.p0_result.overall_score

        return result

    def _run_pattern_detection_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        仅模式检测

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[PATTERN_DETECTION验证模式] 仅模式检测...")

        result = ValidationResult(validation_mode=ValidationMode.PATTERN_DETECTION)

        # 动力学监测
        self._dynamics_monitor = DynamicsMonitoring(
            engine,
            self.global_config,
            self.config.dynamics_config,
            self.device
        )

        self._dynamics_monitor.start_monitoring()

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        # 运行一段时间获取动力学指标
        current_state = initial_state
        for step_idx in range(5000):
            current_state = engine.step(current_state)
            self._dynamics_monitor.update(current_state)

        result.dynamics_result = self._dynamics_monitor.get_current_indicators()
        self._dynamics_monitor.stop_monitoring()

        # 行为学指标判定
        self._behavioral_metrics = BehavioralMetrics(
            engine,
            self.global_config,
            self.config.behavioral_config,
            self.device
        )

        result.behavioral_result = self._behavioral_metrics.run_full_evaluation(
            current_state,
            verbose=verbose
        )

        # 综合判定
        result.p2_passed = result.behavioral_result.emergence_detected
        result.overall_passed = result.p2_passed
        result.overall_score = result.behavioral_result.emergence_score
        result.emergence_detected = result.behavioral_result.emergence_detected

        return result

    _run_emergence_validation = _run_pattern_detection_validation

    def _run_p1_validation(
        self,
        engine: IntegrationEngine,
        verbose: bool
    ) -> Dict[str, Any]:
        """
        P1级验证（功能模块验证）

        包括：
        - DMN功能验证
        - 工作记忆验证
        - L2独立性验证

        Args:
            engine: IntegrationEngine实例
            verbose: 详细日志

        Returns:
            P1级验证结果字典
        """
        if verbose:
            logger.info("  P1级验证（功能模块验证）...")

        p1_result = {
            "passed": False,
            "total_metrics": 0,
            "passed_metrics": 0,
            "dmn": {},
            "working_memory": {},
            "l2_independence": {}
        }

        all_passed = []

        # ===== DMN功能验证 =====
        if verbose:
            logger.info("    - DMN功能验证...")

        dmn_results = self._validate_dmn(engine, verbose)
        p1_result["dmn"] = dmn_results
        all_passed.append(dmn_results["passed"])

        # ===== 工作记忆验证 =====
        if verbose:
            logger.info("    - 工作记忆验证...")

        wm_results = self._validate_working_memory(engine, verbose)
        p1_result["working_memory"] = wm_results
        all_passed.append(wm_results["passed"])

        # ===== L2独立性验证 =====
        if verbose:
            logger.info("    - L2独立性验证...")

        l2_results = self._validate_l2_independence(engine, verbose)
        p1_result["l2_independence"] = l2_results
        all_passed.append(l2_results["passed"])

        # 统计指标
        p1_result["total_metrics"] = sum([
            len(dmn_results.get("metrics", [])),
            len(wm_results.get("metrics", [])),
            len(l2_results.get("metrics", []))
        ])
        p1_result["passed_metrics"] = sum([
            sum(1 for m in dmn_results.get("metrics", []) if m["passed"]),
            sum(1 for m in wm_results.get("metrics", []) if m["passed"]),
            sum(1 for m in l2_results.get("metrics", []) if m["passed"])
        ])

        # 综合判定
        p1_result["passed"] = all(all_passed)

        if verbose:
            logger.info(
                f"  P1级验证完成: "
                f"passed={p1_result['passed']}, "
                f"metrics={p1_result['passed_metrics']}/{p1_result['total_metrics']}"
            )

        return p1_result

    def _validate_dmn(self, engine: IntegrationEngine, verbose: bool) -> Dict[str, Any]:
        """
        DMN功能验证

        Args:
            engine: IntegrationEngine实例
            verbose: 详细日志

        Returns:
            DMN验证结果字典
        """
        metrics = []
        dmn = engine.dmn

        if dmn is None:
            return {
                "passed": False,
                "description": "DMN系统未初始化",
                "metrics": []
            }

        # 指标1: 混沌注入信号有效性验证
        initial_E_fast = torch.randn(engine.engine_config.fast_dim) * 0.1
        B_chaos = dmn.step(dt=0.01, E_fast=initial_E_fast)

        signal_norm = torch.norm(B_chaos).item()
        signal_valid = 0.001 < signal_norm < 100.0
        metrics.append({
            "name": "混沌注入信号有效性",
            "value": signal_norm,
            "threshold": "0.001 < norm < 100.0",
            "passed": signal_valid,
            "description": f"混沌注入信号范数={signal_norm:.4f}"
        })

        # 指标2: 吸引子切换机制验证
        if hasattr(dmn, 'attractor_manager') and dmn.attractor_manager:
            initial_switches = len(dmn.attractor_manager.switch_history)
            # 强制切换吸引子
            dmn.force_attractor_switch()
            after_switches = len(dmn.attractor_manager.switch_history)
            switch_occurred = after_switches > initial_switches
            metrics.append({
                "name": "吸引子切换机制",
                "value": f"{initial_switches} -> {after_switches}",
                "threshold": "switch_count > 0",
                "passed": switch_occurred,
                "description": f"吸引子切换次数增加: {after_switches - initial_switches}"
            })
        else:
            metrics.append({
                "name": "吸引子切换机制",
                "value": "N/A",
                "threshold": "attractor_manager exists",
                "passed": False,
                "description": "attractor_manager未初始化"
            })

        # 指标3: 混沌信号与快变量耦合验证
        # 运行DMN一段时间，检查核心子空间方差变化
        dmn.reset(seed=42)
        dmn.initialize()
        prev_variance = torch.var(dmn.state.E_fast_core).item() if dmn.state.E_fast_core is not None else 0.0

        for _ in range(100):
            dmn.step(dt=0.01)

        current_variance = torch.var(dmn.state.E_fast_core).item() if dmn.state.E_fast_core is not None else 0.0
        variance_increased = current_variance > prev_variance * 0.5
        metrics.append({
            "name": "混沌信号与快变量耦合",
            "value": f"{prev_variance:.6f} -> {current_variance:.6f}",
            "threshold": "variance > initial * 0.5",
            "passed": variance_increased,
            "description": f"核心子空间方差变化: {current_variance / (prev_variance + 1e-10):.2f}x"
        })

        # 指标4: DMN稳定性验证
        stability_maintained = dmn.state.is_stable if dmn.state else False
        metrics.append({
            "name": "DMN系统稳定性",
            "value": stability_maintained,
            "threshold": "is_stable == True",
            "passed": stability_maintained,
            "description": f"DMN系统状态稳定: {stability_maintained}"
        })

        # 指标5: 自适应增益控制验证
        if hasattr(dmn, 'chaos_injector') and dmn.chaos_injector:
            gain_initial = dmn.chaos_injector.current_gain
            # 扰动核心子空间
            dmn.state.E_fast_core = torch.randn_like(dmn.state.E_fast_core) * 2.0
            # 运行几步让增益自适应
            for _ in range(50):
                dmn.step(dt=0.01)
            gain_final = dmn.chaos_injector.current_gain
            gain_adapted = abs(gain_final - gain_initial) > 1e-6 or abs(gain_final - gain_initial) / (abs(gain_initial) + 1e-10) > 0.01
            metrics.append({
                "name": "自适应增益控制",
                "value": f"{gain_initial:.4f} -> {gain_final:.4f}",
                "threshold": "gain change > 1e-6 or relative change > 1%",
                "passed": gain_adapted,
                "description": f"增益自适应变化: {gain_final - gain_initial:.6f}"
            })
        else:
            metrics.append({
                "name": "自适应增益控制",
                "value": "N/A",
                "threshold": "chaos_injector exists",
                "passed": False,
                "description": "chaos_injector未初始化"
            })

        passed = all(m["passed"] for m in metrics)

        if verbose:
            for m in metrics:
                status = "✓" if m["passed"] else "✗"
                logger.debug(f"      {status} {m['name']}: {m['description']}")

        return {
            "passed": passed,
            "description": "DMN功能验证" if passed else "DMN功能验证失败",
            "metrics": metrics
        }

    def _validate_working_memory(self, engine: IntegrationEngine, verbose: bool) -> Dict[str, Any]:
        """
        工作记忆验证

        Args:
            engine: IntegrationEngine实例
            verbose: 详细日志

        Returns:
            工作记忆验证结果字典
        """
        metrics = []

        # 创建工作记忆实例进行验证
        from chronos_core.memory.work_memory import WorkingMemory, ChunkType

        wm = WorkingMemory(
            capacity=7,
            fast_dim=engine.engine_config.fast_dim,
            chunk_dim=256,
            decay_time_constant=10.0,
            min_activation=0.01,
            device=self.device
        )

        # 指标1: 工作记忆容量约束验证
        # 创建超过容量的组块，验证容量约束生效
        test_state = torch.randn(engine.engine_config.fast_dim)

        for i in range(10):
            wm.create_chunk(
                source_state=test_state,
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=1.0 - i * 0.08
            )

        active_chunks = wm.get_active_chunks()
        capacity_respected = len(active_chunks) <= wm.capacity
        metrics.append({
            "name": "工作记忆容量约束",
            "value": f"{len(active_chunks)}/{wm.capacity}",
            "threshold": f"active <= {wm.capacity}",
            "passed": capacity_respected,
            "description": f"激活组块数={len(active_chunks)}, 容量上限={wm.capacity}"
        })

        # 指标2: 组块创建和检索验证
        test_chunk = wm.create_chunk(
            source_state=test_state,
            chunk_type=ChunkType.SEMANTIC,
            initial_activation=1.0
        )
        retrieved_chunk = wm.get_chunk(test_chunk.chunk_id)
        chunk_retrievable = retrieved_chunk is not None
        metrics.append({
            "name": "组块创建和检索",
            "value": chunk_retrievable,
            "threshold": "retrieved_chunk is not None",
            "passed": chunk_retrievable,
            "description": f"组块{test_chunk.chunk_id}创建并成功检索"
        })

        # 指标3: 激活强度衰减机制验证
        if retrieved_chunk:
            initial_activation = wm.activation_strength.get_activation(retrieved_chunk.chunk_id)
            # 模拟时间流逝
            wm.update_activations(delta_time=30.0)
            final_activation = wm.activation_strength.get_activation(retrieved_chunk.chunk_id)
            activation_decayed = final_activation < initial_activation * 0.9
            metrics.append({
                "name": "激活强度衰减机制",
                "value": f"{initial_activation:.4f} -> {final_activation:.4f}",
                "threshold": "final < initial * 0.9",
                "passed": activation_decayed,
                "description": f"30秒后激活强度衰减: {100 * (1 - final_activation / initial_activation):.1f}%"
            })
        else:
            metrics.append({
                "name": "激活强度衰减机制",
                "value": "N/A",
                "threshold": "chunk exists",
                "passed": False,
                "description": "无法测试衰减，组块检索失败"
            })

        # 指标4: 组块恢复机制验证
        # 创建一个组块，让其衰减到休眠状态，然后恢复
        recovery_chunk = wm.create_chunk(
            source_state=test_state * 2,
            chunk_type=ChunkType.EMOTIONAL,
            initial_activation=0.3
        )
        chunk_id = recovery_chunk.chunk_id

        # 让其衰减
        wm.update_activations(delta_time=60.0)

        # 尝试恢复
        restored_chunk = wm.restore_chunk(chunk_id, initial_activation=0.5)
        restoration_successful = restored_chunk is not None
        metrics.append({
            "name": "组块恢复机制",
            "value": restoration_successful,
            "threshold": "restored_chunk is not None",
            "passed": restoration_successful,
            "description": f"组块{chunk_id}恢复成功: {restoration_successful}"
        })

        # 指标5: 容量约束强制执行验证
        # 创建大量组块，验证低激活组块被标记为休眠
        for i in range(15):
            wm.create_chunk(
                source_state=test_state * (i + 1),
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=0.1
            )

        dormant_chunks = wm.get_dormant_chunks()
        dormant_exists = len(dormant_chunks) > 0
        metrics.append({
            "name": "容量约束强制执行",
            "value": f"{len(dormant_chunks)} dormant",
            "threshold": "dormant_count > 0",
            "passed": dormant_exists,
            "description": f"超出容量后休眠组块数={len(dormant_chunks)}"
        })

        passed = all(m["passed"] for m in metrics)

        if verbose:
            for m in metrics:
                status = "✓" if m["passed"] else "✗"
                logger.debug(f"      {status} {m['name']}: {m['description']}")

        return {
            "passed": passed,
            "description": "工作记忆验证通过" if passed else "工作记忆验证失败",
            "metrics": metrics
        }

    def _validate_l2_independence(self, engine: IntegrationEngine, verbose: bool) -> Dict[str, Any]:
        """
        L2独立性验证

        Args:
            engine: IntegrationEngine实例
            verbose: 详细日志

        Returns:
            L2独立性验证结果字典
        """
        metrics = []

        # 创建元认知系统进行验证
        from chronos_core.core.meta_cognitive.meta_cognitive_system import (
            MetaCognitiveSystem,
            MetaCognitiveSystemConfig
        )

        meta_system = MetaCognitiveSystem(
            config=MetaCognitiveSystemConfig(device=self.device),
            dim_config=engine.global_config.dim,
            meta_config=engine.global_config.meta_cognitive,
            memory_config=engine.global_config.memory_temporal,
            global_config=engine.global_config,
            device=self.device
        )

        # 指标1: L2物理隔离验证
        if hasattr(meta_system, 'l2_layer') and meta_system.l2_layer:
            is_isolated, isolation_errors = meta_system.l2_layer.check_physical_isolation()
            metrics.append({
                "name": "L2物理隔离",
                "value": is_isolated,
                "threshold": "is_isolated == True",
                "passed": is_isolated,
                "description": f"L2层物理隔离验证: {is_isolated}, 错误: {isolation_errors}"
            })
        else:
            metrics.append({
                "name": "L2物理隔离",
                "value": "N/A",
                "threshold": "l2_layer exists",
                "passed": False,
                "description": "L2层未初始化"
            })

        # 指标2: 消融测试功能验证
        initial_ablation_state = meta_system.is_ablation_active()
        meta_system.start_ablation_test()
        ablation_active = meta_system.is_ablation_active()
        meta_system.end_ablation_test()
        ablation_inactive = not meta_system.is_ablation_active()

        ablation_works = ablation_active and ablation_inactive and not initial_ablation_state
        metrics.append({
            "name": "消融测试功能",
            "value": f"{initial_ablation_state} -> {ablation_active} -> {ablation_inactive}",
            "threshold": "start/end work correctly",
            "passed": ablation_works,
            "description": f"消融测试状态转换正确: {initial_ablation_state} -> {ablation_active} -> {ablation_inactive}"
        })

        # 指标3: 调控信号生成验证
        semantic_input = torch.randn(engine.global_config.dim.semantic_dim, device=self.device)
        physical_input = torch.randn(engine.global_config.dim.physical_dim, device=self.device)

        # 运行几步让系统稳定
        for _ in range(20):
            outputs = meta_system.forward(
                semantic_input=semantic_input,
                physical_input=physical_input,
                apply_regulation=True
            )

        # 检查是否生成了调控信号
        l2_control = outputs.get("l2_control")
        control_signal_exists = l2_control is not None and l2_control.shape[0] > 0
        metrics.append({
            "name": "调控信号生成",
            "value": f"dim={l2_control.shape[0] if control_signal_exists else 0}",
            "threshold": "control_signal is not None",
            "passed": control_signal_exists,
            "description": f"L2调控信号生成: {control_signal_exists}, 维度={l2_control.shape[0] if control_signal_exists else 0}"
        })

        # 指标4: L2调控对快变量影响验证
        # 比较有/无L2调控时的状态变化
        meta_system.reset()

        # 测试带L2调控
        test_with_l2 = meta_system.test_with_l2(
            semantic_input=semantic_input,
            physical_input=physical_input,
            num_steps=50
        )
        performance_with_l2 = test_with_l2["mean_performance"]

        # 测试不带L2调控（消融）
        meta_system.reset()
        test_without_l2 = meta_system.test_without_l2(
            semantic_input=semantic_input,
            physical_input=physical_input,
            num_steps=50
        )
        performance_without_l2 = test_without_l2["mean_performance"]

        # L2应该能产生可观测的影响
        regulation_effective = abs(performance_with_l2 - performance_without_l2) > 0.001
        metrics.append({
            "name": "L2调控对快变量影响",
            "value": f"with={performance_with_l2:.4f}, without={performance_without_l2:.4f}",
            "threshold": "|with - without| > 0.001",
            "passed": regulation_effective,
            "description": f"L2调控效果差异: {abs(performance_with_l2 - performance_without_l2):.6f}"
        })

        # 指标5: L1-L2信息流验证
        # 验证L1状态能正确传递到L2
        meta_system.reset()

        for _ in range(10):
            outputs = meta_system.forward(
                semantic_input=semantic_input,
                physical_input=physical_input,
                apply_regulation=True
            )

        # 检查L1状态缓存和L2控制信号
        l1_state_cached = meta_system._l1_state_cache is not None
        l2_control_cached = meta_system._l2_control_cache is not None
        info_flow_ok = l1_state_cached and l2_control_cached
        metrics.append({
            "name": "L1-L2信息流",
            "value": f"L1 cached={l1_state_cached}, L2 cached={l2_control_cached}",
            "threshold": "both caches exist",
            "passed": info_flow_ok,
            "description": f"L1→L2信息流正常: L1缓存={l1_state_cached}, L2缓存={l2_control_cached}"
        })

        passed = all(m["passed"] for m in metrics)

        if verbose:
            for m in metrics:
                status = "✓" if m["passed"] else "✗"
                logger.debug(f"      {status} {m['name']}: {m['description']}")

        return {
            "passed": passed,
            "description": "L2独立性验证通过" if passed else "L2独立性验证失败",
            "metrics": metrics
        }

    def save_final_report(
        self,
        result: ValidationResult,
        verbose: bool = True
    ) -> None:
        """
        保存最终验证报告

        Args:
            result: 验证结果
            verbose: 是否输出日志
        """
        output_dir = Path(self.config.report_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON格式报告
        json_path = output_dir / f"validation_report_{result.validation_mode.value}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        result.report_path = str(json_path)

        if verbose:
            logger.info(f"验证报告保存至: {json_path}")

        # 保存Markdown格式报告
        if self.config.report_format == "markdown":
            md_path = output_dir / f"validation_report_{result.validation_mode.value}.md"

            report_md = self._generate_final_markdown_report(result)

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(report_md)

            if verbose:
                logger.info(f"Markdown报告保存至: {md_path}")

        # 保存可视化（如果有）
        if self.config.save_plots:
            if self._p0_validator and result.p0_result:
                plot_path = output_dir / "p0_trajectory.png"
                self._p0_validator.visualize_trajectory(str(plot_path))

            if self._dynamics_monitor:
                plot_path = output_dir / "dynamics_visualization.png"
                self._dynamics_monitor.visualize_all(str(plot_path))

            if self._behavioral_metrics and result.behavioral_result:
                plot_path = output_dir / "behavioral_metrics.png"
                if len(result.behavioral_result.intent_entropy_history) > 0:
                    self._behavioral_metrics.visualize_intent_entropy(str(plot_path))

                if len(result.behavioral_result.l2_recovery_curve) > 0:
                    plot_path = output_dir / "l2_recovery.png"
                    self._behavioral_metrics.visualize_l2_recovery(
                        result.behavioral_result.l2_recovery_curve,
                        str(plot_path)
                    )

    def _generate_final_markdown_report(self, result: ValidationResult) -> str:
        """生成最终Markdown格式报告"""
        report = f"""# Chronos-Self 验证系统最终报告

## 验证概览

- **验证模式**: {result.validation_mode.value}
- **验证时间**: {result.validation_time:.2f}秒
- **计算设备**: {self.device}

## 总体结果

- **验证状态**: {'✓ 通过' if result.overall_passed else '❌ 失败'}
- **总体得分**: {result.overall_score:.2f}
- **涌现判定**: {'✓ 涌现' if result.emergence_detected else '❌ 未涌现'}

## 详细验证结果

"""

        # P0级验证
        if result.p0_result:
            report += """### P0级验证（核心动力学）

"""
            report += f"- **状态**: {'✓ 通过' if result.p0_passed else '❌ 失败'}\n"
            report += f"- **得分**: {result.p0_result.overall_score:.2f}\n"
            report += f"- **72小时开环运行**: {'✓' if result.p0_result.open_loop_passed else '✗'}\n"
            report += f"- **慢变量漂移率**: {'✓' if result.p0_result.drift_passed else '✗'}\n"
            report += f"- **李雅普诺夫指数**: {'✓' if result.p0_result.lyapunov_passed else '✗'}\n"
            report += f"- **动力学对齐**: {'✓' if result.p0_result.alignment_passed else '✗'}\n"

        # P1级验证
        if result.p1_result:
            report += """### P1级验证（功能模块）

"""
            report += f"- **状态**: {'✓ 通过' if result.p1_passed else '❌ 失败'}\n"
            if result.p1_result.get("dmn"):
                report += f"- **DMN功能**: {'✓' if result.p1_result['dmn']['passed'] else '✗'}\n"
            if result.p1_result.get("working_memory"):
                report += f"- **工作记忆**: {'✓' if result.p1_result['working_memory']['passed'] else '✗'}\n"
            if result.p1_result.get("l2_independence"):
                report += f"- **L2独立性**: {'✓' if result.p1_result['l2_independence']['passed'] else '✗'}\n"

        # P2级验证（动力学监测）
        if result.dynamics_result:
            report += """### P2级验证 - 动力学监测

"""
            report += f"- **自相关系数**: {'✓' if result.dynamics_result.autocorrelation_passed else '✗'} "
            report += f"(ρ={result.dynamics_result.autocorrelation_rho:.4f})\n"
            report += f"- **李雅普诺夫指数**: {'✓' if result.dynamics_result.lyapunov_passed else '✗'} "
            report += f"(λ={result.dynamics_result.lyapunov_lambda_mean:.6f})\n"
            report += f"- **自预测误差**: {'✓' if result.dynamics_result.self_prediction_passed else '✗'} "
            report += f"(E={result.dynamics_result.self_prediction_error_mean:.6f})\n"

        # P2级验证（行为学指标）
        if result.behavioral_result:
            report += """### P2级验证 - 行为学指标

"""
            report += f"- **意图熵跃迁**: {'✓' if result.behavioral_result.intent_entropy_passed else '✗'}\n"
            report += f"- **知识迁移**: {'✓' if result.behavioral_result.transfer_passed else '✗'}\n"
            report += f"- **L2恢复**: {'✓' if result.behavioral_result.l2_recovery_passed else '✗'}\n"
            report += f"- **涌现判定**: {'✓' if result.behavioral_result.emergence_detected else '✗'}\n"

        # 结论
        report += """## 结论

"""
        if result.overall_passed and result.emergence_detected:
            report += """✓✓✓ **验证全部通过，涌现判定成功**

系统已通过：
- P0级核心动力学验证
- P1级功能模块验证
- P2级涌现判定验证

系统表现出完整的自我指涉动力学特性和认知涌现特性。
"""
        elif result.overall_passed:
            report += """✓ **验证通过，但未达到涌现判定标准**

系统已通过验证，但涌现特性尚未完全显现。
建议继续运行系统以观察涌现现象。
"""
        else:
            report += """❌ **验证失败**

系统未通过验证，需要检查：
"""
            if result.p0_result and not result.p0_passed:
                report += "- P0级核心动力学验证\n"
            if result.p1_result and not result.p1_passed:
                report += "- P1级功能模块验证\n"
            if (result.dynamics_result or result.behavioral_result) and not result.p2_passed:
                report += "- P2级涌现判定验证\n"

        return report

    def run_p0_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState] = None,
        verbose: bool = True
    ) -> P0ValidationResult:
        """
        执行P0级验证

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            P0ValidationResult
        """
        self._p0_validator = P0Validation(
            engine,
            self.global_config,
            self.config.p0_config,
            self.device
        )

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        return self._p0_validator.run_full_validation(initial_state, verbose)

    def run_p1_validation(
        self,
        engine: IntegrationEngine,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        执行P1级验证

        Args:
            engine: IntegrationEngine实例
            verbose: 详细日志

        Returns:
            P1级验证结果字典
        """
        return self._run_p1_validation(engine, verbose)

    def run_p2_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState] = None,
        verbose: bool = True
    ) -> Tuple[DynamicsIndicators, BehavioralIndicators]:
        """
        执行P2级验证（涌现判定）

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            (DynamicsIndicators, BehavioralIndicators)
        """
        # 动力学监测
        self._dynamics_monitor = DynamicsMonitoring(
            engine,
            self.global_config,
            self.config.dynamics_config,
            self.device
        )

        self._dynamics_monitor.start_monitoring()

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        current_state = initial_state
        for step_idx in range(5000):
            current_state = engine.step(current_state)
            self._dynamics_monitor.update(current_state)

        dynamics_result = self._dynamics_monitor.get_current_indicators()
        self._dynamics_monitor.stop_monitoring()

        # 行为学指标判定
        self._behavioral_metrics = BehavioralMetrics(
            engine,
            self.global_config,
            self.config.behavioral_config,
            self.device
        )

        behavioral_result = self._behavioral_metrics.run_full_evaluation(
            current_state,
            verbose=verbose
        )

        return dynamics_result, behavioral_result

    def monitor_emergence(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState] = None,
        monitoring_hours: float = 24.0,
        verbose: bool = True
    ) -> BehavioralIndicators:
        """
        涌现监测

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            monitoring_hours: 监测时长（小时）
            verbose: 详细日志

        Returns:
            BehavioralIndicators
        """
        self._behavioral_metrics = BehavioralMetrics(
            engine,
            self.global_config,
            self.config.behavioral_config,
            self.device
        )

        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
                E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
                timestamp=0.0
            )

        # 设置监测时长
        self.config.behavioral_config.intent_entropy_window = int(monitoring_hours * 3600 / engine.engine_config.default_dt)

        return self._behavioral_metrics.run_full_evaluation(initial_state, verbose)

    def generate_report(
        self,
        result: ValidationResult,
        format: Optional[str] = None
    ) -> str:
        """
        生成验证报告

        Args:
            result: 验证结果
            format: 格式（json或markdown）

        Returns:
            报告内容（字符串）
        """
        format = format or self.config.report_format

        if format == "json":
            return json.dumps(result.to_dict(), indent=2, ensure_ascii=False)
        elif format == "markdown":
            return self._generate_final_markdown_report(result)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取验证系统统计信息"""
        return {
            "current_level": self._current_level.value if self._current_level else None
        }


ValidationSystem = Validation
ValidationSystemConfig = ValidationConfig


# 导出所有验证组件
__all__ = [
    'ValidationMode',
    'ValidationLevel',
    'ValidationConfig',
    'ValidationResult',
    'Validation',
    'ValidationSystemConfig',
    'ValidationSystem',
    'P0Validation',
    'P0ValidationResult',
    'P0ValidationConfig',
    'DynamicsMonitoring',
    'DynamicsIndicators',
    'DynamicsMonitoringConfig',
    'BehavioralMetrics',
    'BehavioralIndicators',
    'BehavioralMetricsConfig'
]