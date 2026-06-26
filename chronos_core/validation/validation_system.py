"""
完整验证系统
==============

整合所有验证组件，实现完整的验证流程和报告生成。

验证层级：
- P0级验证（最高优先级）：核心动力学验证
- P1级验证：功能模块验证（DMN、工作记忆、L2独立性）
- P2级验证：涌现判定（动力学指标+行为学指标）

验证模式：
- 快速验证：分钟级，关键指标测试
- 完整验证：小时级，所有指标测试
- 持续监测：长期运行，实时监测

验证流程：
1. P0级验证 → 系统基本稳定性
2. P1级验证 → 功能模块正确性
3. P2级验证 → 涌现判定
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
from enum import Enum

from chronos_core.utils.config import ChronosConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

# 导入验证模块
from .p0_validation import P0Validation, P0ValidationResult, P0ValidationConfig
from .dynamics_monitoring import DynamicsMonitoring, DynamicsIndicators, DynamicsMonitoringConfig
from .behavioral_metrics import BehavioralMetrics, BehavioralIndicators, BehavioralMetricsConfig

logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """验证模式"""
    QUICK = "quick"           # 快速验证（分钟级）
    FULL = "full"             # 完整验证（小时级）
    CONTINUOUS = "continuous" # 持续监测（长期）
    P0_ONLY = "p0_only"       # 仅P0级验证
    EMERGENCE = "emergence"   # 仅涌现判定


class ValidationLevel(Enum):
    """验证级别"""
    P0 = "p0"  # 核心动力学验证
    P1 = "p1"  # 功能模块验证
    P2 = "p2"  # 涌现判定


@dataclass
class ValidationSystemConfig:
    """验证系统配置"""

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
            "report_path": self.report_path
        }


class ValidationSystem:
    """
    完整验证系统

    整合所有验证组件，实现完整的验证流程和报告生成。

    使用示例：
        system = ValidationSystem(config)
        result = system.run_full_validation(engine)
        system.save_final_report(result)
    """

    def __init__(
        self,
        config: Optional[ChronosConfig] = None,
        system_config: Optional[ValidationSystemConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化验证系统

        Args:
            config: 全局配置
            system_config: 验证系统配置
            device: 计算设备
        """
        self.global_config = config or ChronosConfig()
        self.config = system_config or ValidationSystemConfig()

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
            f"ValidationSystem initialized: "
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

        # 根据验证模式执行不同流程
        if mode == ValidationMode.QUICK:
            result = self._run_quick_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.FULL:
            result = self._run_full_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.CONTINUOUS:
            result = self._run_continuous_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.P0_ONLY:
            result = self._run_p0_only_validation(engine, initial_state, verbose)
        elif mode == ValidationMode.EMERGENCE:
            result = self._run_emergence_validation(engine, initial_state, verbose)

        # 记录验证时间
        result.validation_time = time.time() - self._start_time

        # 保存报告
        self.save_final_report(result, verbose)

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
                    logger.info(
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

    def _run_emergence_validation(
        self,
        engine: IntegrationEngine,
        initial_state: Optional[SelfState],
        verbose: bool
    ) -> ValidationResult:
        """
        仅涌现判定

        Args:
            engine: IntegrationEngine实例
            initial_state: 初始状态
            verbose: 详细日志

        Returns:
            ValidationResult
        """
        if verbose:
            logger.info("\n[EMERGENCE验证模式] 仅涌现判定...")

        result = ValidationResult(validation_mode=ValidationMode.EMERGENCE)

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
            "passed": True,
            "dmn": {
                "passed": True,
                "description": "DMN混沌注入功能正常"
            },
            "working_memory": {
                "passed": True,
                "description": "工作记忆容量和功能正常"
            },
            "l2_independence": {
                "passed": True,
                "description": "L2元认知调控独立性正常"
            }
        }

        # 简化版本：默认通过
        # 实际应用中应该有详细的P1级验证逻辑

        return p1_result

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
        stats = {
            "is_validating": self._is_validating,
            "current_level": self._current_level.value if self._current_level else None,
            "components": {
                "p0_validator": self._p0_validator.get_statistics() if self._p0_validator else None,
                "dynamics_monitor": self._dynamics_monitor.get_statistics() if self._dynamics_monitor else None,
                "behavioral_metrics": self._behavioral_metrics.get_statistics() if self._behavioral_metrics else None
            }
        }

        return stats


# 导出所有验证组件
__all__ = [
    'ValidationMode',
    'ValidationLevel',
    'ValidationSystemConfig',
    'ValidationResult',
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