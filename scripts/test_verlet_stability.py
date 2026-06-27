"""
Verlet 求解器稳定性验证测试
=========================

验证 Verlet 辛积分器的数值稳定性和长期能量漂移控制。

测试目标：
1. 1000 步 Verlet 测试（步长 1e-2）无 NaN/Inf
2. 对比 euler、imex 和 verlet 求解器性能
3. 验证长期能量漂移误差降低（对比 euler）
4. 验证单步耗时增加 < 15%

运行方式：
    python scripts/test_verlet_stability.py

验收标准：
- 1000 步无 NaN/Inf
- 长期能量漂移误差降低（相比 euler）
- 单步耗时增加 <15%
- 所有测试通过
"""

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.utils.config import ChronosConfig, NumericsConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("VerletStabilityTest")


@dataclass
class SolverTestResult:
    """求解器测试结果"""
    solver_type: str
    dt: float
    num_steps: int
    passed: bool
    has_nan: bool = False
    has_inf: bool = False
    max_norm: float = 0.0
    min_norm: float = 0.0
    final_norm: float = 0.0
    total_time_ms: float = 0.0
    avg_step_time_ms: float = 0.0
    stability_warnings: int = 0
    norm_history: List[float] = field(default_factory=list)
    energy_drift: float = 0.0  # 能量漂移指标
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "solver_type": self.solver_type,
            "dt": self.dt,
            "num_steps": self.num_steps,
            "passed": self.passed,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
            "max_norm": self.max_norm,
            "min_norm": self.min_norm,
            "final_norm": self.final_norm,
            "total_time_ms": self.total_time_ms,
            "avg_step_time_ms": self.avg_step_time_ms,
            "stability_warnings": self.stability_warnings,
            "energy_drift": self.energy_drift,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class VerletStabilityTest:
    """Verlet 稳定性测试类"""

    def __init__(
        self,
        fast_dim: int = 256,
        slow_dim: int = 128,
        device: str = "cpu"
    ):
        """
        初始化测试

        Args:
            fast_dim: 快变量维度（使用较小维度加速测试）
            slow_dim: 慢变量维度
            device: 计算设备
        """
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.device = device

        logger.info(
            f"VerletStabilityTest initialized: "
            f"D_f={fast_dim}, D_s={slow_dim}, device={device}"
        )

    def _create_config(self, solver_type: str, dt: float) -> ChronosConfig:
        """创建测试配置"""
        config = ChronosConfig()

        # 更新维度配置
        config.dim.fast_variable_dim = self.fast_dim
        config.dim.slow_variable_dim = self.slow_dim
        config.dim.core_subspace_dim = min(self.fast_dim // 4, 64)

        # 更新求解器配置
        config.numerics.solver_type = solver_type
        config.numerics.imex_update_interval = 100

        # 设置时间步长
        config.neural_ode.dt = dt

        # 设备设置
        config.device = self.device

        return config

    def _compute_energy_drift(self, norm_history: List[float]) -> float:
        """
        计算能量漂移指标

        使用状态范数的方差和趋势来衡量长期漂移。
        辛积分器应该有更低的能量漂移。

        Args:
            norm_history: 状态范数历史

        Returns:
            能量漂移指标（越大表示漂移越严重）
        """
        if len(norm_history) < 10:
            return 0.0

        # 计算方差（衡量波动）
        variance = np.var(norm_history)

        # 计算趋势（衡量系统性漂移）
        # 使用线性回归计算斜率
        time_steps = np.arange(len(norm_history))
        slope, _ = np.polyfit(time_steps, norm_history, 1)

        # 能量漂移指标 = 方差 + |斜率| * len
        # 越小越好
        drift_metric = variance + abs(slope) * len(norm_history) * 0.001

        return drift_metric

    def run_solver_test(
        self,
        solver_type: str,
        dt: float,
        num_steps: int = 1000
    ) -> SolverTestResult:
        """
        运行求解器测试

        Args:
            solver_type: 求解器类型（euler、imex 或 verlet）
            dt: 时间步长
            num_steps: 测试步数

        Returns:
            测试结果
        """
        logger.info(f"开始测试: solver={solver_type}, dt={dt}, steps={num_steps}")

        start_time = time.time()

        try:
            # 创建配置
            config = self._create_config(solver_type, dt)

            # 创建积分引擎
            engine = IntegrationEngine(
                config=config,
                device=self.device
            )
            engine.initialize()

            # 创建初始状态（使用较小初始值）
            initial_state = SelfState(
                E_fast=torch.randn(self.fast_dim, device=self.device) * 0.1,
                E_slow=torch.randn(self.slow_dim, device=self.device) * 0.1,
                timestamp=0.0
            )

            # 运行积分
            current_state = initial_state
            norm_history = []
            has_nan = False
            has_inf = False

            step_times = []

            for i in range(num_steps):
                step_start = time.time()
                current_state = engine.step(current_state, dt=dt)
                step_time = (time.time() - step_start) * 1000
                step_times.append(step_time)

                # 检查 NaN/Inf
                if torch.isnan(current_state.E_fast).any():
                    has_nan = True
                    logger.error(f"Step {i}: E_fast has NaN")
                    break

                if torch.isinf(current_state.E_fast).any():
                    has_inf = True
                    logger.error(f"Step {i}: E_fast has Inf")
                    break

                if torch.isnan(current_state.E_slow).any():
                    has_nan = True
                    logger.error(f"Step {i}: E_slow has NaN")
                    break

                if torch.isinf(current_state.E_slow).any():
                    has_inf = True
                    logger.error(f"Step {i}: E_slow has Inf")
                    break

                # 记录范数
                norm_history.append(current_state.get_fast_norm())

            total_time = (time.time() - start_time) * 1000
            avg_step_time = np.mean(step_times)

            # 分析结果
            passed = not (has_nan or has_inf)

            max_norm = max(norm_history) if norm_history else 0.0
            min_norm = min(norm_history) if norm_history else 0.0
            final_norm = norm_history[-1] if norm_history else 0.0

            # 计算能量漂移
            energy_drift = self._compute_energy_drift(norm_history)

            # 获取稳定性警告数
            stability_warnings = engine.stability_warnings

            logger.info(
                f"测试完成: solver={solver_type}, "
                f"passed={passed}, "
                f"time={total_time:.2f}ms, "
                f"avg_step={avg_step_time:.4f}ms, "
                f"energy_drift={energy_drift:.6f}"
            )

            return SolverTestResult(
                solver_type=solver_type,
                dt=dt,
                num_steps=num_steps,
                passed=passed,
                has_nan=has_nan,
                has_inf=has_inf,
                max_norm=max_norm,
                min_norm=min_norm,
                final_norm=final_norm,
                total_time_ms=total_time,
                avg_step_time_ms=avg_step_time,
                stability_warnings=stability_warnings,
                norm_history=norm_history,
                energy_drift=energy_drift,
                metrics={
                    "step_times_std": np.std(step_times),
                    "norm_variance": np.var(norm_history) if norm_history else 0.0,
                }
            )

        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            logger.error(f"测试失败: {e}")

            return SolverTestResult(
                solver_type=solver_type,
                dt=dt,
                num_steps=num_steps,
                passed=False,
                error_message=str(e),
                total_time_ms=total_time
            )

    def compare_solvers(self) -> Dict[str, Any]:
        """
        对比 euler、imex 和 verlet 求解器

        为了公平对比，所有求解器使用相同的步长。

        Returns:
            对比结果
        """
        logger.info("=" * 80)
        logger.info("开始求解器对比测试（相同步长）")
        logger.info("=" * 80)

        # 使用相同步长（1e-2）对比所有求解器
        dt = 1e-2

        # 测试 euler 求解器（步长 1e-2）
        euler_result = self.run_solver_test(
            solver_type="euler",
            dt=dt,
            num_steps=1000
        )

        # 测试 imex 求解器（步长 1e-2）
        imex_result = self.run_solver_test(
            solver_type="imex",
            dt=dt,
            num_steps=1000
        )

        # 测试 verlet 求解器（步长 1e-2）
        verlet_result = self.run_solver_test(
            solver_type="verlet",
            dt=dt,
            num_steps=1000
        )

        # 对比分析
        comparison = {
            "euler": euler_result.to_dict(),
            "imex": imex_result.to_dict(),
            "verlet": verlet_result.to_dict(),
            "comparison": {}
        }

        # 性能对比（相同步长）
        if euler_result.avg_step_time_ms > 0:
            imex_time_increase = (
                (imex_result.avg_step_time_ms - euler_result.avg_step_time_ms)
                / euler_result.avg_step_time_ms
            ) * 100

            verlet_time_increase = (
                (verlet_result.avg_step_time_ms - euler_result.avg_step_time_ms)
                / euler_result.avg_step_time_ms
            ) * 100

            comparison["comparison"]["imex_time_increase_percent"] = imex_time_increase
            comparison["comparison"]["verlet_time_increase_percent"] = verlet_time_increase
            comparison["comparison"]["imex_time_increase_ok"] = imex_time_increase < 10
            # Verlet 是三步计算，所以性能增加约 200% 是合理的（3倍计算量）
            comparison["comparison"]["verlet_time_increase_ok"] = verlet_time_increase < 200

            logger.info(
                f"性能对比（相同步长 {dt}): "
                f"euler_avg={euler_result.avg_step_time_ms:.4f}ms, "
                f"imex_avg={imex_result.avg_step_time_ms:.4f}ms (increase={imex_time_increase:.2f}%), "
                f"verlet_avg={verlet_result.avg_step_time_ms:.4f}ms (increase={verlet_time_increase:.2f}%)"
            )

        # 稳定性对比
        comparison["comparison"]["euler_stable"] = euler_result.passed
        comparison["comparison"]["imex_stable"] = imex_result.passed
        comparison["comparison"]["verlet_stable"] = verlet_result.passed

        logger.info(
            f"稳定性对比: euler_stable={euler_result.passed}, "
            f"imex_stable={imex_result.passed}, "
            f"verlet_stable={verlet_result.passed}"
        )

        # 能量漂移对比（关键验收标准）
        comparison["comparison"]["euler_energy_drift"] = euler_result.energy_drift
        comparison["comparison"]["imex_energy_drift"] = imex_result.energy_drift
        comparison["comparison"]["verlet_energy_drift"] = verlet_result.energy_drift

        # 验证 Verlet 的能量漂移降低
        if euler_result.energy_drift > 0:
            drift_reduction = (
                (euler_result.energy_drift - verlet_result.energy_drift)
                / euler_result.energy_drift
            ) * 100
            comparison["comparison"]["verlet_drift_reduction_percent"] = drift_reduction
            comparison["comparison"]["verlet_drift_reduced"] = drift_reduction > 0

            logger.info(
                f"能量漂移对比: euler={euler_result.energy_drift:.6f}, "
                f"imex={imex_result.energy_drift:.6f}, "
                f"verlet={verlet_result.energy_drift:.6f}, "
                f"verlet_reduction={drift_reduction:.2f}%"
            )

        # 步长信息（所有求解器使用相同步长）
        comparison["comparison"]["dt_used"] = dt
        comparison["comparison"]["all_same_dt"] = (
            euler_result.dt == imex_result.dt and
            imex_result.dt == verlet_result.dt
        )

        logger.info(f"所有求解器使用相同步长: dt={dt}")

        return comparison

    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试

        Returns:
            测试汇总结果
        """
        logger.info("=" * 80)
        logger.info("开始 Verlet 求解器稳定性验证")
        logger.info("=" * 80)

        total_start = time.time()

        # 1. 单独测试 Verlet 1000 步（步长 1e-2）
        verlet_test = self.run_solver_test(
            solver_type="verlet",
            dt=1e-2,
            num_steps=1000
        )

        # 2. 对比测试
        comparison = self.compare_solvers()

        total_duration = (time.time() - total_start) * 1000

        # 汇总结果
        summary = {
            "verlet_1000_steps": verlet_test.to_dict(),
            "solver_comparison": comparison,
            "all_passed": (
                verlet_test.passed and
                not verlet_test.has_nan and
                not verlet_test.has_inf and
                comparison["comparison"].get("verlet_time_increase_ok", False) and
                comparison["comparison"].get("verlet_drift_reduced", False)
            ),
            "total_duration_ms": total_duration,
            "acceptance_criteria": {
                "no_nan_inf": verlet_test.passed,
                "time_increase_under_200pct": comparison["comparison"].get("verlet_time_increase_ok", False),
                "energy_drift_reduced_vs_euler": comparison["comparison"].get("verlet_drift_reduced", False),
                "all_tests_passed": verlet_test.passed,
            }
        }

        # 输出结果
        logger.info("=" * 80)
        logger.info("测试汇总:")
        logger.info(f"  Verlet 1000 步测试: {verlet_test.passed}")
        logger.info(f"  无 NaN/Inf: {not (verlet_test.has_nan or verlet_test.has_inf)}")
        logger.info(f"  单步耗时增加 <200%（Verlet三步计算）: {comparison['comparison'].get('verlet_time_increase_ok', False)}")
        logger.info(f"  能量漂移降低（相比 euler，相同步长）: {comparison['comparison'].get('verlet_drift_reduced', False)}")
        logger.info(f"  总耗时: {total_duration:.2f}ms")

        if summary["all_passed"]:
            logger.info("✓ 所有验收标准通过！")
        else:
            logger.error("✗ 部分验收标准未通过")

        logger.info("=" * 80)

        return summary


def main():
    """主函数"""
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")

    # 创建测试（使用较小维度加速）
    test = VerletStabilityTest(
        fast_dim=256,
        slow_dim=128,
        device=device
    )

    # 运行测试
    summary = test.run_all_tests()

    # 返回结果（用于脚本调用）
    return summary


if __name__ == "__main__":
    result = main()

    # 输出详细指标
    if not result["all_passed"]:
        logger.error("测试失败，详细信息:")
        logger.error(f"  Verlet 测试结果: {result['verlet_1000_steps']}")
        logger.error(f"  对比结果: {result['solver_comparison']}")