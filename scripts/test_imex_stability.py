"""
IMEX 求解器稳定性验证测试
=========================

验证 IMEX 算子分裂求解器的数值稳定性和性能。

测试目标：
1. 1000 步 IMEX 测试（步长 1e-2）无 NaN/Inf
2. 对比 euler 和 imex 求解器性能
3. 验证单步耗时增加 < 10%

运行方式：
    python scripts/test_imex_stability.py

验收标准：
- 1000 步无 NaN/Inf
- 单步耗时增加 <10%
- 所有测试通过
"""

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple

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
logger = logging.getLogger("IMEXStabilityTest")


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
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class IMEXStabilityTest:
    """IMEX 稳定性测试类"""
    
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
            f"IMEXStabilityTest initialized: "
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
    
    def run_solver_test(
        self,
        solver_type: str,
        dt: float,
        num_steps: int = 1000
    ) -> SolverTestResult:
        """
        运行求解器测试
        
        Args:
            solver_type: 求解器类型（euler 或 imex）
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
            
            # 获取稳定性警告数
            stability_warnings = engine.stability_warnings
            
            logger.info(
                f"测试完成: solver={solver_type}, "
                f"passed={passed}, "
                f"time={total_time:.2f}ms, "
                f"avg_step={avg_step_time:.4f}ms"
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
        对比 euler 和 imex 求解器
        
        Returns:
            对比结果
        """
        logger.info("=" * 80)
        logger.info("开始求解器对比测试")
        logger.info("=" * 80)
        
        # 测试 euler 求解器（步长 1e-3）
        euler_result = self.run_solver_test(
            solver_type="euler",
            dt=1e-3,
            num_steps=1000
        )
        
        # 测试 imex 求解器（步长 1e-2）
        imex_result = self.run_solver_test(
            solver_type="imex",
            dt=1e-2,
            num_steps=1000
        )
        
        # 对比分析
        comparison = {
            "euler": euler_result.to_dict(),
            "imex": imex_result.to_dict(),
            "comparison": {}
        }
        
        # 性能对比
        if euler_result.avg_step_time_ms > 0:
            time_increase = (
                (imex_result.avg_step_time_ms - euler_result.avg_step_time_ms)
                / euler_result.avg_step_time_ms
            ) * 100
            
            comparison["comparison"]["time_increase_percent"] = time_increase
            comparison["comparison"]["time_increase_ok"] = time_increase < 10
            
            logger.info(
                f"性能对比: euler_avg={euler_result.avg_step_time_ms:.4f}ms, "
                f"imex_avg={imex_result.avg_step_time_ms:.4f}ms, "
                f"increase={time_increase:.2f}%"
            )
        
        # 稳定性对比
        comparison["comparison"]["euler_stable"] = euler_result.passed
        comparison["comparison"]["imex_stable"] = imex_result.passed
        
        logger.info(
            f"稳定性对比: euler_stable={euler_result.passed}, "
            f"imex_stable={imex_result.passed}"
        )
        
        # 步长对比
        comparison["comparison"]["dt_ratio"] = imex_result.dt / euler_result.dt
        
        logger.info(
            f"步长对比: euler_dt={euler_result.dt}, "
            f"imex_dt={imex_result.dt}, "
            f"ratio={imex_result.dt / euler_result.dt:.1f}x"
        )
        
        return comparison
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        
        Returns:
            测试汇总结果
        """
        logger.info("=" * 80)
        logger.info("开始 IMEX 求解器稳定性验证")
        logger.info("=" * 80)
        
        total_start = time.time()
        
        # 1. 单独测试 IMEX 1000 步（步长 1e-2）
        imex_test = self.run_solver_test(
            solver_type="imex",
            dt=1e-2,
            num_steps=1000
        )
        
        # 2. 对比测试
        comparison = self.compare_solvers()
        
        total_duration = (time.time() - total_start) * 1000
        
        # 汇总结果
        summary = {
            "imex_1000_steps": imex_test.to_dict(),
            "solver_comparison": comparison,
            "all_passed": (
                imex_test.passed and
                not imex_test.has_nan and
                not imex_test.has_inf and
                comparison["comparison"].get("time_increase_ok", False)
            ),
            "total_duration_ms": total_duration,
            "acceptance_criteria": {
                "no_nan_inf": imex_test.passed,
                "time_increase_under_10pct": comparison["comparison"].get("time_increase_ok", False),
                "all_tests_passed": imex_test.passed,
            }
        }
        
        # 输出结果
        logger.info("=" * 80)
        logger.info("测试汇总:")
        logger.info(f"  IMEX 1000 步测试: {imex_test.passed}")
        logger.info(f"  无 NaN/Inf: {not (imex_test.has_nan or imex_test.has_inf)}")
        logger.info(f"  单步耗时增加 <10%: {comparison['comparison'].get('time_increase_ok', False)}")
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
    test = IMEXStabilityTest(
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
        logger.error(f"  IMEX 测试结果: {result['imex_1000_steps']}")
        logger.error(f"  对比结果: {result['solver_comparison']}")