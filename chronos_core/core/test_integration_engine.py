"""
测试多时间尺度连续积分引擎核心模块
=====================================

测试内容：
1. Neural ODE 求解器
2. 快变量动力学系统
3. 慢变量动力学系统
4. 耦合与稳定性机制
5. 完整积分引擎
"""

import torch
import numpy as np
import pytest
import logging
from typing import Dict, List

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    NeuralODEConfig,
    CouplingStabilityConfig,
    MemoryTemporalConfig,
    MetaCognitiveConfig,
    ChaosInjectionConfig
)

from chronos_core.core import (
    # Neural ODE 求解器
    NeuralODESolver,
    ODESolverConfig,
    DynamicsFunction,
    # 快变量动力学
    FastDynamicsSystem,
    FastDynamicsFunction,
    FastDynamicsConfig,
    # 慢变量动力学
    SlowDynamicsSystem,
    SlowDynamicsFunction,
    SlowDynamicsConfig,
    PoolingMechanism,
    SpontaneousEvolution,
    # 耦合与稳定性
    CouplingAndStabilitySystem,
    AdaptiveCouplingCoefficients,
    StabilityMonitor,
    CouplingConfig,
    # 积分引擎
    IntegrationEngine,
    IntegrationEngineConfig,
    # 状态管理
    SelfState,
    ExternalInput,
    # 默认模式网络
    DefaultModeNetwork,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDynamicsFunction(DynamicsFunction):
    """简单的动力学函数用于测试"""

    def __init__(self, decay_rate: float = 0.1):
        super().__init__()
        self.decay_rate = decay_rate

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """简单的衰减动力学：dy/dt = -decay * y"""
        return -self.decay_rate * y


def test_neural_ode_solver():
    """测试 Neural ODE 求解器"""
    logger.info("Testing Neural ODE Solver...")

    # 创建求解器
    config = ODESolverConfig(
        method="dopri5",
        atol=1e-6,
        rtol=1e-5,
        step_size=0.01
    )
    solver = NeuralODESolver(config=config)

    # 测试单步积分
    dynamics_fn = SimpleDynamicsFunction(decay_rate=0.1)
    initial_state = torch.tensor([1.0, 0.5])
    t = 0.0
    dt = 0.1

    # 单步积分
    next_state = solver.step(dynamics_fn, initial_state, t, dt)

    # 检查输出
    assert next_state.shape == initial_state.shape
    assert not torch.isnan(next_state).any()
    assert not torch.isinf(next_state).any()

    logger.info(f"Single step integration: initial={initial_state.tolist()}, "
                f"next={next_state.tolist()}")

    # 测试多步积分
    t_span = torch.linspace(0, 1, 10)
    trajectory = solver.integrate(dynamics_fn, initial_state, t_span)

    assert trajectory.shape[0] == 10
    assert trajectory.shape[1] == 2
    assert not torch.isnan(trajectory).any()

    logger.info(f"Multi-step integration successful: trajectory shape={trajectory.shape}")

    # 测试求解器统计
    stats = solver.get_statistics()
    assert stats["total_steps"] > 0
    assert stats["is_stable"]

    logger.info(f"Solver stats: {stats}")

    return True


def test_fast_dynamics_system():
    """测试快变量动力学系统"""
    logger.info("Testing Fast Dynamics System...")

    # 配置
    dim_config = DimensionalityConfig(
        fast_variable_dim=2048,
        slow_variable_dim=512,
        semantic_dim=512,
        physical_dim=512,
        fusion_dim=1024,
        core_subspace_dim=64
    )
    meta_config = MetaCognitiveConfig(
        l2_hidden_dim=128
    )

    # 创建快变量动力学系统
    system = FastDynamicsSystem(
        dim_config=dim_config,
        meta_config=meta_config,
        device='cpu'
    )
    system.initialize()

    # 测试单步更新
    E_fast = torch.randn(2048)
    E_slow = torch.randn(512)

    # 构建输入字典
    inputs = {
        'X_sem': torch.randn(512),
        'X_log': torch.randn(512),
        'B_chaos': torch.randn(64)
    }

    # 单步更新
    E_fast_new = system.step(E_fast, E_slow, inputs, dt=0.01, t=0.0)

    # 检查输出
    assert E_fast_new.shape == E_fast.shape
    assert not torch.isnan(E_fast_new).any()
    assert not torch.isinf(E_fast_new).any()

    logger.info(f"Fast dynamics step: "
                f"initial_norm={torch.norm(E_fast).item():.4f}, "
                f"new_norm={torch.norm(E_fast_new).item():.4f}")

    # 测试统计信息
    stats = system.get_statistics()
    assert stats["initialized"]
    assert stats["fast_dim"] == 2048

    logger.info(f"Fast dynamics stats: initialized={stats['initialized']}")

    return True


def test_slow_dynamics_system():
    """测试慢变量动力学系统"""
    logger.info("Testing Slow Dynamics System...")

    # 配置
    dim_config = DimensionalityConfig(
        fast_variable_dim=2048,
        slow_variable_dim=512
    )
    coupling_config = CouplingStabilityConfig(
        elastic_restoration_coeff=0.001,
        coupling_adaptation_coeff=0.01,
        stability_threshold=1e6
    )
    temporal_config = MemoryTemporalConfig(
        slow_update_frequency=100
    )

    # 创建慢变量动力学系统
    system = SlowDynamicsSystem(
        dim_config=dim_config,
        coupling_config=coupling_config,
        temporal_config=temporal_config,
        device='cpu'
    )
    system.initialize()

    # 测试池化机制
    E_fast = torch.randn(2048)
    E_slow = torch.randn(512)

    # 检查是否应该更新
    for i in range(100):
        should_update = system.should_update_slow()
        if i == 99:
            assert should_update  # 第100步应该更新
        else:
            assert not should_update  # 其他步不应该更新

    logger.info("Low-frequency update mechanism verified")

    # 单步更新（使用快变量）
    system.reset_fast_counter()  # 重置计数器
    system.should_update_slow()  # 使计数器达到更新条件

    E_slow_new = system.step(E_slow, E_fast, dt_slow=1.0, t=0.0)

    # 检查输出
    assert E_slow_new.shape == E_slow.shape
    assert not torch.isnan(E_slow_new).any()
    assert not torch.isinf(E_slow_new).any()

    logger.info(f"Slow dynamics step: "
                f"initial_norm={torch.norm(E_slow).item():.4f}, "
                f"new_norm={torch.norm(E_slow_new).item():.4f}")

    # 测试统计信息
    stats = system.get_statistics()
    assert stats["initialized"]
    assert stats["slow_dim"] == 512

    logger.info(f"Slow dynamics stats: initialized={stats['initialized']}")

    return True


def test_coupling_and_stability():
    """测试耦合与稳定性机制"""
    logger.info("Testing Coupling and Stability System...")

    # 配置
    dim_config = DimensionalityConfig(
        fast_variable_dim=2048,
        slow_variable_dim=512
    )
    coupling_config = CouplingStabilityConfig(
        coupling_adaptation_coeff=0.01,
        stability_threshold=1e6,
        coupling_upper_bound=10.0,
        lyapunov_threshold=0.1
    )

    # 创建耦合与稳定性系统
    system = CouplingAndStabilitySystem(
        coupling_config=coupling_config,
        dim_config=dim_config,
        device='cpu'
    )
    system.initialize()

    # 测试自适应耦合系数
    E_fast = torch.randn(2048)
    coupling = system.update_coupling(E_fast)

    # 检查耦合系数范围
    assert 0.001 <= coupling <= 10.0

    logger.info(f"Adaptive coupling coefficient: {coupling:.4f}")

    # 测试稳定性监测
    E_slow = torch.randn(512)
    stability_report = system.monitor_stability(E_fast, E_slow, step_count=0)

    # 检查报告结构
    assert "is_stable" in stability_report
    assert "warnings" in stability_report

    logger.info(f"Stability report: is_stable={stability_report['is_stable']}")

    # 测试边缘混沌检查
    is_edge_of_chaos = system.check_edge_of_chaos()
    assert isinstance(is_edge_of_chaos, bool)

    logger.info(f"Edge of chaos: {is_edge_of_chaos}")

    # 测试统计信息
    report = system.get_stability_report()
    assert "coupling" in report
    assert "stability" in report

    logger.info("Coupling and stability system tests passed")

    return True


def test_pooling_mechanism():
    """测试池化机制"""
    logger.info("Testing Pooling Mechanism...")

    # 测试平均池化
    pooling_avg = PoolingMechanism(
        fast_dim=2048,
        slow_dim=512,
        method="average",
        device='cpu'
    )

    E_fast = torch.randn(2048)
    pooled_avg = pooling_avg.forward(E_fast)

    assert pooled_avg.shape == (512,)
    assert not torch.isnan(pooled_avg).any()

    logger.info(f"Average pooling: input_shape={(2048,)}, "
                f"output_shape={pooled_avg.shape}")

    # 测试注意力池化
    pooling_attn = PoolingMechanism(
        fast_dim=2048,
        slow_dim=512,
        method="attention",
        attention_heads=4,
        device='cpu'
    )

    pooled_attn = pooling_attn.forward(E_fast)

    assert pooled_attn.shape == (512,)
    assert not torch.isnan(pooled_attn).any()

    logger.info(f"Attention pooling: input_shape={(2048,)}, "
                f"output_shape={pooled_attn.shape}")

    return True


def test_integration_engine():
    """测试完整积分引擎"""
    logger.info("Testing Integration Engine...")

    # 创建全局配置
    config = ChronosConfig()
    config.device = 'cpu'  # 使用 CPU 进行测试
    config.random_seed = 42

    # 创建积分引擎
    engine_config = IntegrationEngineConfig(
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
        default_dt=0.01,
        slow_update_frequency=config.memory_temporal.slow_update_frequency,
        stability_check_interval=10,
        log_interval=10
    )

    engine = IntegrationEngine(
        config=config,
        engine_config=engine_config,
        device='cpu',
        seed=42
    )
    engine.initialize()

    # 创建初始状态
    initial_state = SelfState(
        E_fast=torch.randn(config.dim.fast_variable_dim),
        E_slow=torch.randn(config.dim.slow_variable_dim),
        timestamp=0.0
    )

    # 测试单步积分
    new_state = engine.step(initial_state, dt=0.01)

    # 检查输出
    assert new_state.E_fast.shape[0] == config.dim.fast_variable_dim
    assert new_state.E_slow.shape[0] == config.dim.slow_variable_dim
    assert new_state.timestamp == initial_state.timestamp + 0.01

    logger.info(f"Engine single step: "
                f"fast_norm={new_state.get_fast_norm():.4f}, "
                f"slow_norm={new_state.get_slow_norm():.4f}, "
                f"time={new_state.timestamp:.2f}")

    # 测试多步积分
    num_steps = 100
    trajectory = engine.integrate(
        new_state,  # 从新状态开始
        num_steps=num_steps,
        record_trajectory=True
    )

    assert len(trajectory) == num_steps
    assert trajectory[-1].timestamp > trajectory[0].timestamp

    logger.info(f"Multi-step integration: {len(trajectory)} steps, "
                f"final_time={trajectory[-1].timestamp:.2f}")

    # 测试状态监测
    monitoring = engine.get_state_monitoring()
    assert monitoring["step_count"] > 0
    assert monitoring["fast_updates"] > 0
    assert monitoring["slow_updates"] > 0

    logger.info(f"Engine monitoring: "
                f"steps={monitoring['step_count']}, "
                f"fast_updates={monitoring['fast_updates']}, "
                f"slow_updates={monitoring['slow_updates']}")

    # 重置引擎
    engine.reset()
    assert engine.step_count == 0

    logger.info("Integration engine tests passed")

    return True


def test_stability_and_edge_of_chaos():
    """测试数值稳定性和边缘混沌稳态"""
    logger.info("Testing Numerical Stability and Edge of Chaos...")

    # 创建配置（使用更严格的稳定性参数）
    config = ChronosConfig()
    config.device = 'cpu'
    config.coupling_stability.stability_threshold = 1e5
    config.coupling_stability.lyapunov_threshold = 0.1

    # 创建引擎
    engine = IntegrationEngine(config=config, device='cpu')
    engine.initialize()

    # 创建初始状态（较小范数）
    initial_state = SelfState(
        E_fast=torch.randn(config.dim.fast_variable_dim) * 0.1,
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
        timestamp=0.0
    )

    # 运行1000步，检查稳定性
    current_state = initial_state
    stable_steps = 0
    total_steps = 1000

    for i in range(total_steps):
        # 执行单步
        current_state = engine.step(current_state, dt=0.01)

        # 检查状态有效性
        is_valid, errors = current_state.validate()
        if not is_valid:
            logger.warning(f"Invalid state at step {i}: {errors}")
            break

        stable_steps += 1

        # 检查范数增长
        fast_norm = current_state.get_fast_norm()
        slow_norm = current_state.get_slow_norm()

        if fast_norm > config.coupling_stability.stability_threshold:
            logger.warning(f"Fast norm too large at step {i}: {fast_norm:.4e}")
            break

        if slow_norm > config.coupling_stability.stability_threshold / 10:
            logger.warning(f"Slow norm too large at step {i}: {slow_norm:.4e}")
            break

    logger.info(f"Stability test: {stable_steps}/{total_steps} stable steps")

    # 检查边缘混沌稳态
    edge_of_chaos = engine.coupling_system.check_edge_of_chaos()
    logger.info(f"Edge of chaos status: {edge_of_chaos}")

    # 检查稳定性报告
    stability_report = engine.coupling_system.get_stability_report()
    logger.info(f"Stability report: "
                f"is_stable={stability_report['stability']['is_stable']}, "
                f"warnings={stability_report['stability']['warning_count']}")

    assert stable_steps >= total_steps * 0.95  # 至少95%的步骤应该是稳定的

    logger.info("Stability and edge of chaos tests passed")

    return True


def test_dimension_correctness():
    """测试维度正确性"""
    logger.info("Testing Dimension Correctness...")

    config = ChronosConfig()

    # 测试快变量维度
    assert config.dim.fast_variable_dim == 2048
    logger.info(f"Fast variable dimension: {config.dim.fast_variable_dim}")

    # 测试慢变量维度
    assert config.dim.slow_variable_dim == 512
    logger.info(f"Slow variable dimension: {config.dim.slow_variable_dim}")

    # 测试核心子空间维度
    assert config.dim.core_subspace_dim == 64
    logger.info(f"Core subspace dimension: {config.dim.core_subspace_dim}")

    # 测试慢变量更新频率
    assert config.memory_temporal.slow_update_frequency == 100
    logger.info(f"Slow update frequency: {config.memory_temporal.slow_update_frequency}")

    logger.info("Dimension correctness tests passed")

    return True


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("Running All Integration Engine Tests")
    logger.info("=" * 60)

    test_results = {}

    # 测试列表
    tests = [
        ("dimension_correctness", test_dimension_correctness),
        ("neural_ode_solver", test_neural_ode_solver),
        ("fast_dynamics_system", test_fast_dynamics_system),
        ("slow_dynamics_system", test_slow_dynamics_system),
        ("pooling_mechanism", test_pooling_mechanism),
        ("coupling_and_stability", test_coupling_and_stability),
        ("integration_engine", test_integration_engine),
        ("stability_and_edge_of_chaos", test_stability_and_edge_of_chaos),
    ]

    # 执行测试
    for test_name, test_fn in tests:
        try:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running test: {test_name}")
            logger.info("=" * 60)

            result = test_fn()
            test_results[test_name] = result

            logger.info(f"Test {test_name}: PASSED")

        except Exception as e:
            logger.error(f"Test {test_name}: FAILED - {str(e)}")
            test_results[test_name] = False

    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)

    passed = sum(1 for r in test_results.values() if r)
    total = len(test_results)

    for test_name, result in test_results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    return test_results


if __name__ == "__main__":
    # 运行所有测试
    results = run_all_tests()

    # 如果有测试失败，抛出异常
    if not all(results.values()):
        failed_tests = [name for name, result in results.items() if not result]
        raise AssertionError(f"Some tests failed: {failed_tests}")

    logger.info("\n" + "=" * 60)
    logger.info("All tests PASSED successfully!")
    logger.info("=" * 60)