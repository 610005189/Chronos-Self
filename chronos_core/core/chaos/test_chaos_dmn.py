"""
混沌吸引子和 DMN 系统测试
========================

验证 Task 9-11 的实现正确性：
- LorenzAttractor 正确实现洛伦兹动力学
- RosslerAttractor 正确实现罗斯勒动力学
- ChuaAttractor 正确实现蔡氏电路动力学
- 固定随机正交投影矩阵正确生成
- ChaosInjector 正确实现混沌注入
- 自适应增益控制正确实现
- 多吸引子切换机制正确实现
- 无输入时持续动力学维持测试
"""

import torch
import numpy as np
import pytest
import logging
import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from chronos_core.core.chaos import (
    LorenzAttractor,
    RosslerAttractor,
    ChuaAttractor,
    AttractorManager,
    BaseAttractor,
    AttractorState,
)
from chronos_core.core.chaos_injector import (
    ChaosInjector,
    CoreSubspaceProjector,
)
from chronos_core.core.dmn_system import (
    DefaultModeNetwork,
    DMNConfig,
)
from chronos_core.utils.config import ChaosInjectionConfig, DimensionalityConfig

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestLorenzAttractor:
    """洛伦兹吸引子测试"""

    def test_initialization(self):
        """测试初始化"""
        attractor = LorenzAttractor(sigma=10.0, rho=28.0, beta=8.0/3.0)
        assert attractor.sigma == 10.0
        assert attractor.rho == 28.0
        assert attractor.beta == 8.0/3.0
        assert attractor.dim == 3
        logger.info("LorenzAttractor initialization test passed")

    def test_derivatives(self):
        """测试导数计算"""
        attractor = LorenzAttractor()
        z = torch.tensor([1.0, 1.0, 1.0])
        dz = attractor.derivatives(z)

        # 检查维度
        assert dz.shape == (3,)

        # 检查 Lorenz 方程的正确性
        # dx = sigma*(y-x) = 10*(1-1) = 0
        # dy = x*(rho-z) - y = 1*(28-1) - 1 = 26
        # dz = x*y - beta*z = 1*1 - 8/3*1 = 1 - 2.667 = -1.667
        expected_dx = 10.0 * (1.0 - 1.0)
        expected_dy = 1.0 * (28.0 - 1.0) - 1.0
        expected_dz = 1.0 * 1.0 - (8.0/3.0) * 1.0

        assert abs(dz[0].item() - expected_dx) < 1e-5
        assert abs(dz[1].item() - expected_dy) < 1e-5
        assert abs(dz[2].item() - expected_dz) < 1e-5

        logger.info("LorenzAttractor derivatives test passed")

    def test_integration(self):
        """测试积分演化"""
        attractor = LorenzAttractor(dt=0.01)
        z0 = torch.tensor([1.0, 1.0, 1.0])

        # 执行短时间积分
        result = attractor.integrate(z0, t_span=(0.0, 1.0), n_steps=100)

        # 检查状态有效
        assert not torch.isnan(result.z).any()
        assert not torch.isinf(result.z).any()

        # 检查轨迹进入混沌区域（典型范围）
        z_norm = torch.norm(result.z).item()
        assert 0 < z_norm < 100  # Lorenz 系统的典型范围

        logger.info(f"LorenzAttractor integration test passed: z_norm={z_norm:.4f}")

    def test_chaos_detection(self):
        """测试混沌检测"""
        attractor = LorenzAttractor(rho=28.0)
        assert attractor.is_chaotic() == True

        attractor_stable = LorenzAttractor(rho=20.0)
        assert attractor_stable.is_chaotic() == False

        logger.info("LorenzAttractor chaos detection test passed")

    def test_equilibrium_points(self):
        """测试平衡点计算"""
        attractor = LorenzAttractor(rho=28.0)
        equilibria = attractor.get_equilibrium_points()

        # ρ > 1 时应有三个平衡点
        assert len(equilibria) == 3

        # 原点
        assert equilibria[0] == (0.0, 0.0, 0.0)

        logger.info("LorenzAttractor equilibrium points test passed")


class TestRosslerAttractor:
    """罗斯勒吸引子测试"""

    def test_initialization(self):
        """测试初始化"""
        attractor = RosslerAttractor(a=0.2, b=0.2, c=5.7)
        assert attractor.a == 0.2
        assert attractor.b == 0.2
        assert attractor.c == 5.7
        assert attractor.dim == 3
        logger.info("RosslerAttractor initialization test passed")

    def test_derivatives(self):
        """测试导数计算"""
        attractor = RosslerAttractor()
        z = torch.tensor([1.0, 1.0, 1.0])
        dz = attractor.derivatives(z)

        # 检查维度
        assert dz.shape == (3,)

        # Rossler 方程验证
        # dx = -y - z = -1 - 1 = -2
        # dy = x + a*y = 1 + 0.2*1 = 1.2
        # dz = b + z*(x - c) = 0.2 + 1*(1 - 5.7) = 0.2 - 4.7 = -4.5
        assert abs(dz[0].item() - (-2.0)) < 1e-5
        assert abs(dz[1].item() - 1.2) < 1e-5
        assert abs(dz[2].item() - (-4.5)) < 1e-5

        logger.info("RosslerAttractor derivatives test passed")

    def test_integration(self):
        """测试积分演化"""
        attractor = RosslerAttractor(dt=0.01)
        z0 = torch.tensor([1.0, 1.0, 0.1])

        result = attractor.integrate(z0, t_span=(0.0, 1.0), n_steps=100)

        assert not torch.isnan(result.z).any()
        assert not torch.isinf(result.z).any()

        z_norm = torch.norm(result.z).item()
        assert 0 < z_norm < 50

        logger.info(f"RosslerAttractor integration test passed: z_norm={z_norm:.4f}")

    def test_chaos_detection(self):
        """测试混沌检测"""
        attractor = RosslerAttractor(c=5.7)
        assert attractor.is_chaotic() == True

        attractor_periodic = RosslerAttractor(c=3.0)
        assert attractor_periodic.is_chaotic() == False

        logger.info("RosslerAttractor chaos detection test passed")


class TestChuaAttractor:
    """蔡氏电路吸引子测试"""

    def test_initialization(self):
        """测试初始化"""
        attractor = ChuaAttractor(alpha=15.35, beta=28.0, m0=-1.143, m1=-0.714)
        assert attractor.alpha == 15.35
        assert attractor.beta == 28.0
        assert attractor.m0 == -1.143
        assert attractor.m1 == -0.714
        logger.info("ChuaAttractor initialization test passed")

    def test_chua_function(self):
        """测试 Chua 非线性函数"""
        attractor = ChuaAttractor()

        # 测试不同区域的值
        x1 = torch.tensor(0.5)   # |x| < 1
        x2 = torch.tensor(1.5)   # |x| > 1
        x3 = torch.tensor(-1.5)  # |x| > 1

        f1 = attractor._chua_function(x1)
        f2 = attractor._chua_function(x2)
        f3 = attractor._chua_function(x3)

        # 检查分段线性特性
        # 对于 |x| < 1, f(x) = m1*x + 0.5*(m0-m1)*(|x+1|-|x-1|)
        # 当 x=0.5: f = m1*0.5 + 0.5*(m0-m1)*1.0
        #          = -0.714*0.5 + 0.5*(-1.143+0.714)*1.0
        #          = -0.357 - 0.2145 = -0.5715
        expected_f1 = -0.714 * 0.5 + 0.5 * (-1.143 + 0.714) * 1.0
        assert abs(f1.item() - expected_f1) < 1e-5

        logger.info("ChuaAttractor function test passed")

    def test_derivatives(self):
        """测试导数计算"""
        attractor = ChuaAttractor()
        z = torch.tensor([1.0, 1.0, 1.0])
        dz = attractor.derivatives(z)

        assert dz.shape == (3,)
        assert not torch.isnan(dz).any()
        assert not torch.isinf(dz).any()

        logger.info("ChuaAttractor derivatives test passed")

    def test_integration(self):
        """测试积分演化"""
        attractor = ChuaAttractor(dt=0.01)
        z0 = torch.tensor([2.0, 0.0, 0.5])

        result = attractor.integrate(z0, t_span=(0.0, 1.0), n_steps=100)

        assert not torch.isnan(result.z).any()
        assert not torch.isinf(result.z).any()

        z_norm = torch.norm(result.z).item()
        assert 0 < z_norm < 30

        logger.info(f"ChuaAttractor integration test passed: z_norm={z_norm:.4f}")

    def test_chaos_detection(self):
        """测试混沌检测"""
        attractor = ChuaAttractor(alpha=15.35, beta=28.0)
        assert attractor.is_chaotic() == True

        logger.info("ChuaAttractor chaos detection test passed")


class TestChaosInjector:
    """混沌注入器测试"""

    def test_orthogonal_matrix_generation(self):
        """测试正交投影矩阵生成"""
        injector = ChaosInjector(core_subspace_dim=64, chaos_dim=3, seed=42)

        # 检查矩阵形状
        assert injector.W.shape == (64, 3)

        # 检查正交性：W^T W ≈ I
        WtW = torch.mm(injector.W.t(), injector.W)
        identity = torch.eye(3)
        error = torch.norm(WtW - identity).item()

        assert error < 1e-5, f"Orthogonal error too large: {error}"

        logger.info(f"ChaosInjector orthogonal matrix test passed: error={error:.6f}")

    def test_injection(self):
        """测试注入功能"""
        injector = ChaosInjector(core_subspace_dim=64, base_gain=0.1)

        # 混沌状态
        z = torch.randn(3)

        # 执行注入
        B = injector.inject(z)

        # 检查输出形状
        assert B.shape == (64,)

        # 检查数值有效
        assert not torch.isnan(B).any()
        assert not torch.isinf(B).any()

        logger.info(f"ChaosInjector injection test passed: B_norm={torch.norm(B).item():.4f}")

    def test_adaptive_gain(self):
        """测试自适应增益"""
        injector = ChaosInjector(core_subspace_dim=64, base_gain=0.1, target_variance=1.0)

        # 模拟高方差情况
        E_fast_core_high_var = torch.randn(64) * 10.0
        injector.inject(torch.randn(3), E_fast_core_high_var)

        # 高方差时增益应该降低
        assert injector.current_gain < injector.base_gain

        logger.info(f"ChaosInjector adaptive gain test passed: gain={injector.current_gain:.4f}")

    def test_full_dimension_injection(self):
        """测试全维度注入"""
        injector = ChaosInjector(core_subspace_dim=64)
        z = torch.randn(3)

        B_full = injector.get_full_dimension_injection(z, full_dim=2048)

        assert B_full.shape == (2048,)

        # 只有前64维有值
        assert torch.norm(B_full[:64]).item() > 0
        assert torch.norm(B_full[64:]).item() == 0

        logger.info("ChaosInjector full dimension injection test passed")


class TestAttractorManager:
    """吸引子管理器测试"""

    def test_registration(self):
        """测试吸引子注册"""
        manager = AttractorManager()
        manager.register_default_attractors()

        assert len(manager.attractors) == 3
        assert "Lorenz" in manager.attractor_names
        assert "Rossler" in manager.attractor_names
        assert "Chua" in manager.attractor_names

        logger.info("AttractorManager registration test passed")

    def test_initialization(self):
        """测试状态初始化"""
        manager = AttractorManager(seed=42)
        manager.register_default_attractors()
        manager.initialize_states(seed=42)

        assert manager.current_state is not None
        assert not torch.isnan(manager.current_state.z).any()

        logger.info("AttractorManager initialization test passed")

    def test_step_evolution(self):
        """测试单步演化"""
        manager = AttractorManager(switch_interval=100, seed=42)
        manager.register_default_attractors()
        manager.initialize_states(seed=42)

        # 执行多步演化
        for i in range(50):
            z = manager.step()
            assert not torch.isnan(z).any()
            assert z.shape == (3,)

        logger.info(f"AttractorManager step evolution test passed: steps=50")

    def test_switching(self):
        """测试吸引子切换"""
        manager = AttractorManager(switch_interval=10, transition_steps=5, seed=42)
        manager.register_default_attractors()
        manager.initialize_states(seed=42)

        initial_attractor = manager.attractor_names[manager.current_index]

        # 运行到触发切换
        for i in range(50):
            manager.step()

        # 检查是否发生了切换
        assert len(manager.switch_history) > 0

        logger.info(f"AttractorManager switching test passed: switches={len(manager.switch_history)}")

    def test_smooth_transition(self):
        """测试平滑过渡"""
        manager = AttractorManager(switch_interval=20, transition_steps=10, seed=42)
        manager.register_default_attractors()
        manager.initialize_states(seed=42)

        # 运行并检查过渡过程
        transition_detected = False
        for i in range(100):
            manager.step()
            if manager.is_transitioning:
                transition_detected = True
                # 检查过渡进度有效
                assert 0 <= manager.transition_progress <= 1

        assert transition_detected

        logger.info("AttractorManager smooth transition test passed")


class TestDefaultModeNetwork:
    """默认模式网络系统测试"""

    def test_initialization(self):
        """测试 DMN 初始化"""
        dmn = DefaultModeNetwork(config=DMNConfig(), seed=42)
        dmn.initialize()

        assert dmn._initialized
        assert dmn.attractor_manager is not None
        assert dmn.chaos_injector is not None
        assert dmn.state is not None

        logger.info("DefaultModeNetwork initialization test passed")

    def test_step_operation(self):
        """测试单步操作"""
        dmn = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16
        ), seed=42)
        dmn.initialize()

        # 执行多步
        for i in range(100):
            B = dmn.step()
            assert B.shape == (256,)
            assert not torch.isnan(B).any()

        logger.info(f"DefaultModeNetwork step test passed: steps=100, B_norm={torch.norm(B).item():.4f}")

    def test_stability_maintenance(self):
        """测试稳定性维持"""
        dmn = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16,
            stability_threshold=1000.0,
            base_gain=0.01
        ), seed=42)
        dmn.initialize()

        # 运行较长时间
        for i in range(1000):
            dmn.step(dt=0.001)  # 小步长避免发散

        assert dmn.state.is_stable

        stats = dmn.get_statistics()
        logger.info(f"DefaultModeNetwork stability test passed: E_norm={stats['state']['E_fast_norm']:.4f}")

    def test_continuous_run_short(self):
        """测试短时间连续运行"""
        dmn = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16,
            base_gain=0.01,
            stability_threshold=100.0
        ), seed=42)
        dmn.initialize()

        # 运行一小段时间（秒级模拟）
        stats = dmn.run_continuous(duration_hours=0.001, dt=0.001)  # 3.6秒模拟

        assert stats["stability_maintained"]
        assert stats["actual_duration_hours"] > 0

        logger.info(f"DefaultModeNetwork short continuous run test passed: "
                   f"hours={stats['actual_duration_hours']:.4f}")

    def test_statistics_collection(self):
        """测试统计信息收集"""
        dmn = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16
        ), seed=42)
        dmn.initialize()

        # 运行一些步骤
        for i in range(50):
            dmn.step()

        stats = dmn.get_statistics()

        # 检查统计信息完整性
        assert "system" in stats
        assert "attractor" in stats
        assert "injector" in stats
        assert "state" in stats

        assert stats["system"]["step_count"] == 50

        logger.info("DefaultModeNetwork statistics collection test passed")

    def test_state_save_load(self):
        """测试状态保存和加载"""
        dmn = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16
        ), seed=42)
        dmn.initialize()

        # 运行一些步骤
        for i in range(100):
            dmn.step()

        # 保存状态
        filepath = "test_dmn_state.json"
        dmn.save_state(filepath)

        # 创建新实例并加载
        dmn2 = DefaultModeNetwork(config=DMNConfig(
            fast_variable_dim=256,
            core_subspace_dim=16
        ), seed=42)
        dmn2.initialize()
        dmn2.load_state(filepath)

        # 检查状态恢复
        assert dmn2.state.step_count == dmn.state.step_count

        # 清理测试文件
        if os.path.exists(filepath):
            os.remove(filepath)

        logger.info("DefaultModeNetwork state save/load test passed")


def run_all_tests():
    """运行所有测试"""
    logger.info("=" * 60)
    logger.info("开始运行混沌吸引子和 DMN 系统测试")
    logger.info("=" * 60)

    # 运行测试
    test_lorenz = TestLorenzAttractor()
    test_lorenz.test_initialization()
    test_lorenz.test_derivatives()
    test_lorenz.test_integration()
    test_lorenz.test_chaos_detection()
    test_lorenz.test_equilibrium_points()

    test_rossler = TestRosslerAttractor()
    test_rossler.test_initialization()
    test_rossler.test_derivatives()
    test_rossler.test_integration()
    test_rossler.test_chaos_detection()

    test_chua = TestChuaAttractor()
    test_chua.test_initialization()
    test_chua.test_chua_function()
    test_chua.test_derivatives()
    test_chua.test_integration()
    test_chua.test_chaos_detection()

    test_injector = TestChaosInjector()
    test_injector.test_orthogonal_matrix_generation()
    test_injector.test_injection()
    test_injector.test_adaptive_gain()
    test_injector.test_full_dimension_injection()

    test_manager = TestAttractorManager()
    test_manager.test_registration()
    test_manager.test_initialization()
    test_manager.test_step_evolution()
    test_manager.test_switching()
    test_manager.test_smooth_transition()

    test_dmn = TestDefaultModeNetwork()
    test_dmn.test_initialization()
    test_dmn.test_step_operation()
    test_dmn.test_stability_maintenance()
    test_dmn.test_continuous_run_short()
    test_dmn.test_statistics_collection()
    test_dmn.test_state_save_load()

    logger.info("=" * 60)
    logger.info("所有测试通过！")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_tests()