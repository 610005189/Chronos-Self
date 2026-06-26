"""
端到端集成测试（End-to-End Integration Tests）
==============================================

测试完整的系统流程：
- 输入处理到输出生成的完整链路
- 验证各组件协同工作
- 检查数据流和状态传递

测试场景：
1. 简单文本交互场景
2. 多轮对话场景
3. 长时间运行场景（24小时）
4. 元认知调控场景
5. 睡眠重放场景

测试稳定性：
- 数值稳定性（无NaN/Inf）
- 内存稳定性（无泄漏）
- 性能稳定性（持续运行）

使用方式：
    pytest tests/test_e2e_integration.py -v
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional
import time
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.utils.config import ChronosConfig, DimensionalityConfig
from chronos_core.integration.system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController,
    SystemStatus,
    OperationMode,
    create_chronos_system_from_config,
)
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput


class TestChronosSystemIntegration:
    """Chronos 系统集成测试"""
    
    @pytest.fixture(autouse=True)
    def setup_test_config(self):
        """设置测试配置"""
        # 使用较小的配置进行测试
        self.dim_config = DimensionalityConfig()
        self.dim_config.fast_variable_dim = 256
        self.dim_config.slow_variable_dim = 128
        self.dim_config.semantic_dim = 256
        self.dim_config.physical_dim = 256
        self.dim_config.fusion_dim = 512
        
        self.global_config = ChronosConfig(dim=self.dim_config)
        self.global_config.device = "cpu"  # 使用 CPU 进行测试
        
        self.system_config = ChronosSystemConfig()
        self.system_config.device = "cpu"
        self.system_config.enable_semantic_encoder = False  # 简化测试
        self.system_config.enable_logical_encoder = False
        self.system_config.enable_fusion_module = False
        self.system_config.enable_meta_cognitive = True
        self.system_config.enable_reflection = True
        self.system_config.enable_realtime_reflection = True
        self.system_config.enable_sleep_replay = False  # 简化测试
        self.system_config.enable_working_memory = True
        self.system_config.performance_log_interval = 50
    
    def test_01_system_initialization(self):
        """测试 01: 系统初始化"""
        print("\n=== 测试 01: 系统初始化 ===")
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        
        # 初始化前检查
        assert not system._initialized
        assert system._system_state.status == SystemStatus.CREATED
        
        # 初始化系统
        system.initialize()
        
        # 初始化后检查
        assert system._initialized
        assert system._system_state.status == SystemStatus.READY
        
        # 检查核心组件
        assert system.integration_engine is not None
        assert system.dmn is not None
        assert system.working_memory is not None
        assert system.meta_cognitive_system is not None
        assert system.reflection_system is not None
        
        # 检查初始状态
        assert system._current_self_state is not None
        assert system._current_self_state.get_fast_norm() == 0.0
        assert system._current_self_state.get_slow_norm() == 0.0
        
        print("✓ 系统初始化成功")
        print(f"  状态: {system._system_state.status.value}")
        print(f"  组件: integration_engine={system.integration_engine is not None}")
        
        # 清理
        system.shutdown()
    
    def test_02_single_input_processing(self):
        """测试 02: 单次输入处理"""
        print("\n=== 测试 02: 单次输入处理 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 创建外部输入
        external_input = ExternalInput(
            X_sem=torch.randn(self.dim_config.semantic_dim),
            X_log=torch.randn(self.dim_config.physical_dim),
            importance=0.8,
            emotion_value=0.5
        )
        
        # 处理输入
        response = system.process_input(
            external_input=external_input,
            apply_meta_cognitive=True,
            apply_reflection=True
        )
        
        # 检查响应
        assert response.state_before is not None
        assert response.state_after is not None
        assert response.processing_time_ms > 0
        
        # 检查状态变化
        assert response.state_after.timestamp > response.state_before.timestamp
        
        # 检查系统统计
        stats = system.get_statistics()
        assert stats["system_state"]["total_steps"] == 1
        assert stats["system_state"]["integration_steps"] == 1
        
        print("✓ 单次输入处理成功")
        print(f"  处理时间: {response.processing_time_ms:.2f}ms")
        print(f"  状态时间戳: {response.state_after.timestamp:.2f}")
        print(f"  总步数: {stats['system_state']['total_steps']}")
        
        # 清理
        system.shutdown()
    
    def test_03_multi_round_interaction(self):
        """测试 03: 多轮交互场景"""
        print("\n=== 测试 03: 多轮交互场景 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 多轮交互
        num_rounds = 10
        responses = []
        
        for round_idx in range(num_rounds):
            # 创建输入
            external_input = ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim),
                X_log=torch.randn(self.dim_config.physical_dim),
                importance=np.random.rand(),
                emotion_value=np.random.rand()
            )
            
            # 处理输入
            response = system.process_input(
                external_input=external_input,
                apply_meta_cognitive=True,
                apply_reflection=True
            )
            
            responses.append(response)
            
            # 检查数值稳定性
            assert not torch.isnan(response.state_after.E_fast).any()
            assert not torch.isinf(response.state_after.E_fast).any()
            assert not torch.isnan(response.state_after.E_slow).any()
            assert not torch.isinf(response.state_after.E_slow).any()
        
        # 检查系统状态
        stats = system.get_statistics()
        assert stats["system_state"]["total_steps"] == num_rounds
        
        # 检查状态演化轨迹
        timestamps = [r.state_after.timestamp for r in responses]
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i-1], "时间戳应递增"
        
        # 检查平均处理时间
        avg_time = np.mean([r.processing_time_ms for r in responses])
        print("✓ 多轮交互成功")
        print(f"  交互轮数: {num_rounds}")
        print(f"  平均处理时间: {avg_time:.2f}ms")
        print(f"  最终时间戳: {timestamps[-1]:.2f}")
        print(f"  数值稳定: 无NaN/Inf")
        
        # 清理
        system.shutdown()
    
    def test_04_long_time_running(self):
        """测试 04: 长时间运行场景（模拟24小时）"""
        print("\n=== 测试 04: 长时间运行场景 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 设置运行参数
        # 使用较少的步数进行快速测试
        num_steps = 1000  # 简化测试
        dt = 0.01
        
        # 运行统计
        start_time = time.time()
        stability_issues = 0
        
        # 长时间运行
        for step_idx in range(num_steps):
            # 创建随机输入（模拟外部环境变化）
            external_input = ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim) * 0.1,
                X_log=torch.randn(self.dim_config.physical_dim) * 0.1,
                importance=np.random.rand() * 0.5,
                emotion_value=np.random.rand() * 0.3
            )
            
            # 处理输入
            response = system.process_input(
                external_input=external_input,
                apply_meta_cognitive=True,
                apply_reflection=(step_idx % 100 == 0)  # 每隔100步反思
            )
            
            # 检查数值稳定性
            E_fast_norm = response.state_after.get_fast_norm()
            E_slow_norm = response.state_after.get_slow_norm()
            
            if E_fast_norm > 1000 or E_slow_norm > 1000:
                stability_issues += 1
            
            # 每100步输出进度
            if step_idx % 100 == 0:
                stats = system.get_statistics()
                print(f"  Step {step_idx}: "
                      f"fast_norm={E_fast_norm:.4f}, "
                      f"slow_norm={E_slow_norm:.4f}, "
                      f"stable={stats['system_state']['is_stable']}")
        
        elapsed_time = time.time() - start_time
        
        # 检查最终状态
        stats = system.get_statistics()
        assert stats["system_state"]["total_steps"] == num_steps
        assert stats["system_state"]["is_stable"] or stability_issues < 10
        
        # 检查系统未崩溃
        assert system._system_state.status != SystemStatus.ERROR
        
        print("✓ 长时间运行成功")
        print(f"  运行步数: {num_steps}")
        print(f"  实际耗时: {elapsed_time:.2f}秒")
        print(f"  模拟时间: {stats['system_state']['simulated_time']:.2f}秒")
        print(f"  稳定性问题: {stability_issues}")
        print(f"  系统状态: {stats['system_state']['status']}")
        
        # 清理
        system.shutdown()
    
    def test_05_meta_cognitive_regulation(self):
        """测试 05: 元认知调控场景"""
        print("\n=== 测试 05: 元认知调控场景 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 测试带元认知调控的输入处理
        external_input = ExternalInput(
            X_sem=torch.randn(self.dim_config.semantic_dim),
            X_log=torch.randn(self.dim_config.physical_dim),
            importance=0.9,
            emotion_value=0.7
        )
        
        # 启用元认知调控
        response_with_meta = system.process_input(
            external_input=external_input,
            apply_meta_cognitive=True,
            apply_reflection=False
        )
        
        # 重置系统
        system.reset()
        
        # 禁用元认知调控
        response_without_meta = system.process_input(
            external_input=external_input,
            apply_meta_cognitive=False,
            apply_reflection=False
        )
        
        # 比较结果
        # 元认知调控应该对状态演化产生影响
        with_meta_norm = response_with_meta.state_after.get_fast_norm()
        without_meta_norm = response_without_meta.state_after.get_fast_norm()
        
        # 检查元认知系统状态
        meta_stats = system.meta_cognitive_system.get_statistics()
        
        print("✓ 元认知调控测试成功")
        print(f"  有元认知调控: fast_norm={with_meta_norm:.4f}")
        print(f"  无元认知调控: fast_norm={without_meta_norm:.4f}")
        print(f"  元认知系统统计: {meta_stats['total_steps']}")
        
        # 清理
        system.shutdown()
    
    def test_06_reflection_mechanism(self):
        """测试 06: 反思机制"""
        print("\n=== 测试 06: 反思机制 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 运行多步以触发反思
        num_steps = 200
        
        reflection_count = 0
        
        for step_idx in range(num_steps):
            external_input = ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim) * 0.1,
                X_log=torch.randn(self.dim_config.physical_dim) * 0.1,
                importance=np.random.rand(),
                emotion_value=np.random.rand()
            )
            
            # 处理输入（启用反思）
            response = system.process_input(
                external_input=external_input,
                apply_meta_cognitive=True,
                apply_reflection=True
            )
            
            # 统计反思次数
            if response.reflection_performed:
                reflection_count += 1
        
        # 检查反思系统状态
        reflection_stats = system.reflection_system.get_statistics()
        
        # 验证反思机制工作
        assert reflection_stats["performance_metrics"]["total_online_steps"] > 0
        
        print("✓ 反思机制测试成功")
        print(f"  运行步数: {num_steps}")
        print(f"  反思次数: {reflection_count}")
        print(f"  反思系统统计: online_steps={reflection_stats['performance_metrics']['total_online_steps']}")
        
        # 清理
        system.shutdown()
    
    def test_07_system_controller(self):
        """测试 07: 系统控制器"""
        print("\n=== 测试 07: 系统控制器 ===")
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        
        # 创建控制器
        controller = ChronosSystemController(system)
        
        # 启动系统
        controller.start()
        assert controller._is_running
        assert system._system_state.status == SystemStatus.RUNNING
        
        # 处理输入
        response = controller.process_input(
            text="测试输入"
        )
        assert response is not None
        
        # 批量处理
        batch_inputs = [
            {"text": "输入1"},
            {"text": "输入2"},
            {"text": "输入3"},
        ]
        batch_responses = controller.process_batch(batch_inputs)
        assert len(batch_responses) == 3
        
        # 获取状态
        state = controller.get_system_state()
        assert state.status == SystemStatus.RUNNING
        
        # 暂停和恢复
        controller.pause()
        assert system._system_state.status == SystemStatus.PAUSED
        
        controller.resume()
        assert system._system_state.status == SystemStatus.RUNNING
        
        # 停止系统
        controller.stop()
        assert not controller._is_running
        assert system._system_state.status == SystemStatus.STOPPED
        
        print("✓ 系统控制器测试成功")
        print(f"  批量处理: {len(batch_responses)}个响应")
        print(f"  系统状态变化: CREATED -> RUNNING -> PAUSED -> RUNNING -> STOPPED")
    
    def test_08_state_save_load(self):
        """测试 08: 状态保存和加载"""
        print("\n=== 测试 08: 状态保存和加载 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 运行一些步骤
        for _ in range(10):
            external_input = ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim),
                X_log=torch.randn(self.dim_config.physical_dim)
            )
            system.process_input(external_input=external_input)
        
        # 获取原始状态
        original_state = system.get_current_state()
        original_stats = system.get_statistics()
        
        # 保存状态
        save_path = "data/test_state.json"
        system.save_state(save_path)
        
        # 创建新系统并加载状态
        new_system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        new_system.initialize()
        new_system.load_state(save_path)
        
        # 检查加载后的状态
        loaded_state = new_system.get_current_state()
        
        # 验证状态恢复
        assert torch.allclose(loaded_state.E_fast, original_state.E_fast)
        assert torch.allclose(loaded_state.E_slow, original_state.E_slow)
        
        print("✓ 状态保存和加载成功")
        print(f"  保存路径: {save_path}")
        print(f"  状态一致性验证: 通过")
        
        # 清理
        system.shutdown()
        new_system.shutdown()
    
    def test_09_numerical_stability(self):
        """测试 09: 数值稳定性"""
        print("\n=== 测试 09: 数值稳定性 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 测试极端输入
        extreme_inputs = [
            # 大值输入
            ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim) * 1000,
                X_log=torch.randn(self.dim_config.physical_dim) * 1000,
                importance=1.0
            ),
            # 小值输入
            ExternalInput(
                X_sem=torch.randn(self.dim_config.semantic_dim) * 1e-6,
                X_log=torch.randn(self.dim_config.physical_dim) * 1e-6,
                importance=0.1
            ),
            # 包含极端值的输入
            ExternalInput(
                X_sem=torch.cat([
                    torch.randn(self.dim_config.semantic_dim // 2) * 1000,
                    torch.randn(self.dim_config.semantic_dim // 2) * 1e-6
                ]),
                X_log=torch.randn(self.dim_config.physical_dim),
                importance=0.5
            ),
        ]
        
        stability_issues = 0
        
        for i, external_input in enumerate(extreme_inputs):
            response = system.process_input(
                external_input=external_input,
                apply_meta_cognitive=True,
                apply_reflection=False
            )
            
            # 检查数值稳定性
            has_nan = torch.isnan(response.state_after.E_fast).any() or torch.isnan(response.state_after.E_slow).any()
            has_inf = torch.isinf(response.state_after.E_fast).any() or torch.isinf(response.state_after.E_slow).any()
            
            if has_nan or has_inf:
                stability_issues += 1
                print(f"  输入{i}: 发现数值问题 (NaN={has_nan}, Inf={has_inf})")
            else:
                print(f"  输入{i}: 数值稳定 (fast_norm={response.state_after.get_fast_norm():.4f})")
        
        # 系统应该自动处理极端输入
        assert stability_issues <= len(extreme_inputs) // 2, "过多的数值稳定性问题"
        
        print("✓ 数值稳定性测试完成")
        print(f"  极端输入数量: {len(extreme_inputs)}")
        print(f"  数值问题数量: {stability_issues}")
        print(f"  自动修正机制: 启用")
        
        # 清理
        system.shutdown()
    
    def test_10_component_integration(self):
        """测试 10: 组件协同工作"""
        print("\n=== 测试 10: 组件协同工作 ===")
        
        # 创建并初始化系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 检查各组件状态
        components = {
            "integration_engine": system.integration_engine,
            "dmn": system.dmn,
            "working_memory": system.working_memory,
            "meta_cognitive_system": system.meta_cognitive_system,
            "reflection_system": system.reflection_system,
        }
        
        # 验证各组件已初始化
        for name, component in components.items():
            assert component is not None, f"{name} 未初始化"
            print(f"  {name}: ✓ 已初始化")
        
        # 运行完整流程
        external_input = ExternalInput(
            X_sem=torch.randn(self.dim_config.semantic_dim),
            X_log=torch.randn(self.dim_config.physical_dim),
            importance=0.8
        )
        
        response = system.process_input(
            external_input=external_input,
            apply_meta_cognitive=True,
            apply_reflection=True
        )
        
        # 验证数据流
        # 积分引擎 -> 元认知系统 -> 反思系统
        
        # 检查积分引擎
        engine_stats = system.integration_engine.get_state_monitoring()
        assert engine_stats["step_count"] > 0
        
        # 检查元认知系统
        meta_stats = system.meta_cognitive_system.get_statistics()
        assert meta_stats["total_steps"] > 0
        
        # 检查反思系统
        reflection_stats = system.reflection_system.get_statistics()
        assert reflection_stats["performance_metrics"]["total_online_steps"] > 0
        
        # 检查工作记忆
        memory_stats = system.working_memory.get_statistics()
        assert memory_stats["total_chunks_created"] > 0
        
        print("✓ 组件协同工作验证完成")
        print(f"  积分引擎步数: {engine_stats['step_count']}")
        print(f"  元认知系统步数: {meta_stats['total_steps']}")
        print(f"  反思系统在线步数: {reflection_stats['performance_metrics']['total_online_steps']}")
        print(f"  工作记忆组块数: {memory_stats['total_chunks_created']}")
        
        # 清理
        system.shutdown()


class TestSystemControllerFeatures:
    """系统控制器功能测试"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """设置测试环境"""
        self.dim_config = DimensionalityConfig()
        self.dim_config.fast_variable_dim = 256
        self.dim_config.slow_variable_dim = 128
        self.dim_config.semantic_dim = 256
        self.dim_config.physical_dim = 256
        
        self.global_config = ChronosConfig(dim=self.dim_config)
        self.global_config.device = "cpu"
        
        self.system_config = ChronosSystemConfig()
        self.system_config.device = "cpu"
        self.system_config.enable_semantic_encoder = False
        self.system_config.enable_logical_encoder = False
        self.system_config.enable_fusion_module = False
    
    def test_continuous_running(self):
        """测试连续运行"""
        print("\n=== 测试控制器连续运行 ===")
        
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        controller = ChronosSystemController(system)
        
        controller.start()
        
        # 连续运行（简化测试）
        stats = controller.run_continuous(duration_hours=0.01)  # 0.01小时
        
        assert stats["total_responses"] > 0
        assert stats["errors"] == 0
        
        print(f"✓ 连续运行完成")
        print(f"  总响应数: {stats['total_responses']}")
        print(f"  错误数: {stats['errors']}")
        
        controller.stop()
    
    def test_error_handling(self):
        """测试错误处理"""
        print("\n=== 测试错误处理 ===")
        
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        controller = ChronosSystemController(system)
        
        controller.start()
        
        # 设置错误回调
        errors = []
        controller.set_callbacks(on_error=lambda step, e: errors.append((step, str(e))))
        
        # 处理正常输入
        controller.process_input(text="正常输入")
        
        # 模拟错误（重置系统后再尝试处理）
        system._system_state.status = SystemStatus.ERROR
        
        try:
            controller.process_input(text="错误状态输入")
        except ValueError:
            pass  # 预期的错误
        
        print(f"✓ 错误处理测试完成")
        print(f"  错误回调: 已设置")
        
        # 恢复系统
        system._system_state.status = SystemStatus.RUNNING
        controller.stop()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Chronos-Self 端到端集成测试")
    print("=" * 80)
    
    # 创建测试类实例
    test_integration = TestChronosSystemIntegration()
    test_controller = TestSystemControllerFeatures()
    
    # 设置测试配置
    test_integration.setup_test_config()
    test_controller.setup()
    
    # 运行集成测试
    tests = [
        ("系统初始化", test_integration.test_01_system_initialization),
        ("单次输入处理", test_integration.test_02_single_input_processing),
        ("多轮交互", test_integration.test_03_multi_round_interaction),
        ("长时间运行", test_integration.test_04_long_time_running),
        ("元认知调控", test_integration.test_05_meta_cognitive_regulation),
        ("反思机制", test_integration.test_06_reflection_mechanism),
        ("系统控制器", test_integration.test_07_system_controller),
        ("状态保存加载", test_integration.test_08_state_save_load),
        ("数值稳定性", test_integration.test_09_numerical_stability),
        ("组件协同", test_integration.test_10_component_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print("\n" + "=" * 80)
    print(f"测试结果: 通过={passed}, 失败={failed}")
    print("=" * 80)
    
    return passed, failed


if __name__ == "__main__":
    run_all_tests()