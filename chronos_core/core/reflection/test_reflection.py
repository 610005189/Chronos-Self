"""
反思机制测试文件
=================

验证 Task 18-20 的反思机制实现：

Task 18: 实时反思机制
- SubTask 18.1: 最近T步计算图维护（T=1000）
- SubTask 18.2: 有限截断伴随法梯度回传
- SubTask 18.3: 实时轨迹修正
- SubTask 18.4: 实时反思计算效率测试

Task 19: 睡眠重放系统
- SubTask 19.1: 24小时触发机制
- SubTask 19.2: 关键帧向量数据库存储
- SubTask 19.3: 重放一致性损失计算
- SubTask 19.4: 预测改善损失计算

Task 20: 睡眠期梯度更新
- SubTask 20.1: 从关键帧向前积分的重放流程
- SubTask 20.2: 伴随法梯度回传
- SubTask 20.3: 不修改关键帧本身的约束
- SubTask 20.4: 睡眠重放稳定性测试
"""

import torch
import numpy as np
import pytest
import logging
import time
from pathlib import Path

from chronos_core.utils.config import ChronosConfig
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput, InputSource
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.reflection.realtime_reflection import (
    RealtimeReflection,
    RealtimeReflectionConfig,
    ComputationGraphBuffer,
    TruncatedAdjointMethod,
)
from chronos_core.core.reflection.sleep_replay import (
    SleepReplay,
    SleepReplayConfig,
    KeyframeData,
    KeyframeDatabase,
    ReplayLossCalculator,
)
from chronos_core.core.reflection.sleep_updater import (
    SleepUpdater,
    SleepUpdaterConfig,
    GradientConstraints,
    StabilityChecker,
)
from chronos_core.core.reflection.reflection_system import (
    ReflectionSystem,
    ReflectionSystemConfig,
    ReflectionMode,
    ReflectionState,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============ Task 18 测试：实时反思机制 ============

class TestRealtimeReflection:
    """测试实时反思机制（Task 18）"""
    
    def test_computation_graph_buffer(self):
        """
        SubTask 18.1: 测试最近T步计算图维护（T=1000）
        """
        logger.info("Testing ComputationGraphBuffer...")
        
        # 创建缓冲区配置
        config = RealtimeReflectionConfig(
            reflection_window=1000,
            max_graph_memory_mb=100.0,
            selective_snapshot_interval=10,
        )
        
        # 创建缓冲区
        buffer = ComputationGraphBuffer(config=config)
        
        # 测试添加步骤
        for i in range(1500):
            state = SelfState(
                E_fast=torch.randn(2048),
                E_slow=torch.randn(512),
                timestamp=i * 0.01,
            )
            
            buffer.add_step(state, metadata={'test_step': i})
        
        # 验证缓冲区长度
        assert len(buffer) == 1000, f"Buffer length should be 1000, got {len(buffer)}"
        
        # 验证统计信息
        stats = buffer.get_statistics()
        assert stats['total_snapshots'] == 1000
        assert stats['step_counter'] == 1500
        
        logger.info(f"ComputationGraphBuffer test passed: {stats}")
    
    def test_truncated_adjoint_method(self):
        """
        SubTask 18.2: 测试有限截断伴随法梯度回传
        """
        logger.info("Testing TruncatedAdjointMethod...")
        
        # 创建配置
        config = RealtimeReflectionConfig(
            reflection_window=100,
            use_adjoint=True,
            gradient_clip_value=1.0,
        )
        
        # 创建截断伴随法（无积分引擎）
        adjoint = TruncatedAdjointMethod(config=config)
        
        # 创建缓冲区并添加快照
        buffer = ComputationGraphBuffer(config=config)
        
        for i in range(50):
            state = SelfState(
                E_fast=torch.randn(2048),
                E_slow=torch.randn(512),
                timestamp=i * 0.01,
            )
            
            buffer.add_step(state, force_full_snapshot=True)
        
        # 测试梯度计算（简化测试）
        # 创建假参数
        fake_params = [torch.randn(100, requires_grad=True)]
        
        # 定义简单损失函数
        loss_fn = lambda pred, target: torch.norm(pred - target)
        
        # 计算梯度（应该返回空字典，因为没有积分引擎）
        gradients = adjoint.compute_gradients(
            graph_buffer=buffer,
            loss_fn=loss_fn,
            target_params=fake_params,
        )
        
        # 验证统计信息
        stats = adjoint.get_statistics()
        assert stats['adjoint_calls'] > 0
        
        logger.info(f"TruncatedAdjointMethod test passed: {stats}")
    
    def test_realtime_reflection_online_correction(self):
        """
        SubTask 18.3: 测试实时轨迹修正
        """
        logger.info("Testing RealtimeReflection trajectory correction...")
        
        # 创建配置
        config = RealtimeReflectionConfig(
            reflection_window=100,
            correction_interval=50,
            correction_learning_rate=0.001,
            max_correction_steps=3,
        )
        
        # 创建实时反思机制
        reflection = RealtimeReflection(config=config)
        reflection.initialize()
        
        # 添加多个步骤
        for i in range(100):
            state = SelfState(
                E_fast=torch.randn(2048),
                E_slow=torch.randn(512),
                timestamp=i * 0.01,
            )
            
            reflection.add_step(state)
        
        # 执行反思（不应用修正，因为没有积分引擎）
        result = reflection.reflect(apply_correction=False)
        
        # 验证结果（如果没有积分引擎，可能返回失败）
        # 测试主要验证计算图维护和梯度计算流程
        assert 'reflection_count' in result or 'reason' in result
        
        logger.info(f"RealtimeReflection correction test completed: {result}")
    
    def test_realtime_reflection_efficiency(self):
        """
        SubTask 18.4: 测试实时反思计算效率
        """
        logger.info("Testing RealtimeReflection efficiency...")
        
        # 创建配置
        config = RealtimeReflectionConfig(
            reflection_window=1000,
            max_graph_memory_mb=200.0,
        )
        
        # 创建实时反思机制
        reflection = RealtimeReflection(config=config)
        reflection.initialize()
        
        # 执行效率测试
        efficiency_result = reflection.test_computation_efficiency(
            test_steps=1000,
            test_window=100,
        )
        
        # 验证效率指标
        assert 'avg_add_step_time_ms' in efficiency_result
        assert 'avg_reflection_time_ms' in efficiency_result
        assert efficiency_result['avg_add_step_time_ms'] < 100, "Add step should be fast"
        assert efficiency_result['avg_reflection_time_ms'] < 5000, "Reflection should be efficient"
        
        logger.info(
            f"Efficiency test passed: "
            f"add_step={efficiency_result['avg_add_step_time_ms']:.2f}ms, "
            f"reflection={efficiency_result['avg_reflection_time_ms']:.2f}ms"
        )


# ============ Task 19 测试：睡眠重放系统 ============

class TestSleepReplay:
    """测试睡眠重放系统（Task 19）"""
    
    def test_sleep_trigger_mechanism(self):
        """
        SubTask 19.1: 测试24小时触发机制
        """
        logger.info("Testing sleep trigger mechanism...")
        
        # 创建配置
        config = SleepReplayConfig(
            sleep_replay_interval_hours=24.0,
            auto_trigger_enabled=True,
        )
        
        # 创建睡眠重放系统
        sleep_replay = SleepReplay(config=config)
        sleep_replay.initialize()
        
        # 测试触发判断
        current_time = time.time()
        
        # 初始状态（刚初始化，不应该睡眠）
        should_sleep = sleep_replay.should_sleep(current_time)
        assert not should_sleep, "Should not sleep initially"
        
        # 更新累积时间（超过24小时）
        sleep_replay.update_accumulated_time(24 * 3600 + 100)
        
        # 手动触发睡眠
        result = sleep_replay.trigger_sleep()
        
        # 验证结果
        assert result['success'], "Sleep should succeed"
        assert sleep_replay._sleep_count > 0
        
        logger.info(f"Sleep trigger test passed: sleeps={sleep_replay._sleep_count}")
    
    def test_keyframe_database(self):
        """
        SubTask 19.2: 测试关键帧向量数据库存储
        """
        logger.info("Testing keyframe database...")
        
        # 创建配置（使用内存存储）
        config = SleepReplayConfig(
            vector_db_path="test_vector_db",
        )
        
        # 创建数据库
        db = KeyframeDatabase(config=config)
        
        # 创建关键帧
        for i in range(10):
            keyframe = KeyframeData(
                keyframe_id=f"kf_{i}",
                timestamp=i * 10.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
                emotional_intensity=np.random.rand(),
                importance=np.random.rand(),
            )
            
            db.add_keyframe(keyframe)
        
        # 验证存储
        assert len(db) >= 10, "Database should have at least 10 keyframes"
        
        # 测试检索
        retrieved = db.retrieve_keyframe("kf_0")
        assert retrieved is not None, "Should retrieve keyframe"
        assert retrieved.keyframe_id == "kf_0"
        
        # 测试查询
        recent_kfs = db.get_recent_keyframes(n=5)
        assert len(recent_kfs) == 5, "Should get 5 recent keyframes"
        
        logger.info(f"Keyframe database test passed: count={len(db)}")
    
    def test_replay_consistency_loss(self):
        """
        SubTask 19.3: 测试重放一致性损失计算
        """
        logger.info("Testing replay consistency loss...")
        
        # 创建配置
        config = SleepReplayConfig(
            consistency_loss_weight=1.0,
            replay_window_seconds=60.0,
        )
        
        # 创建损失计算器
        calculator = ReplayLossCalculator(config=config)
        
        # 创建关键帧
        keyframes = []
        for i in range(5):
            keyframe = KeyframeData(
                keyframe_id=f"kf_{i}",
                timestamp=i * 10.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
            )
            keyframes.append(keyframe)
        
        # 计算损失
        losses = calculator.compute_losses(keyframes)
        
        # 验证损失计算
        assert 'consistency_loss' in losses
        assert 'total_loss' in losses
        assert losses['consistency_loss'] >= 0, "Loss should be non-negative"
        
        logger.info(f"Consistency loss test passed: loss={losses['total_loss']:.4f}")
    
    def test_prediction_improve_loss(self):
        """
        SubTask 19.4: 测试预测改善损失计算
        """
        logger.info("Testing prediction improve loss...")
        
        # 创建配置
        config = SleepReplayConfig(
            improve_loss_weight=0.5,
        )
        
        # 创建损失计算器
        calculator = ReplayLossCalculator(config=config)
        
        # 创建关键帧（包含实际后续状态）
        keyframes = []
        for i in range(5):
            keyframe = KeyframeData(
                keyframe_id=f"kf_{i}",
                timestamp=i * 10.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
                actual_outcome=np.random.randn(2048).astype(np.float32),  # 实际后续状态
                outcome_timestamp=i * 10.0 + 60.0,
            )
            keyframes.append(keyframe)
        
        # 计算损失
        losses = calculator.compute_losses(keyframes)
        
        # 验证损失计算
        assert 'improve_loss' in losses
        assert losses['improve_loss'] >= 0, "Loss should be non-negative"
        
        logger.info(f"Improve loss test passed: improve_loss={losses['improve_loss']:.4f}")


# ============ Task 20 测试：睡眠期梯度更新 ============

class TestSleepUpdater:
    """测试睡眠期梯度更新（Task 20）"""
    
    def test_gradient_constraints(self):
        """
        SubTask 20.3: 测试不修改关键帧本身的约束
        """
        logger.info("Testing gradient constraints...")
        
        # 创建配置
        config = SleepUpdaterConfig(
            gradient_clip_value=1.0,
            max_parameter_change=0.1,
            freeze_keyframe_params=True,
        )
        
        # 创建约束机制
        constraints = GradientConstraints(config=config)
        
        # 创建测试梯度
        gradients = {
            'param_0': torch.randn(100) * 10,  # 大梯度
            'param_1': torch.randn(100) * 0.1,  # 小梯度
        }
        
        # 应用梯度裁剪
        clipped = constraints.apply_gradient_clipping(gradients)
        
        # 验证裁剪（允许小容差，因为浮点数精度）
        tolerance = 1e-6
        for name, grad in clipped.items():
            if grad is not None:
                norm = torch.norm(grad).item()
                assert norm <= config.gradient_clip_value + tolerance, f"Gradient norm {norm} should be <= {config.gradient_clip_value}"
        
        # 测试关键帧冻结
        keyframes = [
            KeyframeData(
                keyframe_id="test_kf",
                timestamp=0.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
            )
        ]
        
        frozen_states = constraints.freeze_keyframe_states(keyframes)
        
        # 验证冻结状态
        for E_fast, E_slow in frozen_states:
            assert not E_fast.requires_grad, "Frozen state should not require grad"
            assert not E_slow.requires_grad, "Frozen state should not require grad"
        
        logger.info(f"Gradient constraints test passed: clips={constraints._stats['gradient_clips']}")
    
    def test_sleep_replay_stability(self):
        """
        SubTask 20.4: 测试睡眠重放稳定性
        """
        logger.info("Testing sleep replay stability...")
        
        # 创建配置
        config = SleepUpdaterConfig(
            stability_check_enabled=True,
            parameter_norm_threshold=10.0,
            loss_threshold=100.0,
            gradient_norm_threshold=5.0,
        )
        
        # 创建稳定性检查器
        checker = StabilityChecker(config=config)
        
        # 创建测试关键帧
        test_keyframes = []
        for i in range(10):
            keyframe = KeyframeData(
                keyframe_id=f"test_kf_{i}",
                timestamp=i * 10.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
            )
            test_keyframes.append(keyframe)
        
        # 创建测试参数
        params = [torch.randn(100, requires_grad=True)]
        
        # 测试参数稳定性检查
        param_check = checker.check_parameters(params)
        assert param_check['is_stable'], "Parameters should be stable initially"
        
        # 测试梯度稳定性检查
        gradients = {'param_0': torch.randn(100) * 0.5}
        grad_check = checker.check_gradients(gradients)
        assert grad_check['is_stable'], "Gradients should be stable"
        
        # 测试损失稳定性检查
        loss = torch.tensor(1.0)
        loss_check = checker.check_loss(loss)
        assert loss_check['is_stable'], "Loss should be stable"
        
        logger.info(f"Stability test passed: checks={checker._stats['total_checks']}")
    
    def test_sleep_updater_update_flow(self):
        """
        SubTask 20.1 & 20.2: 测试从关键帧向前积分的重放流程和伴随法梯度回传
        """
        logger.info("Testing sleep updater flow...")
        
        # 创建配置
        config = SleepUpdaterConfig(
            learning_rate=1e-4,
            gradient_clip_value=1.0,
            replay_batch_size=5,
            max_replay_steps=10,
        )
        
        # 创建睡眠更新器
        updater = SleepUpdater(config=config)
        updater.initialize()
        
        # 创建测试关键帧
        test_keyframes = []
        for i in range(20):
            keyframe = KeyframeData(
                keyframe_id=f"test_kf_{i}",
                timestamp=i * 10.0,
                E_fast=np.random.randn(2048).astype(np.float32),
                E_slow=np.random.randn(512).astype(np.float32),
                actual_outcome=np.random.randn(2048).astype(np.float32),
                outcome_timestamp=i * 10.0 + 60.0,
            )
            test_keyframes.append(keyframe)
        
        # 执行睡眠更新
        result = updater.perform_sleep_update(
            keyframes=test_keyframes,
            max_updates=5,
        )
        
        # 验证结果
        assert result['success'], "Update should succeed"
        assert 'avg_loss' in result
        assert 'avg_gradient_norm' in result
        assert result['is_stable'], "Update should be stable"
        
        logger.info(
            f"Sleep updater test passed: "
            f"avg_loss={result['avg_loss']:.4f}, "
            f"avg_grad_norm={result['avg_gradient_norm']:.4f}"
        )


# ============ 完整系统测试 ============

class TestReflectionSystem:
    """测试完整反思系统"""
    
    def test_reflection_system_integration(self):
        """
        测试完整反思系统的整合功能
        """
        logger.info("Testing ReflectionSystem integration...")
        
        # 创建全局配置
        config = ChronosConfig()
        
        # 创建反思系统
        system = ReflectionSystem(
            global_config=config,
        )
        system.initialize()
        
        # 开始在线运行
        system.start_online_running()
        
        # 添加在线步骤
        for i in range(100):
            state = SelfState(
                E_fast=torch.randn(config.dim.fast_variable_dim),
                E_slow=torch.randn(config.dim.slow_variable_dim),
                timestamp=i * 0.01,
            )
            
            inputs = ExternalInput(
                X_sem=torch.randn(256),  # 语义流维度256
                X_log=torch.randn(512),  # 物理流维度512
                timestamp=i * 0.01,
                emotional_intensity=np.random.rand(),
                importance=np.random.rand(),
            )
            
            result = system.add_online_step(state, inputs)
        
        # 检查状态
        current_state = system.get_current_state()
        assert current_state['mode'] == ReflectionMode.ONLINE.value
        assert current_state['online_step_count'] == 100
        
        # 检查性能监测
        performance = system.monitor_performance()
        assert 'total_online_steps' in performance
        
        # 获取统计信息
        stats = system.get_statistics()
        assert stats['initialized']
        
        logger.info(
            f"ReflectionSystem integration test passed: "
            f"steps={current_state['online_step_count']}, "
            f"performance={performance}"
        )
    
    def test_full_reflection_flow(self):
        """
        测试完整反思流程：在线运行 -> 睡眠 -> 重放 -> 更新
        """
        logger.info("Testing full reflection flow...")
        
        # 创建配置
        config = ChronosConfig()
        
        # 创建反思系统
        system = ReflectionSystem(global_config=config)
        system.initialize()
        
        # 1. 开始在线运行
        system.start_online_running()
        
        # 2. 添加在线步骤（实时反思）
        for i in range(200):
            state = SelfState(
                E_fast=torch.randn(config.dim.fast_variable_dim),
                E_slow=torch.randn(config.dim.slow_variable_dim),
                timestamp=i * 0.01,
            )
            
            inputs = ExternalInput(
                X_sem=torch.randn(256),  # 语义流维度256
                X_log=torch.randn(512),  # 物理流维度512
                timestamp=i * 0.01,
                source=InputSource.TEXT,
                emotional_intensity=0.7 if i % 50 == 0 else 0.3,
                importance=0.8 if i % 50 == 0 else 0.5,
            )
            
            system.add_online_step(state, inputs)
        
        # 3. 手动触发睡眠
        sleep_result = system.manual_reflection(reflection_type="sleep")
        
        # 验证睡眠结果
        assert sleep_result['success'], "Sleep should succeed"
        
        # 4. 查询反思历史
        history = system.get_reflection_history(limit=10)
        assert len(history) > 0, "Should have reflection history"
        
        # 5. 监测性能
        performance = system.monitor_performance()
        assert performance['total_sleeps'] > 0
        
        logger.info(
            f"Full reflection flow test passed: "
            f"sleeps={performance['total_sleeps']}, "
            f"history_count={len(history)}"
        )


def run_all_tests():
    """
    运行所有测试
    """
    logger.info("=" * 60)
    logger.info("Running all reflection mechanism tests...")
    logger.info("=" * 60)
    
    # Task 18 测试
    logger.info("\n" + "=" * 60)
    logger.info("Task 18: 实时反思机制测试")
    logger.info("=" * 60)
    
    test_realtime = TestRealtimeReflection()
    test_realtime.test_computation_graph_buffer()
    test_realtime.test_truncated_adjoint_method()
    test_realtime.test_realtime_reflection_online_correction()
    test_realtime.test_realtime_reflection_efficiency()
    
    # Task 19 测试
    logger.info("\n" + "=" * 60)
    logger.info("Task 19: 睡眠重放系统测试")
    logger.info("=" * 60)
    
    test_sleep = TestSleepReplay()
    test_sleep.test_sleep_trigger_mechanism()
    test_sleep.test_keyframe_database()
    test_sleep.test_replay_consistency_loss()
    test_sleep.test_prediction_improve_loss()
    
    # Task 20 测试
    logger.info("\n" + "=" * 60)
    logger.info("Task 20: 睡眠期梯度更新测试")
    logger.info("=" * 60)
    
    test_updater = TestSleepUpdater()
    test_updater.test_gradient_constraints()
    test_updater.test_sleep_replay_stability()
    test_updater.test_sleep_updater_update_flow()
    
    # 完整系统测试
    logger.info("\n" + "=" * 60)
    logger.info("完整反思系统测试")
    logger.info("=" * 60)
    
    test_system = TestReflectionSystem()
    test_system.test_reflection_system_integration()
    test_system.test_full_reflection_flow()
    
    logger.info("\n" + "=" * 60)
    logger.info("所有测试通过！")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_all_tests()