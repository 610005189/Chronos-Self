"""
工作记忆系统测试 - Working Memory System Tests
==============================================

验证 Task 16-17 的完整实现：
- Task 16: 组块形成与激活强度管理
- Task 17: 容量约束与历史信息保留

测试内容：
1. 组块动态生成（注意力绑定）
2. 激活强度向量（ActivationStrength）
3. 激活衰减与更新机制
4. Top-N组块选择机制（N=7）
5. 低激活组块信息保留
6. 快速恢复机制
7. 容量限制验证（满足米勒定律）
"""

import pytest
import torch
import numpy as np
import time
from typing import Dict, List

from chronos_core.memory.work_memory import (
    Chunk,
    ChunkType,
    ChunkStatus,
    ActivationStrength,
    WorkingMemory,
)


class TestChunk:
    """测试组块（Chunk）数据类"""
    
    def test_chunk_creation(self):
        """测试组块创建"""
        content = torch.randn(256)
        attention_weights = torch.ones(256) / 256
        
        chunk = Chunk(
            chunk_id="test_chunk_0",
            content=content,
            attention_weights=attention_weights,
            chunk_type=ChunkType.SEMANTIC,
        )
        
        assert chunk.chunk_id == "test_chunk_0"
        assert chunk.content.shape == (256,)
        assert chunk.chunk_type == ChunkType.SEMANTIC
        assert chunk.status == ChunkStatus.ACTIVE
    
    def test_chunk_default_initialization(self):
        """测试组块默认初始化"""
        chunk = Chunk(chunk_id="default_chunk")
        
        assert chunk.content is not None
        assert chunk.attention_weights is not None
        assert chunk.attention_weights.sum().item() == pytest.approx(1.0, abs=1e-6)
    
    def test_chunk_weighted_content(self):
        """测试组块加权内容计算"""
        content = torch.tensor([1.0, 2.0, 3.0, 4.0])
        attention_weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        
        chunk = Chunk(
            chunk_id="weighted_chunk",
            content=torch.cat([content, torch.zeros(252)]),
            attention_weights=torch.cat([attention_weights, torch.zeros(252)]),
        )
        
        weighted = chunk.compute_weighted_content()
        expected = content * attention_weights
        
        assert torch.allclose(weighted[:4], expected, atol=1e-6)
    
    def test_chunk_attention_entropy(self):
        """测试注意力熵计算"""
        # 均匀分布（高熵）
        uniform_chunk = Chunk(
            chunk_id="uniform",
            content=torch.zeros(256),
            attention_weights=torch.ones(256) / 256,
        )
        uniform_entropy = uniform_chunk.get_attention_entropy()
        
        # 集中分布（低熵）
        concentrated_chunk = Chunk(
            chunk_id="concentrated",
            content=torch.zeros(256),
            attention_weights=torch.cat([torch.tensor([0.9]), torch.zeros(255)]),
        )
        concentrated_entropy = concentrated_chunk.get_attention_entropy()
        
        assert uniform_entropy > concentrated_entropy
    
    def test_chunk_validation(self):
        """测试组块验证"""
        valid_chunk = Chunk(
            chunk_id="valid",
            content=torch.randn(256),
            attention_weights=torch.ones(256) / 256,
        )
        
        is_valid, errors = valid_chunk.validate()
        assert is_valid
        assert len(errors) == 0
        
        # 无效组块（NaN）
        invalid_chunk = Chunk(
            chunk_id="invalid",
            content=torch.tensor([float('nan')] * 256),
            attention_weights=torch.ones(256) / 256,
        )
        
        is_valid, errors = invalid_chunk.validate()
        assert not is_valid
        assert len(errors) > 0
    
    def test_chunk_serialization(self):
        """测试组块序列化"""
        original = Chunk(
            chunk_id="serialize_test",
            content=torch.randn(256),
            attention_weights=torch.ones(256) / 256,
            chunk_type=ChunkType.EMOTIONAL,
            status=ChunkStatus.ACTIVE,
        )
        
        data = original.to_dict()
        restored = Chunk.from_dict(data)
        
        assert restored.chunk_id == original.chunk_id
        assert torch.allclose(restored.content, original.content)
        assert torch.allclose(restored.attention_weights, original.attention_weights)
        assert restored.chunk_type == original.chunk_type
        assert restored.status == original.status


class TestActivationStrength:
    """测试激活强度向量管理类"""
    
    def test_activation_creation(self):
        """测试激活强度创建"""
        activation_strength = ActivationStrength(
            decay_time_constant=10.0,
            min_activation=0.01,
        )
        
        assert activation_strength.decay_time_constant == 10.0
        assert activation_strength.min_activation == 0.01
        assert len(activation_strength) == 0
    
    def test_activation_add_remove(self):
        """测试添加和移除激活强度"""
        activation_strength = ActivationStrength()
        
        activation_strength.add_chunk("chunk_0", initial_activation=0.8)
        activation_strength.add_chunk("chunk_1", initial_activation=0.5)
        
        assert len(activation_strength) == 2
        assert activation_strength.get_activation("chunk_0") == pytest.approx(0.8, abs=1e-6)
        
        activation_strength.remove_chunk("chunk_0")
        assert len(activation_strength) == 1
        assert activation_strength.get_activation("chunk_0") == 0.0
    
    def test_activation_decay(self):
        """测试激活衰减机制
        
        公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a)
        """
        tau_a = 10.0  # 衰减时间常数
        activation_strength = ActivationStrength(decay_time_constant=tau_a)
        
        activation_strength.add_chunk("chunk_0", initial_activation=1.0)
        
        # 衰减 Δt = tau_a（衰减到 ~37%）
        delta_time = tau_a
        decayed = activation_strength.decay_activations(delta_time)
        
        expected_factor = np.exp(-delta_time / tau_a)
        expected_activation = 1.0 * expected_factor
        
        assert decayed["chunk_0"] == pytest.approx(expected_activation, abs=1e-3)
    
    def test_activation_min_threshold(self):
        """测试最小激活阈值
        
        低激活组块保持最小激活状态（a_min = ε > 0）
        """
        epsilon = 0.01
        activation_strength = ActivationStrength(min_activation=epsilon)
        
        activation_strength.add_chunk("chunk_0", initial_activation=0.1)
        
        # 大量衰减，应该保持在最小阈值
        large_delta_time = 1000.0  # 很长时间
        decayed = activation_strength.decay_activations(large_delta_time)
        
        assert decayed["chunk_0"] >= epsilon
        assert decayed["chunk_0"] == pytest.approx(epsilon, abs=1e-6)
    
    def test_activation_input_drive(self):
        """测试输入驱动的激活增强
        
        公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a) + InputDrive_k(t)
        """
        activation_strength = ActivationStrength()
        activation_strength.add_chunk("chunk_0", initial_activation=0.5)
        
        input_drive = {"chunk_0": 0.3, "chunk_1": 0.4}
        updated = activation_strength.apply_input_drive(input_drive)
        
        assert updated["chunk_0"] == pytest.approx(0.8, abs=1e-6)
        assert activation_strength.get_activation("chunk_1") == pytest.approx(0.4, abs=1e-6)
    
    def test_activation_decay_and_drive(self):
        """测试组合衰减和输入驱动"""
        tau_a = 10.0
        activation_strength = ActivationStrength(decay_time_constant=tau_a)
        activation_strength.add_chunk("chunk_0", initial_activation=1.0)
        
        delta_time = 5.0
        input_drive = {"chunk_0": 0.2}
        
        updated = activation_strength.decay_and_drive(delta_time, input_drive)
        
        expected_decay = np.exp(-delta_time / tau_a)
        expected_final = 1.0 * expected_decay + 0.2
        
        assert updated["chunk_0"] == pytest.approx(expected_final, abs=1e-3)
    
    def test_activation_top_n(self):
        """测试Top-N选择"""
        activation_strength = ActivationStrength()
        
        activation_strength.add_chunk("chunk_0", initial_activation=1.0)
        activation_strength.add_chunk("chunk_1", initial_activation=0.8)
        activation_strength.add_chunk("chunk_2", initial_activation=0.6)
        activation_strength.add_chunk("chunk_3", initial_activation=0.4)
        activation_strength.add_chunk("chunk_4", initial_activation=0.2)
        
        top_n = activation_strength.get_top_n_chunks(3)
        
        assert len(top_n) == 3
        assert top_n[0][0] == "chunk_0"
        assert top_n[0][1] == pytest.approx(1.0, abs=1e-6)
        assert top_n[1][0] == "chunk_1"
        assert top_n[2][0] == "chunk_2"
    
    def test_activation_statistics(self):
        """测试激活强度统计"""
        activation_strength = ActivationStrength()
        
        activations = [0.1, 0.3, 0.5, 0.7, 0.9]
        for i, a in enumerate(activations):
            activation_strength.add_chunk(f"chunk_{i}", initial_activation=a)
        
        stats = activation_strength.get_statistics()
        
        assert stats["count"] == 5
        assert stats["max"] == pytest.approx(0.9, abs=1e-6)
        assert stats["min"] == pytest.approx(0.1, abs=1e-6)
        assert stats["mean"] == pytest.approx(0.5, abs=1e-6)


class TestWorkingMemory:
    """测试完整工作记忆系统"""
    
    def test_working_memory_creation(self):
        """测试工作记忆创建"""
        wm = WorkingMemory(capacity=7)
        
        assert wm.capacity == 7
        assert len(wm) == 0
        assert wm.fast_dim == 2048
        assert wm.chunk_dim == 256
    
    def test_miller_law_capacity(self):
        """测试米勒定律容量约束（5-9范围）"""
        # 有效容量
        valid_wm = WorkingMemory(capacity=7)
        assert valid_wm.capacity == 7
        
        # 无效容量（超出范围）
        invalid_wm = WorkingMemory(capacity=15)
        assert invalid_wm.capacity == 9  # 裁剪到最大值
        
        invalid_wm2 = WorkingMemory(capacity=2)
        assert invalid_wm2.capacity == 5  # 裁剪到最小值
    
    def test_chunk_dynamic_generation(self):
        """测试组块动态生成
        
        SubTask 16.1: 实现组块动态生成（注意力绑定）
        """
        wm = WorkingMemory()
        
        # 创建快变量
        fast_state = torch.randn(wm.fast_dim)
        
        # 创建组块
        chunk = wm.create_chunk(
            source_state=fast_state,
            chunk_type=ChunkType.SEMANTIC,
            initial_activation=1.0,
        )
        
        assert chunk is not None
        assert chunk.chunk_id == "chunk_0"
        assert chunk.content.shape == (wm.chunk_dim,)
        assert chunk.status == ChunkStatus.ACTIVE
        assert len(wm) == 1
    
    def test_chunk_with_custom_attention(self):
        """测试自定义注意力权重"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        custom_attention = torch.ones(wm.chunk_dim) / wm.chunk_dim
        
        chunk = wm.create_chunk(
            source_state=fast_state,
            attention_weights=custom_attention,
            chunk_type=ChunkType.PHYSICAL,
        )
        
        assert chunk is not None
        assert torch.allclose(chunk.attention_weights, custom_attention, atol=1e-6)
    
    def test_chunk_content_formula(self):
        """测试组块形成公式
        
        C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
        """
        wm = WorkingMemory(chunk_dim=128)
        
        # 创建简单快变量
        fast_state = torch.ones(128)
        attention_weights = torch.ones(128) / 128
        
        chunk = wm.create_chunk(
            source_state=fast_state,
            attention_weights=attention_weights,
        )
        
        # 验证组块内容 = 快变量 * 注意力权重
        expected_content = fast_state * attention_weights
        assert torch.allclose(chunk.content, expected_content, atol=1e-6)
    
    def test_top_n_selection(self):
        """测试Top-N组块选择机制
        
        SubTask 17.1: 实现Top-N组块选择机制（N=7）
        """
        wm = WorkingMemory(capacity=7)
        
        # 创建10个组块（超过容量）
        fast_state = torch.randn(wm.fast_dim)
        
        for i in range(10):
            chunk = wm.create_chunk(
                source_state=fast_state,
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=1.0 - i * 0.05,  # 递减激活强度
            )
        
        # 验证激活组块数量不超过容量
        active_chunks = wm.get_active_chunks()
        dormant_chunks = wm.get_dormant_chunks()
        
        assert len(active_chunks) <= wm.capacity
        assert len(dormant_chunks) > 0
        assert len(wm) == 10
    
    def test_capacity_constraint_enforcement(self):
        """测试容量约束执行"""
        wm = WorkingMemory(capacity=7)
        
        # 创建多个组块
        for i in range(15):
            fast_state = torch.randn(wm.fast_dim)
            wm.create_chunk(
                source_state=fast_state,
                initial_activation=1.0,
            )
        
        # 验证约束
        is_valid, details = wm.validate_capacity_constraint()
        
        assert details["active_within_capacity"]
        assert details["satisfies_miller_law"]
        assert details["active_count"] <= details["capacity"]
    
    def test_activation_decay_update(self):
        """测试激活衰减与更新机制
        
        SubTask 16.3: 实现激活衰减与更新机制
        """
        wm = WorkingMemory(decay_time_constant=10.0)
        
        # 创建组块
        fast_state = torch.randn(wm.fast_dim)
        chunk = wm.create_chunk(
            source_state=fast_state,
            initial_activation=1.0,
        )
        
        initial_activation = wm.activation_strength.get_activation(chunk.chunk_id)
        
        # 执行衰减
        delta_time = 5.0
        wm.update_activations(delta_time)
        
        after_decay = wm.activation_strength.get_activation(chunk.chunk_id)
        
        # 验证衰减
        assert after_decay < initial_activation
        
        # 测试输入驱动
        input_drive = {chunk.chunk_id: 0.3}
        wm.update_activations(delta_time=0, input_drive=input_drive)
        
        after_drive = wm.activation_strength.get_activation(chunk.chunk_id)
        assert after_drive > after_decay
    
    def test_low_activation_preservation(self):
        """测试低激活组块信息保留
        
        SubTask 17.2: 实现低激活组块的信息保留（a_min=ε）
        """
        epsilon = 0.01
        wm = WorkingMemory(min_activation=epsilon)
        
        fast_state = torch.randn(wm.fast_dim)
        chunk = wm.create_chunk(
            source_state=fast_state,
            initial_activation=0.1,
        )
        
        # 大量衰减
        for _ in range(5):
            wm.update_activations(delta_time=100.0)
        
        # 验证激活强度保持在最小阈值
        activation = wm.activation_strength.get_activation(chunk.chunk_id)
        assert activation >= epsilon
        
        # 验证组块信息仍然保留
        assert wm.get_chunk(chunk.chunk_id) is not None
        assert chunk.content is not None
    
    def test_chunk_recovery(self):
        """测试组块快速恢复机制
        
        SubTask 17.3: 实现组块的快速恢复机制
        """
        wm = WorkingMemory()
        
        # 创建组块
        fast_state = torch.randn(wm.fast_dim)
        chunk = wm.create_chunk(
            source_state=fast_state,
            chunk_type=ChunkType.SEMANTIC,
            initial_activation=0.8,
        )
        
        # 移除组块（保存到历史）
        wm.remove_chunk(chunk.chunk_id, save_to_history=True)
        
        # 验证已移除但保留在历史
        assert wm.get_chunk(chunk.chunk_id) is None
        assert chunk.chunk_id in wm.get_history_chunks()
        
        # 从历史恢复
        restored = wm.restore_chunk(chunk.chunk_id, initial_activation=0.5)
        
        assert restored is not None
        assert restored.chunk_id == chunk.chunk_id
        assert torch.allclose(restored.content, chunk.content)
        assert wm.get_chunk(chunk.chunk_id) is not None
    
    def test_chunk_recovery_by_pattern(self):
        """测试按类型模式恢复组块"""
        wm = WorkingMemory()
        
        # 创建不同类型的组块
        fast_state = torch.randn(wm.fast_dim)
        
        for _ in range(2):
            wm.create_chunk(fast_state, chunk_type=ChunkType.SEMANTIC)
        for _ in range(2):
            wm.create_chunk(fast_state, chunk_type=ChunkType.EMOTIONAL)
        for _ in range(2):
            wm.create_chunk(fast_state, chunk_type=ChunkType.PHYSICAL)
        
        # 移除所有组块
        for chunk_id in list(wm.get_all_chunks().keys()):
            wm.remove_chunk(chunk_id, save_to_history=True)
        
        # 恢复语义类组块
        restored = wm.restore_chunks_by_pattern(
            pattern_type=ChunkType.SEMANTIC,
            max_restore=2,
        )
        
        assert len(restored) == 2
        for chunk in restored:
            assert chunk.chunk_type == ChunkType.SEMANTIC
    
    def test_chunk_merge(self):
        """测试组块合并"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        
        chunk1 = wm.create_chunk(fast_state, chunk_type=ChunkType.SEMANTIC)
        chunk2 = wm.create_chunk(fast_state, chunk_type=ChunkType.EMOTIONAL)
        
        # 合并组块
        merged = wm.merge_chunks([chunk1.chunk_id, chunk2.chunk_id])
        
        assert merged is not None
        assert merged.chunk_type == ChunkType.HYBRID
        
        # 验证原组块已移除
        assert wm.get_chunk(chunk1.chunk_id) is None
        assert wm.get_chunk(chunk2.chunk_id) is None
    
    def test_chunk_split(self):
        """测试组块分解"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        chunk = wm.create_chunk(fast_state, chunk_type=ChunkType.HYBRID)
        
        # 分解组块
        split_chunks = wm.split_chunk(chunk.chunk_id)
        
        assert len(split_chunks) == 2
        
        # 验证原组块已移除
        assert wm.get_chunk(chunk.chunk_id) is None
    
    def test_working_memory_output(self):
        """测试工作记忆输出（用于积分引擎）"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        
        for i in range(3):
            wm.create_chunk(
                fast_state,
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=0.8 - i * 0.1,
            )
        
        output = wm.compute_working_memory_output()
        
        assert output.shape == (wm.chunk_dim,)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_working_memory_serialization(self):
        """测试工作记忆序列化"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        for i in range(3):
            wm.create_chunk(fast_state, chunk_type=ChunkType.TEMPORARY)
        
        # 序列化
        data = wm.to_dict()
        
        # 反序列化
        restored_wm = WorkingMemory.from_dict(data)
        
        assert restored_wm.capacity == wm.capacity
        assert len(restored_wm) == len(wm)
        assert restored_wm._stats == wm._stats
    
    def test_working_memory_validation(self):
        """测试工作记忆验证"""
        wm = WorkingMemory()
        
        # 创建一些组块
        fast_state = torch.randn(wm.fast_dim)
        for i in range(5):
            wm.create_chunk(fast_state, chunk_type=ChunkType.TEMPORARY)
        
        is_valid, errors = wm.validate()
        
        assert is_valid
        assert len(errors) == 0
    
    def test_working_memory_statistics(self):
        """测试工作记忆统计信息"""
        wm = WorkingMemory()
        
        fast_state = torch.randn(wm.fast_dim)
        for i in range(3):
            wm.create_chunk(fast_state, chunk_type=ChunkType.TEMPORARY)
        
        stats = wm.get_statistics()
        
        assert stats["capacity"] == 7
        assert stats["current_chunks"] == 3
        assert stats["creation_stats"]["total_created"] == 3
        assert "activation_stats" in stats


class TestWorkingMemoryIntegration:
    """测试工作记忆与 SelfState 集成"""
    
    def test_integration_with_fast_variables(self):
        """测试与 SelfState 快变量的集成"""
        from chronos_core.core.state import SelfState
        
        # 创建 SelfState
        state = SelfState()
        
        # 创建工作记忆
        wm = WorkingMemory(fast_dim=state.FAST_DIM)
        
        # 使用快变量创建组块
        chunk = wm.create_chunk(
            source_state=state.E_fast,
            chunk_type=ChunkType.TEMPORARY,
        )
        
        assert chunk is not None
        assert chunk.content.shape == (wm.chunk_dim,)
    
    def test_integration_with_external_input(self):
        """测试与 ExternalInput 的集成"""
        from chronos_core.core.external_input import ExternalInput
        
        # 创建 ExternalInput
        input_data = ExternalInput()
        input_data.importance = 0.8
        
        # 创建工作记忆
        wm = WorkingMemory()
        
        # 创建组块
        fast_state = torch.randn(wm.fast_dim)
        chunk = wm.create_chunk(
            source_state=fast_state,
            chunk_type=ChunkType.SEMANTIC,
            initial_activation=input_data.importance,  # 使用重要性作为初始激活
        )
        
        assert chunk is not None
        assert wm.activation_strength.get_activation(chunk.chunk_id) == pytest.approx(0.8, abs=1e-6)


class TestMillerLawVerification:
    """验证米勒定律（7±2组块）"""
    
    def test_miller_law_bounds(self):
        """测试米勒定律边界（5-9）"""
        capacities = [5, 6, 7, 8, 9]
        
        for capacity in capacities:
            wm = WorkingMemory(capacity=capacity)
            assert wm.capacity == capacity
            
            # 创建超过容量的组块
            fast_state = torch.randn(wm.fast_dim)
            for i in range(15):
                wm.create_chunk(fast_state)
            
            # 验证激活组块不超过容量
            active_chunks = wm.get_active_chunks()
            assert len(active_chunks) <= wm.capacity
    
    def test_capacity_7_default(self):
        """测试默认容量为7"""
        wm = WorkingMemory()
        
        assert wm.capacity == 7
        
        is_valid, details = wm.validate_capacity_constraint()
        assert details["satisfies_miller_law"]
        assert wm.capacity == 7
    
    def test_overflow_handling(self):
        """测试超量处理"""
        wm = WorkingMemory(capacity=7)
        
        # 创建20个组块
        fast_state = torch.randn(wm.fast_dim)
        for i in range(20):
            wm.create_chunk(
                fast_state,
                initial_activation=1.0 - i * 0.01,
            )
        
        # 验证约束
        active_count = len(wm.get_active_chunks())
        dormant_count = len(wm.get_dormant_chunks())
        
        assert active_count == wm.capacity
        assert dormant_count == 20 - wm.capacity
        assert len(wm) == 20
        
        # 验证信息保留
        assert dormant_count > 0
        assert all(
            wm.get_chunk(c.chunk_id) is not None
            for c in wm.get_dormant_chunks()
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])