"""
核心数据结构验证测试
验证 Task 2 的所有组件是否正确实现
"""

import torch
import sys
import os

# 确保可以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chronos_core import (
    SelfState,
    StateManager,
    ExternalInput,
    InputSource,
    EvolutionHistory,
    EventType,
    SnapshotType,
)


def test_self_state():
    """测试 SelfState 类"""
    print("\n=== 测试 SelfState ===")

    # 创建默认状态
    state = SelfState()
    print(f"默认状态: {state}")

    # 验证维度
    assert state.E_fast.shape[0] == 2048, "快变量维度应为 2048"
    assert state.E_slow.shape[0] == 512, "慢变量维度应为 512"
    print(f"✓ 维度验证通过: E_fast={state.E_fast.shape}, E_slow={state.E_slow.shape}")

    # 测试序列化
    state_dict = state.to_dict()
    reconstructed_state = SelfState.from_dict(state_dict)
    assert torch.allclose(state.E_fast, reconstructed_state.E_fast), "序列化后快变量应一致"
    assert torch.allclose(state.E_slow, reconstructed_state.E_slow), "序列化后慢变量应一致"
    print("✓ 序列化/反序列化测试通过")

    # 测试克隆
    cloned_state = state.copy()
    assert torch.allclose(state.E_fast, cloned_state.E_fast), "克隆后快变量应一致"
    assert cloned_state is not state, "克隆应创建新实例"
    print("✓ 克隆测试通过")

    # 测试验证
    is_valid, errors = state.validate()
    assert is_valid, f"状态应该有效，但发现错误: {errors}"
    print("✓ 验证测试通过")

    # 测试历史记录
    state.record_history()
    assert len(state.history) > 0, "历史记录应该有数据"
    print(f"✓ 历史记录测试通过，历史长度: {len(state.history)}")

    print("SelfState 所有测试通过 ✓")


def test_state_manager():
    """测试 StateManager 类"""
    print("\n=== 测试 StateManager ===")

    # 创建管理器
    manager = StateManager()
    print(f"管理器: {manager}")

    # 初始化随机状态
    state1 = manager.initialize_state("test_state_1", init_method="random")
    print(f"✓ 初始化随机状态: {state1}")

    # 初始化零状态
    state2 = manager.initialize_state("test_state_2", init_method="zeros")
    assert torch.all(state2.E_fast == 0), "零初始化应该全为 0"
    print(f"✓ 初始化零状态: {state2}")

    # 测试自定义初始化
    custom_E_fast = torch.randn(2048)
    custom_E_slow = torch.randn(512)
    state3 = manager.initialize_state(
        "test_state_3",
        init_method="custom",
        E_fast_init=custom_E_fast,
        E_slow_init=custom_E_slow,
    )
    assert torch.allclose(state3.E_fast, custom_E_fast), "自定义初始化应该匹配"
    print(f"✓ 自定义初始化: {state3}")

    # 测试状态更新
    delta_fast = torch.randn(2048) * 0.01
    delta_slow = torch.randn(512) * 0.01
    updated_state = manager.update_state("test_state_1", delta_fast, delta_slow, dt=0.1)
    print(f"✓ 状态更新: {updated_state}")

    # 测试状态验证
    is_valid, errors = manager.validate_state("test_state_1")
    assert is_valid, f"状态应该有效: {errors}"
    print("✓ 状态验证通过")

    # 测试活跃状态
    active = manager.get_active_state()
    assert active is not None, "应该有活跃状态"
    print(f"✓ 活跃状态: {active}")

    # 测试状态列表
    states = manager.list_states()
    assert len(states) == 3, f"应该有 3 个状态，实际有 {len(states)}"
    print(f"✓ 状态列表: {states}")

    # 测试统计信息
    stats = manager.get_state_statistics("test_state_1")
    print(f"✓ 统计信息: {stats}")

    print("StateManager 所有测试通过 ✓")


def test_external_input():
    """测试 ExternalInput 类"""
    print("\n=== 测试 ExternalInput ===")

    # 创建默认输入
    input1 = ExternalInput()
    print(f"默认输入: {input1}")

    # 验证维度
    assert input1.X_sem.shape[0] == 256, "语义流维度应为 256"
    assert input1.X_log.shape[0] == 512, "物理流维度应为 512"
    assert input1.X_proprio.shape[0] == 256, "本体感觉流维度应为 256"
    assert input1.X_world.shape[0] == 256, "外部世界流维度应为 256"
    print("✓ 维度验证通过")

    # 测试带参数的输入
    input2 = ExternalInput(
        X_sem=torch.randn(256),
        X_log=torch.randn(512),
        timestamp=10.5,
        source=InputSource.TEXT,
        importance=0.8,
        emotional_intensity=0.9,
    )
    print(f"✓ 创建输入: {input2}")

    # 测试高情感强度判断
    assert input2.is_high_emotional(0.7), "情感强度 0.9 应该高于阈值 0.7"
    print("✓ 高情感强度判断通过")

    # 测试组合物理流
    combined = input2.get_combined_physical_flow()
    assert combined.shape[0] == 512, "组合物理流应为 512 维"
    print(f"✓ 组合物理流维度: {combined.shape}")

    # 测试序列化
    input_dict = input2.to_dict()
    reconstructed_input = ExternalInput.from_dict(input_dict)
    assert torch.allclose(input2.X_sem, reconstructed_input.X_sem), "序列化后语义流应一致"
    print("✓ 序列化/反序列化测试通过")

    # 测试验证
    is_valid, errors = input2.validate()
    assert is_valid, f"输入应该有效: {errors}"
    print("✓ 验证测试通过")

    print("ExternalInput 所有测试通过 ✓")


def test_evolution_history():
    """测试 EvolutionHistory 类"""
    print("\n=== 测试 EvolutionHistory ===")

    # 创建历史记录系统
    history = EvolutionHistory()
    print(f"历史系统: {history}")

    # 测试记录状态
    state = SelfState(
        E_fast=torch.randn(2048),
        E_slow=torch.randn(512),
        timestamp=1.0,
    )
    entry1 = history.record_state(state, EventType.STATE_UPDATE)
    print(f"✓ 记录状态: {entry1}")

    # 测试记录输入
    input_data = ExternalInput(
        X_sem=torch.randn(256),
        X_log=torch.randn(512),
        timestamp=2.0,
        source=InputSource.TEXT,
        emotional_intensity=0.8,
    )
    entry2 = history.record_input(input_data, state)
    assert entry2.is_keyframe, "高情感强度应该标记为关键帧"
    print(f"✓ 记录输入（关键帧）: {entry2}")

    # 测试记录响应
    response_data = {"action": "respond", "content": "test"}
    entry3 = history.record_response(response_data, timestamp=3.0, state=state)
    print(f"✓ 记录响应: {entry3}")

    # 测试手动标记关键帧
    history.mark_keyframe(5.0, "manual_test", {"reason": "user_marked"})
    print("✓ 手动标记关键帧")

    # 测试查询
    entries = history.query_by_time_range(0.0, 5.0)
    assert len(entries) > 0, "应该有查询结果"
    print(f"✓ 时间范围查询: 找到 {len(entries)} 条记录")

    # 测试事件类型查询
    state_entries = history.query_by_event_type(EventType.STATE_UPDATE)
    assert len(state_entries) > 0, "应该有状态更新事件"
    print(f"✓ 事件类型查询: 找到 {len(state_entries)} 条状态更新")

    # 测试获取关键帧
    keyframes = history.get_keyframes()
    assert len(keyframes) > 0, "应该有关键帧"
    print(f"✓ 获取关键帧: {len(keyframes)} 个")

    # 测试统计信息
    stats = history.get_statistics()
    print(f"✓ 统计信息: {stats}")

    # 测试导出（创建临时目录）
    export_dir = "test_exports"
    os.makedirs(export_dir, exist_ok=True)

    json_path = os.path.join(export_dir, "test_history.json")
    history.export_to_json(json_path)
    assert os.path.exists(json_path), "JSON 文件应该存在"
    print(f"✓ 导出 JSON: {json_path}")

    csv_path = os.path.join(export_dir, "test_history.csv")
    history.export_to_csv(csv_path)
    assert os.path.exists(csv_path), "CSV 文件应该存在"
    print(f"✓ 导出 CSV: {csv_path}")

    # 测试加载
    new_history = EvolutionHistory()
    new_history.load_from_json(json_path)
    assert len(new_history) == len(history), "加载后历史长度应该一致"
    print(f"✓ 加载历史: {new_history}")

    # 清理测试文件
    import shutil
    shutil.rmtree(export_dir)
    print("✓ 清理测试文件")

    print("EvolutionHistory 所有测试通过 ✓")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("Chronos-Self 核心数据结构验证测试")
    print("=" * 60)

    try:
        test_self_state()
        test_state_manager()
        test_external_input()
        test_evolution_history()

        print("\n" + "=" * 60)
        print("所有测试通过 ✓✓✓")
        print("=" * 60)

        print("\n实现总结：")
        print("1. SelfState ✓ - 快慢变量、序列化、验证功能完整")
        print("2. StateManager ✓ - 多状态管理、初始化、更新、保存功能完整")
        print("3. ExternalInput ✓ - 双通道流、重要性、情感强度标记功能完整")
        print("4. EvolutionHistory ✓ - 状态快照、查询、导出、关键帧功能完整")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)