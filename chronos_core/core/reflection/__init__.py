"""
Chronos-Self 反思机制模块
===========================

实现系统自我修正和学习进化的关键机制：

核心功能：
- 实时反思机制（RealtimeReflection）：维护最近T步计算图，有限截断伴随法梯度回传
- 睡眠重放系统（SleepReplay）：24小时触发机制，关键帧向量数据库存储
- 睡眠期梯度更新（SleepUpdater）：伴随法梯度回传，仅更新积分引擎参数
- 完整反思系统（ReflectionSystem）：整合实时反思和睡眠重放

反思流程：
1. 在线运行：实时反思机制修正近期演化轨迹
2. 达到24小时：触发睡眠期
3. 离线重放：睡眠重放更新参数
4. 重新运行：应用更新后的参数

Task 18-20 实现：
- Task 18: 实时反思机制
  - SubTask 18.1: 最近T步计算图维护（T=1000）
  - SubTask 18.2: 有限截断伴随法梯度回传
  - SubTask 18.3: 实时轨迹修正
  - SubTask 18.4: 实时反思计算效率测试

- Task 19: 睡眠重放系统
  - SubTask 19.1: 24小时触发机制
  - SubTask 19.2: 关键帧向量数据库存储
  - SubTask 19.3: 重放一致性损失计算
  - SubTask 19.4: 预测改善损失计算

- Task 20: 睡眠期梯度更新
  - SubTask 20.1: 从关键帧向前积分的重放流程
  - SubTask 20.2: 伴随法梯度回传（仅更新积分引擎参数）
  - SubTask 20.3: 不修改关键帧本身的约束
  - SubTask 20.4: 睡眠重放稳定性测试
"""

from .realtime_reflection import (
    RealtimeReflection,
    RealtimeReflectionConfig,
    ComputationGraphBuffer,
    TruncatedAdjointMethod,
    create_realtime_reflection_from_config,
)

from .sleep_replay import (
    SleepReplay,
    SleepReplayConfig,
    KeyframeDatabase,
    KeyframeData,
    ReplayLossCalculator,
    create_sleep_replay_from_config,
)

from .sleep_updater import (
    SleepUpdater,
    SleepUpdaterConfig,
    GradientConstraints,
    StabilityChecker,
    create_sleep_updater_from_config,
)

from .reflection_system import (
    Reflection,
    ReflectionConfig,
    ReflectionState,
    ReflectionMode,
    ReflectionSystem,
    ReflectionSystemConfig,
    create_reflection_from_config,
    create_reflection_system_from_config,
)


__all__ = [
    # 实时反思
    "RealtimeReflection",
    "RealtimeReflectionConfig",
    "ComputationGraphBuffer",
    "TruncatedAdjointMethod",
    "create_realtime_reflection_from_config",
    
    # 睡眠重放
    "SleepReplay",
    "SleepReplayConfig",
    "KeyframeDatabase",
    "KeyframeData",
    "ReplayLossCalculator",
    "create_sleep_replay_from_config",
    
    # 睡眠期梯度更新
    "SleepUpdater",
    "SleepUpdaterConfig",
    "GradientConstraints",
    "StabilityChecker",
    "create_sleep_updater_from_config",
    
    # 反思模块（新名称）
    "Reflection",
    "ReflectionConfig",
    "ReflectionState",
    "ReflectionMode",
    "create_reflection_from_config",
    
    # 完整反思系统（向后兼容）
    "ReflectionSystem",
    "ReflectionSystemConfig",
    "create_reflection_system_from_config",
]

__version__ = "0.1.0"