"""
Chronos-Self 反思与离线回放模块
================================

实现系统自我修正和学习进化的关键机制：

核心功能：
- 实时反思机制（RealtimeReflection）：维护最近T步计算图，有限截断伴随法梯度回传
- 离线回放系统（OfflineReplay）：定时触发机制，关键帧向量数据库存储
- 离线期梯度更新（OfflineUpdater）：伴随法梯度回传，仅更新积分引擎参数
- 完整反思系统（ReflectionSystem）：整合实时反思和离线回放

反思流程：
1. 在线运行：实时反思机制修正近期演化轨迹
2. 达到时间阈值：触发离线期
3. 离线回放：离线回放更新参数
4. 重新运行：应用更新后的参数
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

# Import version from single source
from ..._version import __version__