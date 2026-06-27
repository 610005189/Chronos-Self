"""
Chronos-Self Core 模块
核心数据结构与状态管理系统

包含：
- SelfState: 系统状态类（快变量和慢变量）
- StateManager: 状态管理器（初始化、更新、保存、加载）
- ExternalInput: 外部输入类（语义流和物理流）
- EvolutionHistory: 演化历史记录系统
- Chaos: 混沌吸引子库（Lorenz, Rossler, Chua）
- ChaosInjector: 高维混沌注入器
- DefaultModeNetwork: 自持默认模式网络系统
- Neural ODE 求解器
- 快变量动力学系统
- 慢变量动力学系统
- 非对称耦合与稳定性机制
- 完整积分引擎
- 反思机制系统：
  - 实时反思机制（RealtimeReflection）
  - 离线回放系统（SleepReplay）
  - 离线期梯度更新（SleepUpdater）
  - 完整反思系统（ReflectionSystem）
"""

from .state import SelfState
from .state_manager import StateManager
from .external_input import ExternalInput, InputSource
from .history import EvolutionHistory, HistoryEntry, EventType, SnapshotType

# 混沌吸引子系统
from .chaos import (
    BaseAttractor,
    AttractorState,
    LorenzAttractor,
    RosslerAttractor,
    ChuaAttractor,
    AttractorManager,
)

# 混沌注入系统
from .chaos_injector import (
    ChaosInjector,
    CoreSubspaceProjector,
    InjectionConfig,
)

# 默认模式网络系统
from .dmn_system import (
    DefaultModeNetwork,
    DMNConfig,
    DMNState,
    create_dmn_from_config,
)

# Neural ODE 求解器
from .neural_ode import (
    NeuralODESolver,
    ODESolverConfig,
    DynamicsFunction,
    create_solver_from_config,
)

# 快变量动力学系统
from .fast_dynamics import (
    FastDynamicsSystem,
    FastDynamicsFunction,
    FastDynamicsConfig,
    EvolutionFunctionMLP,
    EvolutionFunctionTransformer,
    create_fast_dynamics_from_config,
)

# 慢变量动力学系统
from .slow_dynamics import (
    SlowDynamicsSystem,
    SlowDynamicsFunction,
    SlowDynamicsConfig,
    PoolingMechanism,
    SpontaneousEvolution,
    create_slow_dynamics_from_config,
)

# 非对称耦合与稳定性机制
from .coupling import (
    CouplingAndStabilitySystem,
    AdaptiveCouplingCoefficients,
    StabilityMonitor,
    CouplingConfig,
    create_coupling_system_from_config,
)

# 完整积分引擎
from .integration_engine import (
    IntegrationEngine,
    IntegrationEngineConfig,
    create_integration_engine_from_config,
)

# 反思机制系统（Task 18-20）
from .reflection import (
    # 实时反思
    RealtimeReflection,
    RealtimeReflectionConfig,
    ComputationGraphBuffer,
    TruncatedAdjointMethod,
    create_realtime_reflection_from_config,
    # 睡眠重放
    SleepReplay,
    SleepReplayConfig,
    KeyframeDatabase,
    KeyframeData,
    ReplayLossCalculator,
    create_sleep_replay_from_config,
    # 睡眠期梯度更新
    SleepUpdater,
    SleepUpdaterConfig,
    GradientConstraints,
    StabilityChecker,
    create_sleep_updater_from_config,
    # 完整反思系统
    ReflectionSystem,
    ReflectionSystemConfig,
    ReflectionState,
    ReflectionMode,
    create_reflection_system_from_config,
)

# Import version from single source
from .._version import __version__

__all__ = [
    # 核心状态管理
    "SelfState",
    "StateManager",
    "ExternalInput",
    "InputSource",
    "EvolutionHistory",
    "HistoryEntry",
    "EventType",
    "SnapshotType",
    # 混沌吸引子
    "BaseAttractor",
    "AttractorState",
    "LorenzAttractor",
    "RosslerAttractor",
    "ChuaAttractor",
    "AttractorManager",
    # 混沌注入
    "ChaosInjector",
    "CoreSubspaceProjector",
    "InjectionConfig",
    # 默认模式网络
    "DefaultModeNetwork",
    "DMNConfig",
    "DMNState",
    "create_dmn_from_config",
    # Neural ODE 求解器
    "NeuralODESolver",
    "ODESolverConfig",
    "DynamicsFunction",
    "create_solver_from_config",
    # 快变量动力学
    "FastDynamicsSystem",
    "FastDynamicsFunction",
    "FastDynamicsConfig",
    "EvolutionFunctionMLP",
    "EvolutionFunctionTransformer",
    "create_fast_dynamics_from_config",
    # 慢变量动力学
    "SlowDynamicsSystem",
    "SlowDynamicsFunction",
    "SlowDynamicsConfig",
    "PoolingMechanism",
    "SpontaneousEvolution",
    "create_slow_dynamics_from_config",
    # 耦合与稳定性
    "CouplingAndStabilitySystem",
    "AdaptiveCouplingCoefficients",
    "StabilityMonitor",
    "CouplingConfig",
    "create_coupling_system_from_config",
    # 积分引擎
    "IntegrationEngine",
    "IntegrationEngineConfig",
    "create_integration_engine_from_config",
    # 反思机制（Task 18-20）
    "RealtimeReflection",
    "RealtimeReflectionConfig",
    "ComputationGraphBuffer",
    "TruncatedAdjointMethod",
    "create_realtime_reflection_from_config",
    "SleepReplay",
    "SleepReplayConfig",
    "KeyframeDatabase",
    "KeyframeData",
    "ReplayLossCalculator",
    "create_sleep_replay_from_config",
    "SleepUpdater",
    "SleepUpdaterConfig",
    "GradientConstraints",
    "StabilityChecker",
    "create_sleep_updater_from_config",
    "ReflectionSystem",
    "ReflectionSystemConfig",
    "ReflectionState",
    "ReflectionMode",
    "create_reflection_system_from_config",
]