"""
递归状态监控系统 - Recursive State Monitoring System
=====================================================

三层递归状态监控架构，实现状态监控截断机制。

层级结构：
- L0: 感知层（Perception Layer）- 仅处理感知输入
- L1: 状态层（State Layer）- 完整状态积分，包含系统状态
- L2: 监控层（Monitoring Layer）- 高阶调控，物理隔离于L0

核心模块：
- perception_layer: L0感知层实现
- self_state_layer: L1状态层实现
- meta_cognitive_layer: L2监控层实现
- meta_cognitive_manager: L2扰动训练与独立性验证
- meta_cognitive_system: 完整系统整合
"""

from .perception_layer import (
    PerceptionLayer,
    PerceptionLayerConfig,
    RAGKnowledgeBase,
    PerceptionEncoder,
    PerceptionFilter,
)

from .self_state_layer import (
    SelfStateLayer,
    SelfStateLayerConfig,
    AttentionFocusManager,
    WorkingMemoryIntegrator,
)

from .meta_cognitive_layer import (
    MetaCognitiveLayer,
    MetaCognitiveLayerConfig,
    JohnsonLindenstraussProjection,
    HighOrderStatisticsExtractor,
    MetaParameterController,
)

from .meta_cognitive_manager import (
    MetaCognitiveManager,
    MetaCognitiveManagerConfig,
    L2PerturbationTrainer,
    L2AblationTester,
)

from .meta_cognitive_system import (
    MetaCognitive,
    MetaCognitiveConfig,
    MetaCognitiveSystem,
    MetaCognitiveSystemConfig,
    create_meta_cognitive_from_config,
    create_meta_cognitive_system_from_config,
)

from .curiosity_engine import (
    CuriosityEngine,
    CuriosityConfig,
    CuriosityMetrics,
    NoveltyDetector,
    ComplexityScorer,
    UncertaintyEstimator,
    InputPriorityQueue,
    PrioritizedInput,
    create_curiosity_engine_from_config,
)

from .competitive_emergence import (
    CompetitiveEmergence,
    EmergenceConfig,
    EmergenceState,
    EmergenceIndicators,
    EmergenceStatistics,
)

__all__ = [
    # L0 感知层
    "PerceptionLayer",
    "PerceptionLayerConfig",
    "RAGKnowledgeBase",
    "PerceptionEncoder",
    "PerceptionFilter",
    
    # L1 自我状态层
    "SelfStateLayer",
    "SelfStateLayerConfig",
    "AttentionFocusManager",
    "WorkingMemoryIntegrator",
    
    # L2 元认知层
    "MetaCognitiveLayer",
    "MetaCognitiveLayerConfig",
    "JohnsonLindenstraussProjection",
    "HighOrderStatisticsExtractor",
    "MetaParameterController",
    
    # 元认知管理
    "MetaCognitiveManager",
    "MetaCognitiveManagerConfig",
    "L2PerturbationTrainer",
    "L2AblationTester",
    
    # 元认知模块（新名称）
    "MetaCognitive",
    "MetaCognitiveConfig",
    "create_meta_cognitive_from_config",
    
    # 完整系统（向后兼容）
    "MetaCognitiveSystem",
    "MetaCognitiveSystemConfig",
    "create_meta_cognitive_system_from_config",
    
    # 好奇心引擎
    "CuriosityEngine",
    "CuriosityConfig",
    "CuriosityMetrics",
    "NoveltyDetector",
    "ComplexityScorer",
    "UncertaintyEstimator",
    "InputPriorityQueue",
    "PrioritizedInput",
    "create_curiosity_engine_from_config",
    
    # 竞争性涌现
    "CompetitiveEmergence",
    "EmergenceConfig",
    "EmergenceState",
    "EmergenceIndicators",
    "EmergenceStatistics",
]
