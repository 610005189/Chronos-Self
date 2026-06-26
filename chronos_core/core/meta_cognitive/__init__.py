"""
元认知调控系统 - Meta-Cognitive Control System
===============================================

层级化自指与元认知调控系统，实现自指递归截断机制。

层级结构：
- L0: 感知层（Perception Layer）- 无自指能力，仅处理感知
- L1: 自我状态层（Self-State Layer）- 完整认知积分，包含自我状态
- L2: 元认知层（Meta-Cognitive Layer）- 高阶调控，物理隔离于L0

核心模块：
- perception_layer: L0感知层实现
- self_state_layer: L1自我状态层实现
- meta_cognitive_layer: L2元认知层实现
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
    MetaCognitiveSystem,
    MetaCognitiveSystemConfig,
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
    
    # 完整系统
    "MetaCognitiveSystem",
    "MetaCognitiveSystemConfig",
]