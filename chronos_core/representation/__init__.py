"""
Chronos-Self Representation Module
===================================

This module implements the dual-channel representation system:

- SemanticEncoder: Transformer-based semantic intent encoder
- LogicalEncoder: SSM-based physical/logical encoder
- CrossAttention: Bidirectional cross-attention mechanisms
- FusionModule: Dual-channel fusion and integration

Components:
- semantic_encoder.py: Semantic intent encoding (implemented)
- logical_encoder.py: Physical/logical state encoding (implemented)
- ssm.py: Structured State Space Model core components (implemented)
- proprioceptive_encoder.py: Proprioceptive flow encoder (implemented)
- world_encoder.py: External world flow encoder (implemented)
- causal_encoder.py: Physical constraints and causal chain encoder (implemented)
- cross_attention.py: Cross-attention mechanisms (to be implemented)
- fusion_module.py: Channel fusion (to be implemented)

"""

from chronos_core.representation.semantic_encoder import (
    SemanticEncoder,
    SentimentExtractor,
    IntentExtractor,
    IntentVector,
    LightweightTransformerEncoder,
    PositionalEncoding,
    create_semantic_encoder
)

from chronos_core.representation.logical_encoder import (
    LogicalEncoder,
    create_logical_encoder
)

from chronos_core.representation.ssm import (
    StateSpaceModel,
    SSMBlock,
    StackedSSM,
    check_numerical_stability
)

from chronos_core.representation.proprioceptive_encoder import (
    ProprioceptiveEncoder,
    ProprioceptiveState
)

from chronos_core.representation.world_encoder import (
    WorldEncoder,
    WorldState
)

from chronos_core.representation.causal_encoder import (
    CausalEncoder,
    PhysicalConstraints,
    CausalChain,
    CausalReasoningModule
)

from chronos_core.representation.fusion import (
    SemanticToPhysicalCrossAttention,
    PhysicalToSemanticCrossAttention,
    FusionModule,
    FusionOutput,
    ScaledDotProductAttention,
    create_fusion_module
)

__all__ = [
    # Semantic encoder components
    'SemanticEncoder',
    'SentimentExtractor',
    'IntentExtractor',
    'IntentVector',
    'LightweightTransformerEncoder',
    'PositionalEncoding',
    'create_semantic_encoder',

    # Logical encoder components
    'LogicalEncoder',
    'create_logical_encoder',

    # SSM components
    'StateSpaceModel',
    'SSMBlock',
    'StackedSSM',
    'check_numerical_stability',

    # Proprioceptive encoder components
    'ProprioceptiveEncoder',
    'ProprioceptiveState',

    # World encoder components
    'WorldEncoder',
    'WorldState',

    # Causal encoder components
    'CausalEncoder',
    'PhysicalConstraints',
    'CausalChain',
    'CausalReasoningModule',

    # Fusion components
    'SemanticToPhysicalCrossAttention',
    'PhysicalToSemanticCrossAttention',
    'FusionModule',
    'FusionOutput',
    'ScaledDotProductAttention',
    'create_fusion_module',
]