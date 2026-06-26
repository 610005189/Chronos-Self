"""
Chronos-Self Training Module
============================

This module implements the training system and loss functions:

- TrainingSystem: Complete training system with all components
- LossFunctions: Combined losses (prediction, anti-quietus, inertia)
- DynamicsAlignment: Dynamics alignment with multi-step consistency
- FreezingStrategy: Phase-based parameter freezing

Implemented components:
- loss_functions.py: Loss function implementations
- dynamics_alignment.py: Dynamics alignment training
- freezing_strategy.py: Parameter freezing strategies
- training_system.py: Main training loop
"""

from chronos_core.training.loss_functions import (
    LossFunctions,
    LossFunctionsConfig,
    PredictionLoss,
    AntiDecayLoss,
    InertiaRegularizationLoss,
    create_loss_functions_from_config,
)

from chronos_core.training.dynamics_alignment import (
    DynamicsAlignment,
    DynamicsAlignmentConfig,
    MultiStepConsistencyLoss,
    SemigroupRegularizationLoss,
    LongSequenceOpenLoopLoss,
    PeriodicValidation,
    create_dynamics_alignment_from_config,
)

from chronos_core.training.freezing_strategy import (
    FreezingStrategy,
    FreezingStrategyConfig,
    L0EncoderFreezing,
    L1IntegrationTraining,
    FixedProjectionFreezing,
    FreezingValidator,
    create_freezing_strategy_from_config,
)

from chronos_core.training.training_system import (
    TrainingSystem,
    TrainingSystemConfig,
    TrainingHistory,
    create_training_system_from_config,
)

__all__ = [
    # Loss Functions
    'LossFunctions',
    'LossFunctionsConfig',
    'PredictionLoss',
    'AntiDecayLoss',
    'InertiaRegularizationLoss',
    'create_loss_functions_from_config',

    # Dynamics Alignment
    'DynamicsAlignment',
    'DynamicsAlignmentConfig',
    'MultiStepConsistencyLoss',
    'SemigroupRegularizationLoss',
    'LongSequenceOpenLoopLoss',
    'PeriodicValidation',
    'create_dynamics_alignment_from_config',

    # Freezing Strategy
    'FreezingStrategy',
    'FreezingStrategyConfig',
    'L0EncoderFreezing',
    'L1IntegrationTraining',
    'FixedProjectionFreezing',
    'FreezingValidator',
    'create_freezing_strategy_from_config',

    # Training System
    'TrainingSystem',
    'TrainingSystemConfig',
    'TrainingHistory',
    'create_training_system_from_config',
]