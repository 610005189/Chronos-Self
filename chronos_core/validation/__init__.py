"""
Chronos-Self Validation Module
==============================

Complete validation and pattern detection system for Chronos-Self.

This module implements comprehensive validation and pattern detection systems:

**Validation Levels:**
- P0 (Core Dynamics): 72-hour open-loop test, baseline drift, Lyapunov exponent, dynamics alignment
- P1 (Functional Modules): DMN, working memory, L2 independence
- P2 (Pattern Detection): Dynamics indicators + behavioral indicators

**Validation Modes:**
- QUICK: Minute-level, key indicators
- FULL: Hour-level, all indicators
- CONTINUOUS: Long-term, real-time monitoring
- P0_ONLY: Core dynamics only
- PATTERN_DETECTION: Pattern detection only

**Components:**
- P0Validation: Core dynamics validation
- DynamicsMonitoring: Dynamics indicators monitoring
- BehavioralMetrics: Behavioral indicators assessment
- Validation: Complete validation module integration

**Validation Criteria:**
- Dynamics indicators: ρ(τ) > 0.3, λ_max ∈ (0, 0.1), E_self ∈ [ε_min, ε_max]
- Behavioral indicators: Intent entropy transition, transfer rate transition, S-shaped recovery
- Pattern detection: 3 dynamics indicators + 2 behavioral indicators

Usage:
    from chronos_core.validation import Validation
    
    validation = Validation(config)
    result = validation.run_validation(engine, mode='full')
    validation.save_final_report(result)
"""

# Import all validation components
from .validation_system import (
    ValidationMode,
    ValidationLevel,
    ValidationConfig,
    ValidationResult,
    Validation,
    ValidationSystemConfig,
    ValidationSystem
)

from .p0_validation import (
    P0Validation,
    P0ValidationResult,
    P0ValidationConfig
)

from .dynamics_monitoring import (
    DynamicsMonitoring,
    DynamicsIndicators,
    DynamicsMonitoringConfig
)

from .behavioral_metrics import (
    BehavioralMetrics,
    BehavioralIndicators,
    BehavioralMetricsConfig
)

# Module exports
__all__ = [
    # Validation module
    'ValidationMode',
    'ValidationLevel',
    'ValidationConfig',
    'ValidationResult',
    'Validation',
    
    # Validation system (backward compatible)
    'ValidationSystemConfig',
    'ValidationSystem',
    
    # P0 validation
    'P0Validation',
    'P0ValidationResult',
    'P0ValidationConfig',
    
    # Dynamics monitoring
    'DynamicsMonitoring',
    'DynamicsIndicators',
    'DynamicsMonitoringConfig',
    
    # Behavioral metrics
    'BehavioralMetrics',
    'BehavioralIndicators',
    'BehavioralMetricsConfig'
]


# Module version
__version__ = '1.0.0'