"""
Chronos-Self Utilities Module
=============================

This module provides configuration, logging, and utility functions:

- ChronosConfig: Master configuration class with all hyperparameters
- ChronosLogger: Multi-level logging system with file/console output
- Helper utilities for data processing and visualization

Components:
- config.py: Configuration management system
- logger.py: Logging system with structured output
- helpers.py: Miscellaneous helper functions (to be implemented)
- visualization.py: Visualization utilities (to be implemented)
"""

from .config import (
    ChronosConfig,
    get_config,
    set_config,
    init_config,
    DimensionalityConfig,
    MemoryTemporalConfig,
    CouplingStabilityConfig,
    ChaosInjectionConfig,
    TrainingConfig,
    NeuralODEConfig,
    EncoderConfig,
    MetaCognitiveConfig,
    ValidationConfig,
    LoggingConfig,
    PathsConfig,
)

from .logger import (
    ChronosLogger,
    get_logger,
    set_logger,
    init_logger,
    debug,
    info,
    warning,
    error,
    critical,
    exception,
)

__all__ = [
    "ChronosConfig",
    "get_config",
    "set_config",
    "init_config",
    "DimensionalityConfig",
    "MemoryTemporalConfig",
    "CouplingStabilityConfig",
    "ChaosInjectionConfig",
    "TrainingConfig",
    "NeuralODEConfig",
    "EncoderConfig",
    "MetaCognitiveConfig",
    "ValidationConfig",
    "LoggingConfig",
    "PathsConfig",
    "ChronosLogger",
    "get_logger",
    "set_logger",
    "init_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    "exception",
]