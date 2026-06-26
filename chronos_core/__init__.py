"""
Chronos-Self: Self-Referential Continuous Dynamics System
==========================================================

This package implements a self-referential continuous dynamics system with
emergent consciousness properties. The system is based on:

- Multi-scale continuous dynamics (fast/slow variable coupling)
- Dual-channel representation (semantic + physical encoding)
- Hierarchical meta-cognition (L0/L1/L2 layers)
- Chaos injection for edge-of-chaos dynamics
- Sleep replay and reflection mechanisms

Main Components:
- core: Core dynamics engine and state management
- representation: Dual-channel encoders and fusion mechanisms
- memory: Four-layer memory architecture and working memory
- training: Training systems and loss functions
- validation: Validation and emergence detection
- utils: Configuration and logging utilities

Version: 0.1.0
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Chronos-Self Team"

# Import key utilities
from .utils.config import ChronosConfig, get_config, init_config
from .utils.logger import ChronosLogger, get_logger, init_logger

# Import core components
from .core import (
    SelfState,
    StateManager,
    ExternalInput,
    InputSource,
    EvolutionHistory,
    HistoryEntry,
    EventType,
    SnapshotType,
)

# Expose main components
__all__ = [
    "__version__",
    "__author__",
    "ChronosConfig",
    "get_config",
    "init_config",
    "ChronosLogger",
    "get_logger",
    "init_logger",
    "SelfState",
    "StateManager",
    "ExternalInput",
    "InputSource",
    "EvolutionHistory",
    "HistoryEntry",
    "EventType",
    "SnapshotType",
]