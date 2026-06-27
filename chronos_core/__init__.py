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
- validation: Validation and pattern detection
- utils: Configuration and logging utilities

License: MIT
"""

# Version management - single source of truth
from ._version import __version__, __version_info__, __author__, __license__

# Re-export for convenience
__all__ = [
    "__version__",
    "__version_info__",
    "__author__",
    "__license__",
]

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

# Import meta-cognitive module
from .core.meta_cognitive import (
    MetaCognitive,
    MetaCognitiveConfig,
    MetaCognitiveSystem,
    MetaCognitiveSystemConfig,
)

# Import reflection module
from .core.reflection import (
    Reflection,
    ReflectionConfig,
    ReflectionSystem,
    ReflectionSystemConfig,
)

# Import validation module
from .validation import (
    Validation,
    ValidationConfig,
    ValidationSystem,
    ValidationSystemConfig,
)

# Expose main components
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__license__",
    # Config and logging
    "ChronosConfig",
    "get_config",
    "init_config",
    "ChronosLogger",
    "get_logger",
    "init_logger",
    # Core state management
    "SelfState",
    "StateManager",
    "ExternalInput",
    "InputSource",
    "EvolutionHistory",
    "HistoryEntry",
    "EventType",
    "SnapshotType",
    # Meta-cognitive module
    "MetaCognitive",
    "MetaCognitiveConfig",
    "MetaCognitiveSystem",
    "MetaCognitiveSystemConfig",
    # Reflection module
    "Reflection",
    "ReflectionConfig",
    "ReflectionSystem",
    "ReflectionSystemConfig",
    # Validation module
    "Validation",
    "ValidationConfig",
    "ValidationSystem",
    "ValidationSystemConfig",
]