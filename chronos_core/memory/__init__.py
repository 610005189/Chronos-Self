"""
Chronos-Self Memory Module
==========================

This module implements the four-layer memory architecture:

- WorkingMemory: Miller's law based working memory (7 chunks)
- ShortTermMemory: Recent trajectory buffer
- LongTermMemory: Vector database for keyframes
- EpisodicMemory: Sleep replay mechanism

Components implemented:
- work_memory.py: Working memory with activation strength and chunk management

Components to be implemented:
- short_term_memory.py: Recent trajectory storage
- long_term_memory.py: Vector database integration
- episodic_memory.py: Sleep replay system
"""

from .work_memory import (
    Chunk,
    ChunkType,
    ChunkStatus,
    ActivationStrength,
    WorkingMemory,
)

__all__ = [
    # Working Memory Components (Phase 6, Task 16-17)
    "Chunk",
    "ChunkType",
    "ChunkStatus",
    "ActivationStrength",
    "WorkingMemory",
]