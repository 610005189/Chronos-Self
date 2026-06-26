"""
系统集成模块
============

整合所有核心组件，实现完整的 Chronos-Self 系统。

核心组件：
- SystemIntegration: 完整系统集成
- ChronosSystemController: 系统控制器
"""

from .system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController,
    SystemState,
    create_chronos_system_from_config,
)

__all__ = [
    'ChronosSystem',
    'ChronosSystemConfig',
    'ChronosSystemController',
    'SystemState',
    'create_chronos_system_from_config',
]