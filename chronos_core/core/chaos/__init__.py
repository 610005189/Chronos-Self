"""
混沌吸引子库
============

实现内源性默认模式网络所需的混沌吸引子系统，包括：
- LorenzAttractor: 洛伦兹吸引子
- RosslerAttractor: 罗斯勒吸引子
- ChuaAttractor: 蔡氏电路吸引子
- AttractorManager: 吸引子管理和选择接口
"""

from .lorenz_attractor import LorenzAttractor
from .rossler_attractor import RosslerAttractor
from .chua_attractor import ChuaAttractor
from .attractor_manager import AttractorManager
from .base_attractor import BaseAttractor, AttractorState

__all__ = [
    "BaseAttractor",
    "AttractorState",
    "LorenzAttractor",
    "RosslerAttractor",
    "ChuaAttractor",
    "AttractorManager",
]