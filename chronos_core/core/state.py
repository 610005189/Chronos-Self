"""
SelfState 类定义 - 自我状态数据结构
包含快变量和慢变量的核心状态表示
"""

import torch
import copy
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class SelfState:
    """
    自我状态类

    包含快变量（反映短期感知和情绪反应）和慢变量（反映长期人格和目标）

    Attributes:
        E_fast: 快变量张量，维度 D_f = 2048，反映毫秒-秒级的状态变化
        E_slow: 慢变量张量，维度 D_s = 512，反映小时-天级的状态变化
        timestamp: 当前时间戳（模拟时间，单位：秒）
        history: 演化历史记录列表，每个元素为 (timestamp, E_fast_norm, E_slow_norm) 的元组
        metadata: 额外的元数据信息
    """

    # 状态变量维度常量
    FAST_DIM: int = field(default=2048, init=False, repr=False)
    SLOW_DIM: int = field(default=512, init=False, repr=False)

    # 核心状态变量
    E_fast: torch.Tensor = field(default=None)
    E_slow: torch.Tensor = field(default=None)
    timestamp: float = field(default=0.0)
    history: List[Tuple[float, float, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后处理，确保张量维度正确"""
        if self.E_fast is None:
            self.E_fast = torch.zeros(self.FAST_DIM)
            logger.debug(f"Initialized E_fast with zeros: shape {self.E_fast.shape}")
        elif isinstance(self.E_fast, torch.Tensor):
            if self.E_fast.shape[0] != self.FAST_DIM:
                raise ValueError(
                    f"E_fast dimension mismatch: expected {self.FAST_DIM}, "
                    f"got {self.E_fast.shape[0]}"
                )
        else:
            self.E_fast = torch.tensor(self.E_fast, dtype=torch.float32)

        if self.E_slow is None:
            self.E_slow = torch.zeros(self.SLOW_DIM)
            logger.debug(f"Initialized E_slow with zeros: shape {self.E_slow.shape}")
        elif isinstance(self.E_slow, torch.Tensor):
            if self.E_slow.shape[0] != self.SLOW_DIM:
                raise ValueError(
                    f"E_slow dimension mismatch: expected {self.SLOW_DIM}, "
                    f"got {self.E_slow.shape[0]}"
                )
        else:
            self.E_slow = torch.tensor(self.E_slow, dtype=torch.float32)

        # 确保张量是连续的
        self.E_fast = self.E_fast.contiguous()
        self.E_slow = self.E_slow.contiguous()

    def to_dict(self) -> Dict[str, Any]:
        """
        将状态序列化为字典

        Returns:
            包含所有状态信息的字典
        """
        return {
            "E_fast": self.E_fast.detach().cpu().numpy().tolist(),
            "E_slow": self.E_slow.detach().cpu().numpy().tolist(),
            "timestamp": self.timestamp,
            "history": self.history.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfState":
        """
        从字典反序列化状态

        Args:
            data: 包含状态信息的字典

        Returns:
            SelfState 实例
        """
        E_fast = torch.tensor(data["E_fast"], dtype=torch.float32)
        E_slow = torch.tensor(data["E_slow"], dtype=torch.float32)

        return cls(
            E_fast=E_fast,
            E_slow=E_slow,
            timestamp=data.get("timestamp", 0.0),
            history=data.get("history", []),
            metadata=data.get("metadata", {}),
        )

    def copy(self) -> "SelfState":
        """
        深拷贝当前状态

        Returns:
            新的 SelfState 实例，包含相同的数据
        """
        return SelfState(
            E_fast=self.E_fast.clone().detach(),
            E_slow=self.E_slow.clone().detach(),
            timestamp=self.timestamp,
            history=self.history.copy(),
            metadata=copy.deepcopy(self.metadata),
        )

    def record_history(self, max_length: Optional[int] = 10000):
        """
        记录当前状态的快照到历史

        Args:
            max_length: 历史记录最大长度，超过时会删除最早的记录
        """
        # 计算状态的 L2 范数作为快照
        E_fast_norm = torch.norm(self.E_fast).item()
        E_slow_norm = torch.norm(self.E_slow).item()

        self.history.append((self.timestamp, E_fast_norm, E_slow_norm))

        # 限制历史记录长度
        if max_length and len(self.history) > max_length:
            self.history = self.history[-max_length:]

    def get_fast_norm(self) -> float:
        """获取快变量的 L2 范数"""
        return torch.norm(self.E_fast).item()

    def get_slow_norm(self) -> float:
        """获取慢变量的 L2 范数"""
        return torch.norm(self.E_slow).item()

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证状态的有效性

        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []

        # 检查张量维度
        if self.E_fast.shape[0] != self.FAST_DIM:
            errors.append(
                f"E_fast dimension mismatch: expected {self.FAST_DIM}, "
                f"got {self.E_fast.shape[0]}"
            )

        if self.E_slow.shape[0] != self.SLOW_DIM:
            errors.append(
                f"E_slow dimension mismatch: expected {self.SLOW_DIM}, "
                f"got {self.E_slow.shape[0]}"
            )

        # 检查数值稳定性（NaN 和 Inf）
        if torch.isnan(self.E_fast).any():
            errors.append("E_fast contains NaN values")
        if torch.isinf(self.E_fast).any():
            errors.append("E_fast contains Inf values")
        if torch.isnan(self.E_slow).any():
            errors.append("E_slow contains NaN values")
        if torch.isinf(self.E_slow).any():
            errors.append("E_slow contains Inf values")

        # 检查时间戳
        if self.timestamp < 0:
            errors.append(f"Invalid timestamp: {self.timestamp} (should be >= 0)")

        is_valid = len(errors) == 0
        return is_valid, errors

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"SelfState(timestamp={self.timestamp:.2f}, "
            f"E_fast_norm={self.get_fast_norm():.4f}, "
            f"E_slow_norm={self.get_slow_norm():.4f}, "
            f"history_length={len(self.history)})"
        )