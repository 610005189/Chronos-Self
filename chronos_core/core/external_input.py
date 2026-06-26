"""
ExternalInput 类定义 - 外部输入数据结构
包含语义意图流和逻辑物理流的双通道输入表示
"""

import torch
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class InputSource(Enum):
    """输入来源标识"""

    TEXT = "text"  # 文本输入
    SPEECH = "speech"  # 语音输入
    VISION = "vision"  # 视觉输入
    SENSOR = "sensor"  # 传感器输入
    INTERNAL = "internal"  # 内部生成（如梦境、反思）
    SYSTEM = "system"  # 系统输入
    UNKNOWN = "unknown"  # 未知来源


@dataclass
class ExternalInput:
    """
    外部输入类

    包含语义意图流和逻辑物理流的双通道表示

    Attributes:
        X_sem: 语义意图流，维度 d_1，捕获高层目标和情感倾向
        X_log: 逻辑物理流，维度 d_2，捕获结构化物理约束
        X_proprio: 本体感觉流（物理流的子部分），反映内部状态（姿态、能量、资源占用）
        X_world: 外部世界流（物理流的子部分），反映环境状态
        timestamp: 输入时间戳（秒）
        source: 输入来源标识
        metadata: 额外的元数据信息
        importance: 输入重要性权重（0.0-1.0）
        emotional_intensity: 情感强度（0.0-1.0），用于关键帧标记
    """

    # 语义流和物理流的维度（可配置）
    semantic_dim: int = field(default=256, init=False, repr=False)
    physical_dim: int = field(default=512, init=False, repr=False)
    proprio_dim: int = field(default=256, init=False, repr=False)  # 本体感觉维度
    world_dim: int = field(default=256, init=False, repr=False)  # 外部世界维度

    # 核心输入张量
    X_sem: torch.Tensor = field(default=None)
    X_log: torch.Tensor = field(default=None)
    X_proprio: torch.Tensor = field(default=None)
    X_world: torch.Tensor = field(default=None)

    # 时间戳和来源信息
    timestamp: float = field(default=0.0)
    source: InputSource = field(default=InputSource.UNKNOWN)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 重要性和情感强度
    importance: float = field(default=1.0)
    emotional_intensity: float = field(default=0.0)

    def __post_init__(self):
        """初始化后处理，确保张量维度正确"""
        # 初始化语义流
        if self.X_sem is None:
            self.X_sem = torch.zeros(self.semantic_dim)
            logger.debug(f"Initialized X_sem with zeros: shape {self.X_sem.shape}")
        elif isinstance(self.X_sem, torch.Tensor):
            if self.X_sem.shape[0] != self.semantic_dim:
                raise ValueError(
                    f"X_sem dimension mismatch: expected {self.semantic_dim}, "
                    f"got {self.X_sem.shape[0]}"
                )
        else:
            self.X_sem = torch.tensor(self.X_sem, dtype=torch.float32)

        # 初始化逻辑物理流
        if self.X_log is None:
            self.X_log = torch.zeros(self.physical_dim)
            logger.debug(f"Initialized X_log with zeros: shape {self.X_log.shape}")
        elif isinstance(self.X_log, torch.Tensor):
            if self.X_log.shape[0] != self.physical_dim:
                raise ValueError(
                    f"X_log dimension mismatch: expected {self.physical_dim}, "
                    f"got {self.X_log.shape[0]}"
                )
        else:
            self.X_log = torch.tensor(self.X_log, dtype=torch.float32)

        # 初始化本体感觉流
        if self.X_proprio is None:
            self.X_proprio = torch.zeros(self.proprio_dim)
            logger.debug(
                f"Initialized X_proprio with zeros: shape {self.X_proprio.shape}"
            )
        elif isinstance(self.X_proprio, torch.Tensor):
            if self.X_proprio.shape[0] != self.proprio_dim:
                raise ValueError(
                    f"X_proprio dimension mismatch: expected {self.proprio_dim}, "
                    f"got {self.X_proprio.shape[0]}"
                )
        else:
            self.X_proprio = torch.tensor(self.X_proprio, dtype=torch.float32)

        # 初始化外部世界流
        if self.X_world is None:
            self.X_world = torch.zeros(self.world_dim)
            logger.debug(f"Initialized X_world with zeros: shape {self.X_world.shape}")
        elif isinstance(self.X_world, torch.Tensor):
            if self.X_world.shape[0] != self.world_dim:
                raise ValueError(
                    f"X_world dimension mismatch: expected {self.world_dim}, "
                    f"got {self.X_world.shape[0]}"
                )
        else:
            self.X_world = torch.tensor(self.X_world, dtype=torch.float32)

        # 确保张量是连续的
        self.X_sem = self.X_sem.contiguous()
        self.X_log = self.X_log.contiguous()
        self.X_proprio = self.X_proprio.contiguous()
        self.X_world = self.X_world.contiguous()

        # 验证重要性和情感强度范围
        if not 0.0 <= self.importance <= 1.0:
            logger.warning(
                f"Importance {self.importance} out of range [0, 1], clamping"
            )
            self.importance = max(0.0, min(1.0, self.importance))

        if not 0.0 <= self.emotional_intensity <= 1.0:
            logger.warning(
                f"Emotional intensity {self.emotional_intensity} out of range [0, 1], clamping"
            )
            self.emotional_intensity = max(0.0, min(1.0, self.emotional_intensity))

    def to_dict(self) -> Dict[str, Any]:
        """
        将输入序列化为字典

        Returns:
            包含所有输入信息的字典
        """
        return {
            "X_sem": self.X_sem.detach().cpu().numpy().tolist(),
            "X_log": self.X_log.detach().cpu().numpy().tolist(),
            "X_proprio": self.X_proprio.detach().cpu().numpy().tolist(),
            "X_world": self.X_world.detach().cpu().numpy().tolist(),
            "timestamp": self.timestamp,
            "source": self.source.value,
            "metadata": self.metadata.copy(),
            "importance": self.importance,
            "emotional_intensity": self.emotional_intensity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExternalInput":
        """
        从字典反序列化输入

        Args:
            data: 包含输入信息的字典

        Returns:
            ExternalInput 实例
        """
        X_sem = torch.tensor(data["X_sem"], dtype=torch.float32)
        X_log = torch.tensor(data["X_log"], dtype=torch.float32)
        X_proprio = torch.tensor(
            data.get("X_proprio", [0.0] * 256), dtype=torch.float32
        )
        X_world = torch.tensor(data.get("X_world", [0.0] * 256), dtype=torch.float32)

        source_str = data.get("source", "unknown")
        try:
            source = InputSource(source_str)
        except ValueError:
            source = InputSource.UNKNOWN

        return cls(
            X_sem=X_sem,
            X_log=X_log,
            X_proprio=X_proprio,
            X_world=X_world,
            timestamp=data.get("timestamp", 0.0),
            source=source,
            metadata=data.get("metadata", {}),
            importance=data.get("importance", 1.0),
            emotional_intensity=data.get("emotional_intensity", 0.0),
        )

    def get_combined_physical_flow(self) -> torch.Tensor:
        """
        获取组合的物理流（本体感觉 + 外部世界）

        Returns:
            组合的物理流张量
        """
        # 简单拼接，也可以考虑其他融合方式
        return torch.cat([self.X_proprio, self.X_world], dim=0)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证输入的有效性

        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []

        # 检查张量维度
        if self.X_sem.shape[0] != self.semantic_dim:
            errors.append(
                f"X_sem dimension mismatch: expected {self.semantic_dim}, "
                f"got {self.X_sem.shape[0]}"
            )

        if self.X_log.shape[0] != self.physical_dim:
            errors.append(
                f"X_log dimension mismatch: expected {self.physical_dim}, "
                f"got {self.X_log.shape[0]}"
            )

        if self.X_proprio.shape[0] != self.proprio_dim:
            errors.append(
                f"X_proprio dimension mismatch: expected {self.proprio_dim}, "
                f"got {self.X_proprio.shape[0]}"
            )

        if self.X_world.shape[0] != self.world_dim:
            errors.append(
                f"X_world dimension mismatch: expected {self.world_dim}, "
                f"got {self.X_world.shape[0]}"
            )

        # 检查数值稳定性（NaN 和 Inf）
        if torch.isnan(self.X_sem).any():
            errors.append("X_sem contains NaN values")
        if torch.isinf(self.X_sem).any():
            errors.append("X_sem contains Inf values")

        if torch.isnan(self.X_log).any():
            errors.append("X_log contains NaN values")
        if torch.isinf(self.X_log).any():
            errors.append("X_log contains Inf values")

        if torch.isnan(self.X_proprio).any():
            errors.append("X_proprio contains NaN values")
        if torch.isinf(self.X_proprio).any():
            errors.append("X_proprio contains Inf values")

        if torch.isnan(self.X_world).any():
            errors.append("X_world contains NaN values")
        if torch.isinf(self.X_world).any():
            errors.append("X_world contains Inf values")

        # 检查时间戳
        if self.timestamp < 0:
            errors.append(f"Invalid timestamp: {self.timestamp} (should be >= 0)")

        # 检查重要性范围
        if not 0.0 <= self.importance <= 1.0:
            errors.append(
                f"Importance {self.importance} out of range [0, 1]"
            )

        # 检查情感强度范围
        if not 0.0 <= self.emotional_intensity <= 1.0:
            errors.append(
                f"Emotional intensity {self.emotional_intensity} out of range [0, 1]"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    def is_high_emotional(self, threshold: float = 0.7) -> bool:
        """
        判断是否为高情感强度输入（用于关键帧标记）

        Args:
            threshold: 情感强度阈值

        Returns:
            是否为高情感强度
        """
        return self.emotional_intensity >= threshold

    def get_semantic_norm(self) -> float:
        """获取语义流的 L2 范数"""
        return torch.norm(self.X_sem).item()

    def get_physical_norm(self) -> float:
        """获取物理流的 L2 范数"""
        return torch.norm(self.X_log).item()

    def get_proprio_norm(self) -> float:
        """获取本体感觉流的 L2 范数"""
        return torch.norm(self.X_proprio).item()

    def get_world_norm(self) -> float:
        """获取外部世界流的 L2 范数"""
        return torch.norm(self.X_world).item()

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"ExternalInput(timestamp={self.timestamp:.2f}, "
            f"source={self.source.value}, "
            f"X_sem_norm={self.get_semantic_norm():.4f}, "
            f"X_log_norm={self.get_physical_norm():.4f}, "
            f"importance={self.importance:.2f}, "
            f"emotional_intensity={self.emotional_intensity:.2f})"
        )