"""
演化历史记录系统
记录状态演化、外部输入和系统响应的完整历史
"""

import torch
import json
import csv
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from datetime import datetime
import logging

from .state import SelfState
from .external_input import ExternalInput

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""

    STATE_UPDATE = "state_update"  # 状态更新
    INPUT_RECEIVED = "input_received"  # 接收外部输入
    SYSTEM_RESPONSE = "system_response"  # 系统响应
    KEYFRAME = "keyframe"  # 关键帧（高情感强度时刻）
    SLEEP_START = "sleep_start"  # 睡眠期开始
    SLEEP_END = "sleep_end"  # 睡眠期结束
    REFLECTION = "reflection"  # 反思事件
    ERROR = "error"  # 错误事件


class SnapshotType(Enum):
    """快照类型枚举"""

    FULL = "full"  # 完整快照（包含完整状态向量）
    COMPRESSED = "compressed"  # 压缩快照（仅包含范数和统计信息）
    MINIMAL = "minimal"  # 最小快照（仅包含时间戳和关键指标）


@dataclass
class HistoryEntry:
    """
    历史记录条目

    Attributes:
        timestamp: 时间戳（秒）
        event_type: 事件类型
        state_snapshot: 状态快照（完整或压缩）
        input_data: 外部输入数据（如果是输入事件）
        response_data: 系统响应数据（如果是响应事件）
        metadata: 额外元数据
        is_keyframe: 是否为关键帧
    """

    timestamp: float
    event_type: EventType
    state_snapshot: Optional[Dict[str, Any]] = None
    input_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_keyframe: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "state_snapshot": self.state_snapshot,
            "input_data": self.input_data,
            "response_data": self.response_data,
            "metadata": self.metadata,
            "is_keyframe": self.is_keyframe,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryEntry":
        """从字典反序列化"""
        return cls(
            timestamp=data["timestamp"],
            event_type=EventType(data["event_type"]),
            state_snapshot=data.get("state_snapshot"),
            input_data=data.get("input_data"),
            response_data=data.get("response_data"),
            metadata=data.get("metadata", {}),
            is_keyframe=data.get("is_keyframe", False),
        )


class EvolutionHistory:
    """
    演化历史记录系统

    管理完整的演化历史，包括状态快照、输入响应和关键帧标记
    """

    def __init__(
        self,
        max_entries: int = 100000,
        snapshot_type: SnapshotType = SnapshotType.COMPRESSED,
        auto_keyframe_threshold: float = 0.7,
        compression_interval: int = 1000,
    ):
        """
        初始化历史记录系统

        Args:
            max_entries: 最大历史条目数
            snapshot_type: 默认快照类型
            auto_keyframe_threshold: 自动标记关键帧的情感强度阈值
            compression_interval: 压缩间隔（每隔多少条完整快照压缩为压缩快照）
        """
        self.max_entries = max_entries
        self.snapshot_type = snapshot_type
        self.auto_keyframe_threshold = auto_keyframe_threshold
        self.compression_interval = compression_interval

        # 历史记录列表
        self._entries: List[HistoryEntry] = []

        # 关键帧索引（时间戳列表）
        self._keyframes: List[float] = []

        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_entries": 0,
            "keyframes_count": 0,
            "events_by_type": {},
            "start_time": None,
            "end_time": None,
        }

        logger.info(
            f"EvolutionHistory initialized: max_entries={max_entries}, "
            f"snapshot_type={snapshot_type.value}, "
            f"auto_keyframe_threshold={auto_keyframe_threshold}"
        )

    def record_state(
        self,
        state: SelfState,
        event_type: EventType = EventType.STATE_UPDATE,
        snapshot_type: Optional[SnapshotType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        """
        记录状态快照

        Args:
            state: 要记录的状态
            event_type: 事件类型
            snapshot_type: 快照类型（如果为 None，使用默认类型）
            metadata: 额外元数据

        Returns:
            创建的历史条目
        """
        snapshot_type = snapshot_type or self.snapshot_type

        # 根据快照类型创建快照
        if snapshot_type == SnapshotType.FULL:
            state_snapshot = state.to_dict()
        elif snapshot_type == SnapshotType.COMPRESSED:
            state_snapshot = {
                "timestamp": state.timestamp,
                "E_fast_norm": state.get_fast_norm(),
                "E_slow_norm": state.get_slow_norm(),
                "E_fast_mean": state.E_fast.mean().item(),
                "E_slow_mean": state.E_slow.mean().item(),
                "E_fast_std": state.E_fast.std().item(),
                "E_slow_std": state.E_slow.std().item(),
                "history_length": len(state.history),
            }
        else:  # MINIMAL
            state_snapshot = {
                "timestamp": state.timestamp,
                "E_fast_norm": state.get_fast_norm(),
                "E_slow_norm": state.get_slow_norm(),
            }

        # 创建历史条目
        entry = HistoryEntry(
            timestamp=state.timestamp,
            event_type=event_type,
            state_snapshot=state_snapshot,
            metadata=metadata or {},
            is_keyframe=False,
        )

        # 添加到历史
        self._add_entry(entry)

        logger.debug(
            f"Recorded state snapshot at timestamp={state.timestamp:.2f}, "
            f"type={snapshot_type.value}"
        )

        return entry

    def record_input(
        self,
        input_data: ExternalInput,
        state: Optional[SelfState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        """
        记录外部输入

        Args:
            input_data: 外部输入数据
            state: 当前状态（可选）
            metadata: 额外元数据

        Returns:
            创建的历史条目
        """
        # 判断是否为关键帧
        is_keyframe = input_data.is_high_emotional(self.auto_keyframe_threshold)

        # 创建输入数据快照
        input_snapshot = {
            "timestamp": input_data.timestamp,
            "source": input_data.source.value,
            "importance": input_data.importance,
            "emotional_intensity": input_data.emotional_intensity,
            "X_sem_norm": input_data.get_semantic_norm(),
            "X_log_norm": input_data.get_physical_norm(),
            "X_proprio_norm": input_data.get_proprio_norm(),
            "X_world_norm": input_data.get_world_norm(),
        }

        # 可选：记录完整输入数据
        if self.snapshot_type == SnapshotType.FULL:
            input_snapshot["full_data"] = input_data.to_dict()

        # 创建状态快照（如果提供）
        state_snapshot = None
        if state is not None:
            if self.snapshot_type == SnapshotType.FULL:
                state_snapshot = state.to_dict()
            else:
                state_snapshot = {
                    "timestamp": state.timestamp,
                    "E_fast_norm": state.get_fast_norm(),
                    "E_slow_norm": state.get_slow_norm(),
                }

        # 创建历史条目
        entry = HistoryEntry(
            timestamp=input_data.timestamp,
            event_type=EventType.INPUT_RECEIVED,
            state_snapshot=state_snapshot,
            input_data=input_snapshot,
            metadata=metadata or {},
            is_keyframe=is_keyframe,
        )

        # 添加到历史
        self._add_entry(entry)

        # 如果是关键帧，添加到关键帧索引
        if is_keyframe:
            self._keyframes.append(input_data.timestamp)
            logger.info(
                f"Marked keyframe at timestamp={input_data.timestamp:.2f}, "
                f"emotional_intensity={input_data.emotional_intensity:.2f}"
            )

        logger.debug(
            f"Recorded input at timestamp={input_data.timestamp:.2f}, "
            f"source={input_data.source.value}, is_keyframe={is_keyframe}"
        )

        return entry

    def record_response(
        self,
        response_data: Dict[str, Any],
        timestamp: float,
        state: Optional[SelfState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> HistoryEntry:
        """
        记录系统响应

        Args:
            response_data: 响应数据字典
            timestamp: 时间戳
            state: 当前状态（可选）
            metadata: 额外元数据

        Returns:
            创建的历史条目
        """
        # 创建状态快照（如果提供）
        state_snapshot = None
        if state is not None:
            if self.snapshot_type == SnapshotType.FULL:
                state_snapshot = state.to_dict()
            else:
                state_snapshot = {
                    "timestamp": state.timestamp,
                    "E_fast_norm": state.get_fast_norm(),
                    "E_slow_norm": state.get_slow_norm(),
                }

        # 创建历史条目
        entry = HistoryEntry(
            timestamp=timestamp,
            event_type=EventType.SYSTEM_RESPONSE,
            state_snapshot=state_snapshot,
            response_data=response_data,
            metadata=metadata or {},
            is_keyframe=False,
        )

        # 添加到历史
        self._add_entry(entry)

        logger.debug(f"Recorded response at timestamp={timestamp:.2f}")

        return entry

    def mark_keyframe(
        self,
        timestamp: float,
        reason: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        手动标记关键帧

        Args:
            timestamp: 时间戳
            reason: 关键帧原因
            metadata: 额外元数据
        """
        # 创建关键帧条目
        entry = HistoryEntry(
            timestamp=timestamp,
            event_type=EventType.KEYFRAME,
            metadata={"reason": reason, **(metadata or {})},
            is_keyframe=True,
        )

        # 添加到历史
        self._add_entry(entry)

        # 添加到关键帧索引
        self._keyframes.append(timestamp)

        logger.info(f"Manually marked keyframe at timestamp={timestamp:.2f}, reason={reason}")

    def _add_entry(self, entry: HistoryEntry):
        """
        添加历史条目（内部方法）

        Args:
            entry: 要添加的条目
        """
        # 添加到历史列表
        self._entries.append(entry)

        # 更新统计信息
        self._stats["total_entries"] += 1
        if entry.is_keyframe:
            self._stats["keyframes_count"] += 1

        event_type_str = entry.event_type.value
        if event_type_str not in self._stats["events_by_type"]:
            self._stats["events_by_type"][event_type_str] = 0
        self._stats["events_by_type"][event_type_str] += 1

        # 更新时间范围
        if self._stats["start_time"] is None:
            self._stats["start_time"] = entry.timestamp
        self._stats["end_time"] = entry.timestamp

        # 限制历史长度
        if len(self._entries) > self.max_entries:
            # 移除最早的条目
            removed_entry = self._entries.pop(0)
            self._stats["total_entries"] -= 1
            if removed_entry.is_keyframe:
                self._stats["keyframes_count"] -= 1
                # 从关键帧索引中移除
                if removed_entry.timestamp in self._keyframes:
                    self._keyframes.remove(removed_entry.timestamp)

            logger.debug(
                f"Removed oldest entry to maintain max_entries={self.max_entries}"
            )

    def query_by_time_range(
        self,
        start_time: float,
        end_time: float,
        event_types: Optional[List[EventType]] = None,
    ) -> List[HistoryEntry]:
        """
        查询指定时间范围内的历史记录

        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            event_types: 事件类型过滤列表（如果为 None，返回所有类型）

        Returns:
            符合条件的历史条目列表
        """
        results = []

        for entry in self._entries:
            # 检查时间范围
            if not (start_time <= entry.timestamp <= end_time):
                continue

            # 检查事件类型
            if event_types and entry.event_type not in event_types:
                continue

            results.append(entry)

        logger.debug(
            f"Query by time range: [{start_time:.2f}, {end_time:.2f}], "
            f"found {len(results)} entries"
        )

        return results

    def query_by_event_type(
        self,
        event_type: EventType,
        limit: Optional[int] = None,
    ) -> List[HistoryEntry]:
        """
        查询指定事件类型的历史记录

        Args:
            event_type: 事件类型
            limit: 返回数量限制（如果为 None，返回所有）

        Returns:
            符合条件的历史条目列表
        """
        results = []

        for entry in self._entries:
            if entry.event_type == event_type:
                results.append(entry)

        # 应用数量限制
        if limit and len(results) > limit:
            results = results[-limit:]

        logger.debug(
            f"Query by event type: {event_type.value}, "
            f"found {len(results)} entries"
        )

        return results

    def get_keyframes(self) -> List[HistoryEntry]:
        """
        获取所有关键帧

        Returns:
            关键帧条目列表
        """
        return [entry for entry in self._entries if entry.is_keyframe]

    def get_keyframe_timestamps(self) -> List[float]:
        """
        获取关键帧时间戳列表

        Returns:
            关键帧时间戳列表
        """
        return self._keyframes.copy()

    def get_entry_at_time(self, timestamp: float) -> Optional[HistoryEntry]:
        """
        获取指定时间戳的历史条目（最近的前一个条目）

        Args:
            timestamp: 目标时间戳

        Returns:
            最近的历史条目，如果不存在则返回 None
        """
        # 二分查找最近的前一个条目
        if not self._entries:
            return None

        # 简单遍历查找（可以优化为二分查找）
        closest_entry = None
        for entry in self._entries:
            if entry.timestamp <= timestamp:
                closest_entry = entry
            else:
                break

        return closest_entry

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取历史统计信息

        Returns:
            统计信息字典
        """
        return self._stats.copy()

    def export_to_json(
        self,
        filepath: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        include_keyframes_only: bool = False,
    ):
        """
        导出历史到 JSON 文件

        Args:
            filepath: 文件路径
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            include_keyframes_only: 是否仅导出关键帧
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 过滤数据
        entries_to_export = []
        for entry in self._entries:
            # 时间范围过滤
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue

            # 关键帧过滤
            if include_keyframes_only and not entry.is_keyframe:
                continue

            entries_to_export.append(entry)

        # 序列化
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_entries": len(entries_to_export),
                "start_time": start_time,
                "end_time": end_time,
                "include_keyframes_only": include_keyframes_only,
            },
            "entries": [entry.to_dict() for entry in entries_to_export],
            "statistics": self._stats,
        }

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"Exported {len(entries_to_export)} entries to JSON: {filepath}"
        )

    def export_to_csv(
        self,
        filepath: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
    ):
        """
        导出历史到 CSV 文件

        Args:
            filepath: 文件路径
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）
            event_types: 事件类型过滤列表（可选）
        """
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # 过滤数据
        entries_to_export = self.query_by_time_range(
            start_time or 0.0, end_time or float("inf"), event_types
        )

        # CSV 列定义
        fieldnames = [
            "timestamp",
            "event_type",
            "is_keyframe",
            "E_fast_norm",
            "E_slow_norm",
            "importance",
            "emotional_intensity",
            "metadata",
        ]

        # 写入 CSV
        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for entry in entries_to_export:
                row = {
                    "timestamp": entry.timestamp,
                    "event_type": entry.event_type.value,
                    "is_keyframe": entry.is_keyframe,
                    "E_fast_norm": "",
                    "E_slow_norm": "",
                    "importance": "",
                    "emotional_intensity": "",
                    "metadata": json.dumps(entry.metadata),
                }

                # 提取状态快照数据
                if entry.state_snapshot:
                    row["E_fast_norm"] = entry.state_snapshot.get("E_fast_norm", "")
                    row["E_slow_norm"] = entry.state_snapshot.get("E_slow_norm", "")

                # 提取输入数据
                if entry.input_data:
                    row["importance"] = entry.input_data.get("importance", "")
                    row["emotional_intensity"] = entry.input_data.get(
                        "emotional_intensity", ""
                    )

                writer.writerow(row)

        logger.info(
            f"Exported {len(entries_to_export)} entries to CSV: {filepath}"
        )

    def load_from_json(self, filepath: str):
        """
        从 JSON 文件加载历史

        Args:
            filepath: 文件路径
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 清空当前历史
        self._entries.clear()
        self._keyframes.clear()
        self._stats = {
            "total_entries": 0,
            "keyframes_count": 0,
            "events_by_type": {},
            "start_time": None,
            "end_time": None,
        }

        # 加载条目
        for entry_dict in data["entries"]:
            entry = HistoryEntry.from_dict(entry_dict)
            self._add_entry(entry)

        logger.info(
            f"Loaded {len(self._entries)} entries from JSON: {filepath}"
        )

    def clear(self):
        """
        清空历史记录
        """
        self._entries.clear()
        self._keyframes.clear()
        self._stats = {
            "total_entries": 0,
            "keyframes_count": 0,
            "events_by_type": {},
            "start_time": None,
            "end_time": None,
        }

        logger.info("History cleared")

    def get_recent_entries(self, n: int = 100) -> List[HistoryEntry]:
        """
        获取最近的 n 条历史记录

        Args:
            n: 数量

        Returns:
            最近的历史条目列表
        """
        return self._entries[-n:] if len(self._entries) >= n else self._entries.copy()

    def __len__(self) -> int:
        """获取历史记录数量"""
        return len(self._entries)

    def __repr__(self) -> str:
        """字符串表示"""
        start_time_str = (
            f"{self._stats['start_time']:.2f}"
            if self._stats['start_time'] is not None
            else "N/A"
        )
        end_time_str = (
            f"{self._stats['end_time']:.2f}"
            if self._stats['end_time'] is not None
            else "N/A"
        )
        return (
            f"EvolutionHistory(entries={len(self._entries)}, "
            f"keyframes={len(self._keyframes)}, "
            f"start_time={start_time_str}, "
            f"end_time={end_time_str})"
        )