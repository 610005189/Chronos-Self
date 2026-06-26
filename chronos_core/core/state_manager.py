"""
StateManager 类定义 - 状态管理器
负责状态的初始化、更新、保存、加载和验证
"""

import torch
import json
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

from .state import SelfState

logger = logging.getLogger(__name__)


class StateManager:
    """
    状态管理器

    负责管理 SelfState 实例的完整生命周期，包括：
    - 初始化（随机初始化或指定初始值）
    - 更新（更新快变量和慢变量）
    - 持久化（保存和加载）
    - 验证（维度和数值稳定性检查）
    - 多状态实例管理
    """

    def __init__(
        self,
        default_device: Optional[torch.device] = None,
        max_history_length: int = 10000,
    ):
        """
        初始化状态管理器

        Args:
            default_device: 默认计算设备，如果为 None 则自动选择
            max_history_length: 历史记录最大长度
        """
        self.default_device = default_device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.max_history_length = max_history_length

        # 状态实例注册表：state_id -> SelfState
        self._states: Dict[str, SelfState] = {}

        # 活跃状态 ID
        self._active_state_id: Optional[str] = None

        logger.info(
            f"StateManager initialized with device: {self.default_device}, "
            f"max_history_length: {max_history_length}"
        )

    def initialize_state(
        self,
        state_id: str,
        init_method: str = "random",
        E_fast_init: Optional[torch.Tensor] = None,
        E_slow_init: Optional[torch.Tensor] = None,
        timestamp: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelfState:
        """
        初始化一个新状态

        Args:
            state_id: 状态实例的唯一标识符
            init_method: 初始化方法，可选 "random", "zeros", "custom"
            E_fast_init: 自定义快变量初始值（仅当 init_method="custom" 时使用）
            E_slow_init: 自定义慢变量初始值（仅当 init_method="custom" 时使用）
            timestamp: 初始时间戳
            metadata: 元数据字典

        Returns:
            初始化后的 SelfState 实例

        Raises:
            ValueError: 如果参数无效
        """
        logger.info(
            f"Initializing state '{state_id}' with method: {init_method}, timestamp: {timestamp}"
        )

        # 检查 state_id 是否已存在
        if state_id in self._states:
            raise ValueError(f"State with id '{state_id}' already exists")

        # 根据初始化方法创建状态
        if init_method == "random":
            # 使用正态分布随机初始化
            E_fast = torch.randn(SelfState.FAST_DIM, device=self.default_device)
            E_slow = torch.randn(SelfState.SLOW_DIM, device=self.default_device)

            # 归一化到单位球面
            E_fast = E_fast / torch.norm(E_fast)
            E_slow = E_slow / torch.norm(E_slow)

            logger.debug(
                f"Random initialization: E_fast_norm={torch.norm(E_fast):.4f}, "
                f"E_slow_norm={torch.norm(E_slow):.4f}"
            )

        elif init_method == "zeros":
            # 零初始化
            E_fast = torch.zeros(SelfState.FAST_DIM, device=self.default_device)
            E_slow = torch.zeros(SelfState.SLOW_DIM, device=self.default_device)
            logger.debug("Zero initialization")

        elif init_method == "custom":
            # 自定义初始化
            if E_fast_init is None or E_slow_init is None:
                raise ValueError(
                    "E_fast_init and E_slow_init must be provided for custom initialization"
                )

            E_fast = E_fast_init.clone().detach().to(self.default_device)
            E_slow = E_slow_init.clone().detach().to(self.default_device)

            # 验证维度
            if E_fast.shape[0] != SelfState.FAST_DIM:
                raise ValueError(
                    f"E_fast_init dimension mismatch: expected {SelfState.FAST_DIM}, "
                    f"got {E_fast.shape[0]}"
                )
            if E_slow.shape[0] != SelfState.SLOW_DIM:
                raise ValueError(
                    f"E_slow_init dimension mismatch: expected {SelfState.SLOW_DIM}, "
                    f"got {E_slow.shape[0]}"
                )

            logger.debug(
                f"Custom initialization: E_fast_norm={torch.norm(E_fast):.4f}, "
                f"E_slow_norm={torch.norm(E_slow):.4f}"
            )

        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

        # 创建状态实例
        state = SelfState(
            E_fast=E_fast,
            E_slow=E_slow,
            timestamp=timestamp,
            metadata=metadata or {},
        )

        # 记录初始历史快照
        state.record_history(self.max_history_length)

        # 注册状态
        self._states[state_id] = state

        # 如果是第一个状态，设置为活跃状态
        if self._active_state_id is None:
            self._active_state_id = state_id

        logger.info(f"State '{state_id}' initialized successfully: {state}")

        return state

    def update_state(
        self,
        state_id: str,
        delta_E_fast: torch.Tensor,
        delta_E_slow: torch.Tensor,
        dt: float,
        record_history: bool = True,
    ) -> SelfState:
        """
        更新状态

        Args:
            state_id: 状态实例 ID
            delta_E_fast: 快变量增量
            delta_E_slow: 慢变量增量
            dt: 时间步长（秒）
            record_history: 是否记录到历史

        Returns:
            更新后的 SelfState 实例

        Raises:
            ValueError: 如果状态不存在或参数无效
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        state = self._states[state_id]

        logger.debug(
            f"Updating state '{state_id}': dt={dt:.6f}s, "
            f"delta_E_fast_norm={torch.norm(delta_E_fast):.6f}, "
            f"delta_E_slow_norm={torch.norm(delta_E_slow):.6f}"
        )

        # 验证增量维度
        if delta_E_fast.shape[0] != SelfState.FAST_DIM:
            raise ValueError(
                f"delta_E_fast dimension mismatch: expected {SelfState.FAST_DIM}, "
                f"got {delta_E_fast.shape[0]}"
            )
        if delta_E_slow.shape[0] != SelfState.SLOW_DIM:
            raise ValueError(
                f"delta_E_slow dimension mismatch: expected {SelfState.SLOW_DIM}, "
                f"got {delta_E_slow.shape[0]}"
            )

        # 更新状态
        state.E_fast = state.E_fast + delta_E_fast.to(state.E_fast.device)
        state.E_slow = state.E_slow + delta_E_slow.to(state.E_slow.device)
        state.timestamp += dt

        # 数值稳定性检查和裁剪
        state = self._apply_stability_constraints(state)

        # 记录历史
        if record_history:
            state.record_history(self.max_history_length)

        logger.debug(
            f"State updated: timestamp={state.timestamp:.2f}, "
            f"E_fast_norm={state.get_fast_norm():.6f}, "
            f"E_slow_norm={state.get_slow_norm():.6f}"
        )

        return state

    def _apply_stability_constraints(self, state: SelfState) -> SelfState:
        """
        应用数值稳定性约束

        Args:
            state: 要约束的状态

        Returns:
            约束后的状态
        """
        # 检查并替换 NaN 和 Inf
        if torch.isnan(state.E_fast).any() or torch.isinf(state.E_fast).any():
            logger.warning("Detected NaN/Inf in E_fast, replacing with zeros")
            state.E_fast = torch.where(
                torch.isnan(state.E_fast) | torch.isinf(state.E_fast),
                torch.zeros_like(state.E_fast),
                state.E_fast,
            )

        if torch.isnan(state.E_slow).any() or torch.isinf(state.E_slow).any():
            logger.warning("Detected NaN/Inf in E_slow, replacing with zeros")
            state.E_slow = torch.where(
                torch.isnan(state.E_slow) | torch.isinf(state.E_slow),
                torch.zeros_like(state.E_slow),
                state.E_slow,
            )

        # 可选：应用 L2 范数裁剪，防止数值过大
        max_norm = 1000.0
        E_fast_norm = torch.norm(state.E_fast)
        if E_fast_norm > max_norm:
            state.E_fast = state.E_fast * (max_norm / E_fast_norm)
            logger.debug(f"Clipped E_fast norm from {E_fast_norm:.4f} to {max_norm}")

        E_slow_norm = torch.norm(state.E_slow)
        if E_slow_norm > max_norm:
            state.E_slow = state.E_slow * (max_norm / E_slow_norm)
            logger.debug(f"Clipped E_slow norm from {E_slow_norm:.4f} to {max_norm}")

        return state

    def validate_state(self, state_id: str) -> Tuple[bool, List[str]]:
        """
        验证状态的有效性

        Args:
            state_id: 状态实例 ID

        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表

        Raises:
            ValueError: 如果状态不存在
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        state = self._states[state_id]
        is_valid, errors = state.validate()

        if not is_valid:
            logger.warning(f"State validation failed for '{state_id}': {errors}")
        else:
            logger.debug(f"State '{state_id}' validation passed")

        return is_valid, errors

    def get_state(self, state_id: str) -> SelfState:
        """
        获取状态实例

        Args:
            state_id: 状态实例 ID

        Returns:
            SelfState 实例

        Raises:
            ValueError: 如果状态不存在
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        return self._states[state_id]

    def get_active_state(self) -> Optional[SelfState]:
        """
        获取当前活跃状态

        Returns:
            活跃的 SelfState 实例，如果没有则返回 None
        """
        if self._active_state_id is None:
            return None

        return self._states.get(self._active_state_id)

    def set_active_state(self, state_id: str):
        """
        设置活跃状态

        Args:
            state_id: 要设置为活跃状态的 ID

        Raises:
            ValueError: 如果状态不存在
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        self._active_state_id = state_id
        logger.info(f"Active state set to '{state_id}'")

    def list_states(self) -> List[str]:
        """
        列出所有状态 ID

        Returns:
            状态 ID 列表
        """
        return list(self._states.keys())

    def remove_state(self, state_id: str):
        """
        移除状态实例

        Args:
            state_id: 要移除的状态 ID

        Raises:
            ValueError: 如果状态不存在
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        del self._states[state_id]

        # 如果移除的是活跃状态，清除活跃状态标记
        if self._active_state_id == state_id:
            self._active_state_id = None
            logger.info(f"Active state '{state_id}' removed, no active state now")
        else:
            logger.info(f"State '{state_id}' removed")

    def save_state(
        self,
        state_id: str,
        filepath: str,
        format: str = "json",
        include_history: bool = True,
    ):
        """
        保存状态到文件

        Args:
            state_id: 状态实例 ID
            filepath: 保存文件路径
            format: 保存格式，可选 "json" 或 "pt" (PyTorch)
            include_history: 是否包含历史记录

        Raises:
            ValueError: 如果状态不存在或参数无效
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        state = self._states[state_id]

        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # JSON 格式保存
            data = state.to_dict()

            # 可选：排除历史记录以减小文件大小
            if not include_history:
                data["history"] = []

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            logger.info(
                f"State '{state_id}' saved to {filepath} (format: json, "
                f"include_history: {include_history})"
            )

        elif format == "pt":
            # PyTorch 格式保存
            data = {
                "E_fast": state.E_fast.cpu(),
                "E_slow": state.E_slow.cpu(),
                "timestamp": state.timestamp,
                "history": state.history if include_history else [],
                "metadata": state.metadata,
            }

            torch.save(data, filepath)

            logger.info(
                f"State '{state_id}' saved to {filepath} (format: pt, "
                f"include_history: {include_history})"
            )

        else:
            raise ValueError(f"Unknown format: {format}")

    def load_state(
        self,
        state_id: str,
        filepath: str,
        format: str = "json",
        set_as_active: bool = False,
    ) -> SelfState:
        """
        从文件加载状态

        Args:
            state_id: 新状态实例的 ID
            filepath: 文件路径
            format: 文件格式，可选 "json" 或 "pt" (PyTorch)
            set_as_active: 是否设置为活跃状态

        Returns:
            加载的 SelfState 实例

        Raises:
            ValueError: 如果状态 ID 已存在或参数无效
            FileNotFoundError: 如果文件不存在
        """
        if state_id in self._states:
            raise ValueError(f"State with id '{state_id}' already exists")

        if format == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            state = SelfState.from_dict(data)

        elif format == "pt":
            data = torch.load(filepath, map_location="cpu")

            state = SelfState(
                E_fast=data["E_fast"],
                E_slow=data["E_slow"],
                timestamp=data["timestamp"],
                history=data["history"],
                metadata=data.get("metadata", {}),
            )

        else:
            raise ValueError(f"Unknown format: {format}")

        # 注册状态
        self._states[state_id] = state

        if set_as_active or self._active_state_id is None:
            self._active_state_id = state_id

        logger.info(f"State loaded from {filepath} with id '{state_id}': {state}")

        return state

    def save_all_states(self, directory: str, format: str = "json"):
        """
        保存所有状态到目录

        Args:
            directory: 保存目录
            format: 文件格式
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        for state_id, state in self._states.items():
            # 使用 state_id 作为文件名
            filename = f"{state_id}.{format}"
            filepath = os.path.join(directory, filename)
            self.save_state(state_id, filepath, format=format)

        logger.info(
            f"Saved {len(self._states)} states to directory: {directory} (format: {format})"
        )

    def get_state_statistics(self, state_id: str) -> Dict[str, Any]:
        """
        获取状态的统计信息

        Args:
            state_id: 状态实例 ID

        Returns:
            包含统计信息的字典

        Raises:
            ValueError: 如果状态不存在
        """
        if state_id not in self._states:
            raise ValueError(f"State '{state_id}' not found")

        state = self._states[state_id]

        return {
            "state_id": state_id,
            "timestamp": state.timestamp,
            "E_fast_norm": state.get_fast_norm(),
            "E_slow_norm": state.get_slow_norm(),
            "E_fast_mean": state.E_fast.mean().item(),
            "E_slow_mean": state.E_slow.mean().item(),
            "E_fast_std": state.E_fast.std().item(),
            "E_slow_std": state.E_slow.std().item(),
            "history_length": len(state.history),
            "is_active": state_id == self._active_state_id,
        }

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"StateManager(num_states={len(self._states)}, "
            f"active_state='{self._active_state_id}', "
            f"device={self.default_device})"
        )