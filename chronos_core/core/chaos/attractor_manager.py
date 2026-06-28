"""
混沌吸引子管理器
================

管理和切换多个混沌吸引子，支持平滑过渡和状态管理。
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import logging
import random

from .base_attractor import BaseAttractor, AttractorState
from .lorenz_attractor import LorenzAttractor
from .rossler_attractor import RosslerAttractor
from .chua_attractor import ChuaAttractor

logger = logging.getLogger(__name__)


@dataclass
class SwitchRecord:
    """切换历史记录"""
    step: int
    from_attractor: str
    to_attractor: str
    transition_duration: int
    timestamp: float


class AttractorManager:
    """
    混沌吸引子管理器

    管理多个混沌吸引子，实现：
    - 吸引子注册和选择
    - 多吸引子随机切换
    - 切换时的平滑插值过渡
    - 状态持久化和恢复

    Attributes:
        attractors: 注册的吸引子字典
        current_index: 当前激活的吸引子索引
        switch_interval: 切换间隔（步数）
        transition_steps: 过渡步数
        smoothing_factor: 平滑因子
    """

    def __init__(
        self,
        core_subspace_dim: int = 64,
        switch_interval: int = 5000,
        transition_steps: int = 100,
        smoothing_factor: float = 0.95,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化吸引子管理器

        Args:
            core_subspace_dim: 核心子空间维度
            switch_interval: 吸引子切换间隔（步数）
            transition_steps: 过渡期间的步数
            smoothing_factor: 过渡平滑因子 (0-1)
            device: 计算设备
            seed: 随机种子
        """
        self.core_subspace_dim = core_subspace_dim
        self.switch_interval = switch_interval
        self.transition_steps = transition_steps
        self.smoothing_factor = smoothing_factor
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 随机种子
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 注册的吸引子
        self.attractors: Dict[str, BaseAttractor] = {}
        self.attractor_names: List[str] = []

        # 当前状态
        self.current_index: int = 0
        self.current_state: Optional[AttractorState] = None
        self.previous_state: Optional[AttractorState] = None

        # 切换状态
        self.step_counter: int = 0
        self.is_transitioning: bool = False
        self.transition_progress: float = 0.0

        # 切换历史
        self.switch_history: List[SwitchRecord] = []

        logger.debug(
            f"AttractorManager initialized: switch_interval={switch_interval}, "
            f"transition_steps={transition_steps}, smoothing={smoothing_factor}"
        )

    def register_attractor(self, attractor: BaseAttractor) -> None:
        """
        注册一个吸引子

        Args:
            attractor: 要注册的吸引子实例
        """
        self.attractors[attractor.name] = attractor
        self.attractor_names.append(attractor.name)
        logger.debug(f"Registered attractor: {attractor.name}")

    def register_default_attractors(
        self,
        lorenz_params: Optional[Dict] = None,
        rossler_params: Optional[Dict] = None,
        chua_params: Optional[Dict] = None
    ) -> None:
        """
        注册默认的三个混沌吸引子

        Args:
            lorenz_params: Lorenz 吸引子参数覆盖
            rossler_params: Rossler 吸引子参数覆盖
            chua_params: Chua 吸引子参数覆盖
        """
        # 默认参数
        default_lorenz = {"sigma": 10.0, "rho": 28.0, "beta": 8.0/3.0, "device": self.device}
        default_rossler = {"a": 0.2, "b": 0.2, "c": 5.7, "device": self.device}
        default_chua = {"alpha": 15.35, "beta": 28.0, "m0": -1.143, "m1": -0.714, "device": self.device}

        # 应用参数覆盖
        if lorenz_params:
            default_lorenz.update(lorenz_params)
        if rossler_params:
            default_rossler.update(rossler_params)
        if chua_params:
            default_chua.update(chua_params)

        # 创建并注册
        self.register_attractor(LorenzAttractor(**default_lorenz))
        self.register_attractor(RosslerAttractor(**default_rossler))
        self.register_attractor(ChuaAttractor(**default_chua))

        logger.debug("Registered default attractors: Lorenz, Rossler, Chua")

    def get_current_attractor(self) -> BaseAttractor:
        """
        获取当前激活的吸引子

        Returns:
            当前吸引子实例
        """
        if len(self.attractor_names) == 0:
            raise ValueError("No attractors registered")

        name = self.attractor_names[self.current_index]
        return self.attractors[name]

    def get_attractor_by_name(self, name: str) -> Optional[BaseAttractor]:
        """
        通过名称获取吸引子

        Args:
            name: 吸引子名称

        Returns:
            吸引子实例（如果存在）
        """
        return self.attractors.get(name)

    def initialize_states(self, seed: Optional[int] = None) -> None:
        """
        初始化所有吸引子的状态

        Args:
            seed: 随机种子
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 初始化当前状态
        current_attractor = self.get_current_attractor()
        z0 = current_attractor.generate_initial_state(seed=seed)

        # 预热积分（让系统进入吸引子）
        warmup_result = current_attractor.integrate(
            z0=z0,
            t_span=(0.0, 10.0),
            n_steps=1000,
            method="rk4"
        )

        self.current_state = warmup_result

        # 初始化下一个吸引子的状态（用于过渡）
        next_index = (self.current_index + 1) % len(self.attractor_names)
        next_attractor = self.attractors[self.attractor_names[next_index]]

        # 使用相似的初始状态（有助于平滑过渡）
        z0_next = next_attractor.generate_initial_state(
            seed=seed + 1 if seed is not None else None
        )

        warmup_next = next_attractor.integrate(
            z0=z0_next,
            t_span=(0.0, 10.0),
            n_steps=1000,
            method="rk4"
        )

        self.previous_state = warmup_next

        logger.debug(
            f"Initialized attractor states: current={current_attractor.name}, "
            f"state_norm={torch.norm(self.current_state.z).item():.4f}"
        )

    def step(self, dt: Optional[float] = None) -> torch.Tensor:
        """
        执行单步演化

        包括：
        - 当前吸引子的积分步
        - 切换检查
        - 过渡处理（如果正在过渡）

        Args:
            dt: 积分步长

        Returns:
            演化后的状态向量 (3,)
        """
        dt = dt or 0.01

        # 检查是否需要开始切换
        if self.step_counter >= self.switch_interval and not self.is_transitioning:
            self._start_transition()

        # 执行演化
        if self.is_transitioning:
            z = self._transition_step(dt)
        else:
            z = self._normal_step(dt)

        self.step_counter += 1

        return z

    def _normal_step(self, dt: float) -> torch.Tensor:
        """正常单步演化"""
        attractor = self.get_current_attractor()
        z_new = attractor.step(self.current_state.z, dt)

        # 更新状态
        self.current_state.z = z_new
        self.current_state.t += dt

        return z_new

    def _transition_step(self, dt: float) -> torch.Tensor:
        """过渡期间的演化"""
        # 计算过渡权重
        self.transition_progress += dt / (self.transition_steps * dt)
        alpha = self.transition_progress

        # 平滑权重（使用 sigmoid 或线性插值）
        smooth_alpha = self._compute_smooth_weight(alpha)

        # 两个吸引子各自演化
        current_attractor = self.get_current_attractor()
        z_current = current_attractor.step(self.current_state.z, dt)

        next_index = (self.current_index + 1) % len(self.attractor_names)
        next_attractor = self.attractors[self.attractor_names[next_index]]
        z_next = next_attractor.step(self.previous_state.z, dt)

        # 加权混合
        z_mixed = smooth_alpha * z_next + (1 - smooth_alpha) * z_current

        # 更新状态
        self.current_state.z = z_current
        self.previous_state.z = z_next

        # 检查过渡完成
        if self.transition_progress >= 1.0:
            self._complete_transition()

        return z_mixed

    def _compute_smooth_weight(self, alpha: float) -> float:
        """
        计算平滑过渡权重

        使用 Hermite 插值或 sigmoid 函数实现平滑过渡。

        Args:
            alpha: 线性进度 (0-1)

        Returns:
            平滑后的权重
        """
        # Hermite 插值 (smoothstep)
        # h(t) = 3t^2 - 2t^3
        if alpha <= 0:
            return 0.0
        elif alpha >= 1:
            return 1.0
        else:
            return 3 * alpha**2 - 2 * alpha**3

    def _start_transition(self) -> None:
        """开始吸引子切换"""
        self.is_transitioning = True
        self.transition_progress = 0.0
        self.step_counter = 0  # 重置计数器

        from_name = self.attractor_names[self.current_index]
        to_index = self._select_next_attractor()
        to_name = self.attractor_names[to_index]

        # 准备下一个吸引子的初始状态
        next_attractor = self.attractors[to_name]
        if self.previous_state is None:
            z0_next = next_attractor.generate_initial_state()
            warmup = next_attractor.integrate(
                z0=z0_next,
                t_span=(0.0, 5.0),
                n_steps=500,
                method="rk4"
            )
            self.previous_state = warmup

        logger.info(
            f"Starting transition: {from_name} -> {to_name}, "
            f"duration={self.transition_steps} steps"
        )

    def _select_next_attractor(self) -> int:
        """
        选择下一个吸引子

        支持随机选择和序列切换。

        Returns:
            下一个吸引子的索引
        """
        # 默认使用随机选择
        available_indices = list(range(len(self.attractor_names)))
        available_indices.remove(self.current_index)  # 排除当前

        next_index = random.choice(available_indices)
        return next_index

    def _complete_transition(self) -> None:
        """完成吸引子切换"""
        # 更换当前吸引子
        old_index = self.current_index
        self.current_index = (self.current_index + 1) % len(self.attractor_names)

        # 更新状态
        self.current_state = self.previous_state
        self.previous_state = None

        # 重置过渡状态
        self.is_transitioning = False
        self.transition_progress = 0.0

        # 记录切换历史
        record = SwitchRecord(
            step=self.step_counter,
            from_attractor=self.attractor_names[old_index],
            to_attractor=self.attractor_names[self.current_index],
            transition_duration=self.transition_steps,
            timestamp=self.current_state.t
        )
        self.switch_history.append(record)

        logger.info(
            f"Completed transition: now using {self.attractor_names[self.current_index]}, "
            f"total_switches={len(self.switch_history)}"
        )

    def force_switch(self, target_name: Optional[str] = None) -> None:
        """
        强制立即切换到指定吸引子

        Args:
            target_name: 目标吸引子名称（如果为None，随机选择）
        """
        if target_name and target_name in self.attractor_names:
            target_index = self.attractor_names.index(target_name)
        else:
            target_index = self._select_next_attractor()

        if target_index == self.current_index:
            logger.debug(f"Already using attractor {target_name}, no switch needed")
            return

        # 立即切换（无过渡）
        old_name = self.attractor_names[self.current_index]
        self.current_index = target_index

        # 初始化新吸引子状态
        new_attractor = self.get_current_attractor()
        z0 = new_attractor.generate_initial_state()
        warmup = new_attractor.integrate(
            z0=z0,
            t_span=(0.0, 5.0),
            n_steps=500,
            method="rk4"
        )
        self.current_state = warmup

        # 重置计数器
        self.step_counter = 0
        self.is_transitioning = False
        self.transition_progress = 0.0

        # 记录
        record = SwitchRecord(
            step=self.step_counter,
            from_attractor=old_name,
            to_attractor=self.attractor_names[self.current_index],
            transition_duration=0,
            timestamp=self.current_state.t
        )
        self.switch_history.append(record)

        logger.info(f"Force switched to {self.attractor_names[self.current_index]}")

    def get_state(self) -> torch.Tensor:
        """
        获取当前混沌状态向量

        Returns:
            当前状态向量 (3,)
        """
        if self.current_state is None:
            raise ValueError("Attractor states not initialized")
        return self.current_state.z.clone()

    def get_trajectory(self, n_steps: int) -> torch.Tensor:
        """
        生成一段混沌轨迹

        Args:
            n_steps: 轨迹长度

        Returns:
            轨迹张量 (n_steps, 3)
        """
        trajectory = []
        z = self.get_state()

        for _ in range(n_steps):
            z = self.step()
            trajectory.append(z.clone())

        return torch.stack(trajectory)

    def get_statistics(self) -> Dict:
        """
        获取当前状态统计信息

        Returns:
            统计信息字典
        """
        if self.current_state is None:
            return {"status": "not_initialized"}

        z = self.current_state.z
        return {
            "current_attractor": self.attractor_names[self.current_index],
            "state_norm": torch.norm(z).item(),
            "step_counter": self.step_counter,
            "transition_progress": self.transition_progress,
            "total_switches": len(self.switch_history)
        }

    def reset(self, seed: Optional[int] = None) -> None:
        """
        重置管理器状态

        Args:
            seed: 新的随机种子
        """
        self.step_counter = 0
        self.current_index = 0
        self.is_transitioning = False
        self.transition_progress = 0.0
        self.switch_history.clear()

        self.initialize_states(seed)

        logger.debug("AttractorManager reset")

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "current_index": self.current_index,
            "step_counter": self.step_counter,
            "is_transitioning": self.is_transitioning,
            "transition_progress": self.transition_progress,
            "current_state": self.current_state.to_dict() if self.current_state else None,
            "previous_state": self.previous_state.to_dict() if self.previous_state else None,
            "switch_history": [
                {
                    "step": r.step,
                    "from": r.from_attractor,
                    "to": r.to_attractor,
                    "duration": r.transition_duration,
                    "timestamp": r.timestamp
                }
                for r in self.switch_history
            ],
            "config": {
                "switch_interval": self.switch_interval,
                "transition_steps": self.transition_steps,
                "smoothing_factor": self.smoothing_factor,
                "core_subspace_dim": self.core_subspace_dim
            }
        }

    @classmethod
    def from_dict(cls, data: Dict, attractors: Dict[str, BaseAttractor]) -> "AttractorManager":
        """
        从字典恢复状态

        Args:
            data: 序列化的数据
            attractors: 预注册的吸引子字典

        Returns:
            AttractorManager 实例
        """
        config = data.get("config", {})
        manager = cls(
            core_subspace_dim=config.get("core_subspace_dim", 64),
            switch_interval=config.get("switch_interval", 5000),
            transition_steps=config.get("transition_steps", 100),
            smoothing_factor=config.get("smoothing_factor", 0.95)
        )

        manager.attractors = attractors
        manager.attractor_names = list(attractors.keys())

        manager.current_index = data.get("current_index", 0)
        manager.step_counter = data.get("step_counter", 0)
        manager.is_transitioning = data.get("is_transitioning", False)
        manager.transition_progress = data.get("transition_progress", 0.0)

        if data.get("current_state"):
            manager.current_state = AttractorState.from_dict(data["current_state"])
        if data.get("previous_state"):
            manager.previous_state = AttractorState.from_dict(data["previous_state"])

        return manager

    def __repr__(self) -> str:
        current_name = self.attractor_names[self.current_index] if self.attractor_names else "none"
        return (
            f"AttractorManager(attractors={len(self.attractors)}, "
            f"current='{current_name}', step={self.step_counter}, "
            f"switches={len(self.switch_history)})"
        )