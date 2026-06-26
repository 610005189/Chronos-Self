"""
混沌吸引子基类
==============

定义所有混沌吸引子的通用接口和基础功能。
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttractorState:
    """
    吸引子状态容器

    Attributes:
        z: 当前状态向量 (x, y, z)
        t: 当前时间
        trajectory: 历史轨迹（可选）
        metadata: 额外元数据
    """
    z: torch.Tensor  # shape: (3,)
    t: float = 0.0
    trajectory: Optional[List[torch.Tensor]] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.trajectory is None:
            self.trajectory = []

    def to_dict(self) -> dict:
        """序列化为字典"""
        return {
            "z": self.z.detach().cpu().numpy().tolist(),
            "t": self.t,
            "trajectory": [z.detach().cpu().numpy().tolist() for z in self.trajectory],
            "metadata": self.metadata.copy()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttractorState":
        """从字典反序列化"""
        trajectory = [torch.tensor(z, dtype=torch.float32) for z in data.get("trajectory", [])]
        return cls(
            z=torch.tensor(data["z"], dtype=torch.float32),
            t=data.get("t", 0.0),
            trajectory=trajectory if trajectory else None,
            metadata=data.get("metadata", {})
        )


class BaseAttractor(ABC):
    """
    混沌吸引子抽象基类

    所有混沌吸引子都需要实现此接口，提供统一的动力学演化接口。

    Attributes:
        name: 吸引子名称
        dim: 状态空间维度（默认为3）
        dt: 默认积分步长
        device: 计算设备
    """

    def __init__(
        self,
        name: str,
        dim: int = 3,
        dt: float = 0.01,
        device: Optional[str] = None
    ):
        """
        初始化吸引子

        Args:
            name: 吸引子名称
            dim: 状态空间维度
            dt: 默认积分步长
            device: 计算设备 ('cpu' 或 'cuda')
        """
        self.name = name
        self.dim = dim
        self.dt = dt
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 数值稳定性阈值
        self.stability_threshold = 1e10
        self.min_value = -1e10
        self.max_value = 1e10

        logger.debug(f"Initialized {name} attractor on {self.device}")

    @abstractmethod
    def derivatives(self, z: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        计算状态向量的导数 dz/dt

        Args:
            z: 当前状态向量 (dim,)
            t: 当前时间（某些吸引子可能需要）

        Returns:
            状态导数向量 (dim,)
        """
        pass

    def step(
        self,
        z: torch.Tensor,
        dt: Optional[float] = None,
        method: str = "rk4"
    ) -> torch.Tensor:
        """
        执行单步积分

        Args:
            z: 当前状态向量 (dim,)
            dt: 积分步长（使用默认值如果未指定）
            method: 积分方法 ('euler', 'rk4')

        Returns:
            新的状态向量 (dim,)
        """
        if dt is None:
            dt = self.dt

        if method == "euler":
            return self._euler_step(z, dt)
        elif method == "rk4":
            return self._rk4_step(z, dt)
        else:
            raise ValueError(f"Unknown integration method: {method}")

    def _euler_step(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """Euler 方法单步积分"""
        dz = self.derivatives(z)
        return z + dt * dz

    def _rk4_step(self, z: torch.Tensor, dt: float) -> torch.Tensor:
        """四阶 Runge-Kutta 方法单步积分"""
        k1 = self.derivatives(z)
        k2 = self.derivatives(z + 0.5 * dt * k1)
        k3 = self.derivatives(z + 0.5 * dt * k2)
        k4 = self.derivatives(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(
        self,
        z0: torch.Tensor,
        t_span: Tuple[float, float],
        n_steps: Optional[int] = None,
        method: str = "rk4",
        record_trajectory: bool = False
    ) -> AttractorState:
        """
        积分求解吸引子动力学

        Args:
            z0: 初始状态向量 (dim,)
            t_span: 时间范围 (t_start, t_end)
            n_steps: 积分步数（如果为None，使用默认dt计算）
            method: 积分方法
            record_trajectory: 是否记录轨迹

        Returns:
            AttractorState 包含最终状态和轨迹
        """
        t_start, t_end = t_span

        if n_steps is None:
            n_steps = int((t_end - t_start) / self.dt)

        dt = (t_end - t_start) / n_steps

        # 确保张量在正确的设备上
        z = z0.clone().to(self.device)
        t = t_start

        trajectory = [z.clone()] if record_trajectory else None

        for _ in range(n_steps):
            z = self.step(z, dt, method)
            t += dt

            # 数值稳定性检查
            if not self._check_stability(z):
                logger.warning(
                    f"{self.name} attractor became unstable at t={t:.4f}, "
                    f"z_norm={torch.norm(z).item():.4e}"
                )
                break

            if record_trajectory:
                trajectory.append(z.clone())

        return AttractorState(
            z=z,
            t=t,
            trajectory=trajectory,
            metadata={"method": method, "n_steps": n_steps}
        )

    def _check_stability(self, z: torch.Tensor) -> bool:
        """检查数值稳定性"""
        z_norm = torch.norm(z).item()

        # 检查是否发散
        if z_norm > self.stability_threshold:
            return False

        # 检查 NaN 或 Inf
        if torch.isnan(z).any() or torch.isinf(z).any():
            return False

        return True

    def generate_initial_state(
        self,
        seed: Optional[int] = None,
        range_scale: float = 1.0
    ) -> torch.Tensor:
        """
        生成随机初始状态

        Args:
            seed: 随机种子
            range_scale: 状态范围缩放因子

        Returns:
            随机初始状态向量 (dim,)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # 默认在 [-1, 1] 范围内生成随机初始状态
        z0 = torch.randn(self.dim, device=self.device) * range_scale
        return z0

    def clip_state(self, z: torch.Tensor) -> torch.Tensor:
        """
        裁剪状态向量以确保数值稳定性

        Args:
            z: 状态向量

        Returns:
            裁剪后的状态向量
        """
        return torch.clamp(z, self.min_value, self.max_value)

    def compute_lyapunov_exponent(
        self,
        z0: torch.Tensor,
        n_steps: int = 10000,
        dt: Optional[float] = None
    ) -> float:
        """
        估算最大 Lyapunov 指数

        使用 Rosenstein 算法的简化版本。

        Args:
            z0: 初始状态
            n_steps: 积分步数
            dt: 积分步长

        Returns:
            估算的最大 Lyapunov 指数
        """
        if dt is None:
            dt = self.dt

        # 预热阶段
        z = z0.clone()
        for _ in range(1000):
            z = self.step(z, dt)

        # 计算参考轨迹
        z_ref = z.clone()
        distances = []

        for _ in range(n_steps):
            # 演化参考轨迹
            z_ref = self.step(z_ref, dt)

            # 创建微扰轨迹
            epsilon = 1e-8
            z_pert = z_ref + torch.randn_like(z_ref) * epsilon
            z_pert = self.step(z_pert, dt)

            # 计算距离
            dist = torch.norm(z_pert - z_ref).item()
            if dist > 0:
                distances.append(np.log(dist / epsilon))

        if len(distances) == 0:
            return 0.0

        # Lyapunov 指数 = 距离增长率的平均
        return np.mean(distances) / dt

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', dim={self.dim}, device='{self.device}')"