"""
洛伦兹吸引子
============

实现经典的 Lorenz 混沌系统，用于生成内源性动力学。

洛伦兹方程：
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz

经典参数：σ=10, ρ=28, β=8/3
"""

import torch
from typing import Optional
import logging

from .base_attractor import BaseAttractor

logger = logging.getLogger(__name__)


class LorenzAttractor(BaseAttractor):
    """
    洛伦兹混沌吸引子

    经典的三维混沌系统，产生著名的"蝴蝶效应"轨迹。

    Attributes:
        sigma: Prandtl 数 (σ)
        rho: Rayleigh 数 (ρ)
        beta: 几何因子 (β)
    """

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0 / 3.0,
        dt: float = 0.01,
        device: Optional[str] = None
    ):
        """
        初始化洛伦兹吸引子

        Args:
            sigma: Prandtl 数，控制对流强度 (默认 10.0)
            rho: Rayleigh 数，控制混沌程度 (默认 28.0，产生混沌)
            beta: 几何因子 (默认 8/3)
            dt: 积分步长
            device: 计算设备
        """
        super().__init__(
            name="Lorenz",
            dim=3,
            dt=dt,
            device=device
        )

        self.sigma = sigma
        self.rho = rho
        self.beta = beta

        # 洛伦兹系统特有的稳定性阈值（轨迹通常在 ~30 范围内）
        self.stability_threshold = 100.0

        logger.info(
            f"Lorenz attractor initialized: σ={sigma}, ρ={rho}, β={beta:.4f}"
        )

    def derivatives(self, z: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        计算洛伦兹系统的导数

        Lorenz 方程组：
            dx/dt = σ(y - x)
            dy/dt = x(ρ - z) - y
            dz/dt = xy - βz

        Args:
            z: 状态向量 [x, y, z]
            t: 时间（洛伦兹系统不显式依赖时间）

        Returns:
            导数向量 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z_val = z[0], z[1], z[2]

        # 洛伦兹方程
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z_val) - y
        dz = x * y - self.beta * z_val

        return torch.stack([dx, dy, dz])

    def get_equilibrium_points(self) -> list:
        """
        计算洛伦兹系统的平衡点

        对于 ρ > 1，有三个平衡点：
        - O = (0, 0, 0)
        - C+ = (sqrt(β(ρ-1)), sqrt(β(ρ-1)), ρ-1)
        - C- = (-sqrt(β(ρ-1)), -sqrt(β(ρ-1)), ρ-1)

        Returns:
            平衡点列表
        """
        equilibria = [(0.0, 0.0, 0.0)]

        if self.rho > 1:
            sqrt_val = torch.sqrt(torch.tensor(self.beta * (self.rho - 1)))
            c_plus = (sqrt_val.item(), sqrt_val.item(), self.rho - 1)
            c_minus = (-sqrt_val.item(), -sqrt_val.item(), self.rho - 1)
            equilibria.extend([c_plus, c_minus])

        return equilibria

    def is_chaotic(self) -> bool:
        """
        判断当前参数是否产生混沌行为

        对于标准参数 σ=10, β=8/3，当 ρ > 24.74 时产生混沌。

        Returns:
            是否为混沌状态
        """
        # 混沌阈值约为 24.74 (对于标准参数)
        critical_rho = 24.74
        return self.rho > critical_rho

    def get_bifurcation_type(self) -> str:
        """
        判断当前的分岔类型

        Returns:
            分岔类型描述
        """
        if self.rho < 1:
            return "stable_origin"
        elif 1 <= self.rho < 24.74:
            return "stable_two_equilibria"
        else:
            return "chaotic_attractor"

    def generate_near_equilibrium(
        self,
        equilibrium_index: int = 1,
        perturbation: float = 0.1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        在平衡点附近生成初始状态

        Args:
            equilibrium_index: 平衡点索引 (0=原点, 1=C+, 2=C-)
            perturbation: 微扰幅度
            seed: 随机种子

        Returns:
            初始状态向量
        """
        if seed is not None:
            torch.manual_seed(seed)

        equilibria = self.get_equilibrium_points()

        if equilibrium_index >= len(equilibria):
            equilibrium_index = 0

        eq = equilibria[equilibrium_index]
        z0 = torch.tensor(eq, dtype=torch.float32, device=self.device)
        z0 = z0 + torch.randn(3, device=self.device) * perturbation

        return z0

    def compute_energy(self, z: torch.Tensor) -> float:
        """
        计算状态的"能量"（近似）

        使用 z^2 + x^2 + y^2 作为近似能量度量。

        Args:
            z: 状态向量

        Returns:
            能量值
        """
        return torch.norm(z).pow(2).item()

    def __repr__(self) -> str:
        return (
            f"LorenzAttractor(σ={self.sigma}, ρ={self.rho}, "
            f"β={self.beta:.4f}, device='{self.device}')"
        )