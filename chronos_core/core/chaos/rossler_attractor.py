"""
罗斯勒吸引子
============

实现 Rössler 混沌系统，用于生成内源性动力学。

Rössler 方程：
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

经典参数：a=0.2, b=0.2, c=5.7
"""

import torch
from typing import Optional
import logging

from .base_attractor import BaseAttractor

logger = logging.getLogger(__name__)


class RosslerAttractor(BaseAttractor):
    """
    Rössler 混沌吸引子

    简化的三维混沌系统，轨迹呈现螺旋状结构。

    Attributes:
        a: 线性耦合参数
        b: 基线偏移参数
        c: 非线性强度参数
    """

    def __init__(
        self,
        a: float = 0.2,
        b: float = 0.2,
        c: float = 5.7,
        dt: float = 0.01,
        device: Optional[str] = None
    ):
        """
        初始化 Rössler 吸引子

        Args:
            a: 线性耦合参数 (默认 0.2)
            b: 基线偏移参数 (默认 0.2)
            c: 非线性强度参数 (默认 5.7，产生混沌)
            dt: 积分步长
            device: 计算设备
        """
        super().__init__(
            name="Rossler",
            dim=3,
            dt=dt,
            device=device
        )

        self.a = a
        self.b = b
        self.c = c

        # Rössler 系统特有的稳定性阈值（轨迹通常在 ~20 范围内）
        self.stability_threshold = 50.0

        logger.info(
            f"Rossler attractor initialized: a={a}, b={b}, c={c}"
        )

    def derivatives(self, z: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        计算 Rössler 系统的导数

        Rössler 方程组：
            dx/dt = -y - z
            dy/dt = x + a*y
            dz/dt = b + z*(x - c)

        Args:
            z: 状态向量 [x, y, z]
            t: 时间（Rössler 系统不显式依赖时间）

        Returns:
            导数向量 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z_val = z[0], z[1], z[2]

        # Rössler 方程
        dx = -y - z_val
        dy = x + self.a * y
        dz = self.b + z_val * (x - self.c)

        return torch.stack([dx, dy, dz])

    def get_equilibrium_points(self) -> list:
        """
        计算 Rössler 系统的平衡点

        平衡点由以下方程确定：
            -y - z = 0
            x + a*y = 0
            b + z*(x - c) = 0

        Returns:
            平衡点列表
        """
        # 对于标准参数，有两个平衡点
        # 计算需要解非线性方程，这里给出近似值
        try:
            # 使用数值方法求解
            import numpy as np
            from scipy.optimize import fsolve

            def equations(vars):
                x, y, z = vars
                return [
                    -y - z,
                    x + self.a * y,
                    self.b + z * (x - self.c)
                ]

            # 从多个初始点尝试找到平衡点
            initial_guesses = [
                (0.0, 0.0, 0.0),
                (self.c, 0.0, 0.0),
                (self.c, -self.b, self.b),
            ]

            equilibria = []
            for guess in initial_guesses:
                try:
                    sol = fsolve(equations, guess, maxfev=1000)
                    if not any(np.isnan(sol)):
                        equilibria.append(tuple(sol))
                except:
                    continue

            return equilibria
        except ImportError:
            # 如果 scipy 不可用，返回空列表
            return []

    def is_chaotic(self) -> bool:
        """
        判断当前参数是否产生混沌行为

        对于标准参数 a=0.2, b=0.2，当 c > 4.2 时产生混沌。

        Returns:
            是否为混沌状态
        """
        critical_c = 4.2
        return self.c > critical_c

    def get_bifurcation_type(self) -> str:
        """
        判断当前的分岔类型

        Returns:
            分岔类型描述
        """
        if self.c < 2.5:
            return "period_1"
        elif 2.5 <= self.c < 3.5:
            return "period_2"
        elif 3.5 <= self.c < 4.2:
            return "period_4"
        else:
            return "chaotic"

    def generate_near_band(
        self,
        band_position: float = 0.0,
        perturbation: float = 0.1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        在螺旋带附近生成初始状态

        Rössler 吸引子的轨迹形成一个扩展的螺旋带。

        Args:
            band_position: 螺旋带的位置参数
            perturbation: 微扰幅度
            seed: 随机种子

        Returns:
            初始状态向量
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Rössler 吸引子的典型轨迹范围
        x_approx = band_position * 5.0
        y_approx = -self.b * 0.5
        z_approx = 0.1

        z0 = torch.tensor(
            [x_approx, y_approx, z_approx],
            dtype=torch.float32,
            device=self.device
        )
        z0 = z0 + torch.randn(3, device=self.device) * perturbation

        return z0

    def compute_rotation_frequency(self, z: torch.Tensor) -> float:
        """
        估算螺旋旋转频率

        基于 x-y 平面的旋转估算频率。

        Args:
            z: 状态向量

        Returns:
            估算的旋转频率
        """
        # 使用 x-y 平面的旋转来估算
        x, y = z[0].item(), z[1].item()
        angle = torch.atan2(torch.tensor(y), torch.tensor(x)).item()

        # 频率约为 1/(2π) 到 2/(2π)，取决于参数
        return 1.0 / (2.0 * torch.pi).item()

    def __repr__(self) -> str:
        return (
            f"RosslerAttractor(a={self.a}, b={self.b}, "
            f"c={self.c}, device='{self.device}')"
        )