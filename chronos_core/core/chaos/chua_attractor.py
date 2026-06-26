"""
蔡氏电路吸引子
==============

实现 Chua's Circuit 混沌系统，用于生成内源性动力学。

Chua 方程：
    dx/dt = α(y - x - f(x))
    dy/dt = x - y + z
    dz/dt = -β*y

非线性函数 f(x):
    f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)

经典参数：α=15.35, β=28, m0=-1.143, m1=-0.714
"""

import torch
from typing import Optional
import logging

from .base_attractor import BaseAttractor

logger = logging.getLogger(__name__)


class ChuaAttractor(BaseAttractor):
    """
    Chua's Circuit 混沌吸引子

    简单电路实现的混沌系统，产生双螺旋吸引子。

    Attributes:
        alpha: 状态空间缩放参数
        beta: 反馈增益参数
        m0: 非线性函数第一段斜率
        m1: 非线性函数第二段斜率
    """

    def __init__(
        self,
        alpha: float = 15.35,
        beta: float = 28.0,
        m0: float = -1.143,
        m1: float = -0.714,
        dt: float = 0.01,
        device: Optional[str] = None
    ):
        """
        初始化 Chua 吸引子

        Args:
            alpha: 状态空间缩放参数 (默认 15.35)
            beta: 反馈增益参数 (默认 28.0)
            m0: 非线性函数第一段斜率 (默认 -1.143)
            m1: 非线性函数第二段斜率 (默认 -0.714)
            dt: 积分步长
            device: 计算设备
        """
        super().__init__(
            name="Chua",
            dim=3,
            dt=dt,
            device=device
        )

        self.alpha = alpha
        self.beta = beta
        self.m0 = m0
        self.m1 = m1

        # Chua 系统特有的稳定性阈值（轨迹通常在 ~10 范围内）
        self.stability_threshold = 30.0

        logger.info(
            f"Chua attractor initialized: α={alpha}, β={beta}, "
            f"m0={m0}, m1={m1}"
        )

    def _chua_function(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chua 非线性函数

        f(x) = m1*x + 0.5*(m0 - m1)*(|x+1| - |x-1|)

        这是一个三段分段线性函数。

        Args:
            x: 输入值

        Returns:
            f(x) 的值
        """
        return self.m1 * x + 0.5 * (self.m0 - self.m1) * (
            torch.abs(x + 1.0) - torch.abs(x - 1.0)
        )

    def derivatives(self, z: torch.Tensor, t: float = 0.0) -> torch.Tensor:
        """
        计算 Chua 系统的导数

        Chua 方程组：
            dx/dt = α(y - x - f(x))
            dy/dt = x - y + z
            dz/dt = -β*y

        Args:
            z: 状态向量 [x, y, z]
            t: 时间（Chua 系统不显式依赖时间）

        Returns:
            导数向量 [dx/dt, dy/dt, dz/dt]
        """
        x, y, z_val = z[0], z[1], z[2]

        # Chua 方程
        fx = self._chua_function(x)
        dx = self.alpha * (y - x - fx)
        dy = x - y + z_val
        dz = -self.beta * y

        return torch.stack([dx, dy, dz])

    def get_equilibrium_points(self) -> list:
        """
        计算 Chua 系统的平衡点

        对于标准参数，有三个平衡点：
        - 原点 (0, 0, 0)
        - 两个对称的非零平衡点

        Returns:
            平衡点列表
        """
        # 原点总是平衡点
        equilibria = [(0.0, 0.0, 0.0)]

        # 计算非零平衡点
        # 在线性区域 f(x) = m0*x 或 m1*x
        # 对于 |x| < 1, f(x) = m1*x
        # 平衡条件：y = x + f(x) = x + m1*x
        #          z = y - x = m1*x
        #          0 = -β*y => y = 0

        # 这意味着 x = 0（原点）
        # 对于 |x| > 1, f(x) = m0*x + (m1-m0) 或 m0*x - (m1-m0)

        # 非零平衡点的近似
        if self.alpha > 0 and self.beta > 0:
            try:
                c = (self.m1 - self.m0) / 2.0
                # 非零平衡点大约在 x ≈ ±(1 + c/alpha)
                x_nonzero = 1.0 + abs(c) / self.alpha
                y_eq = 0.0
                z_eq = -y_eq + x_nonzero  # 从 dy=0: z = y - x

                equilibria.append((x_nonzero, y_eq, z_eq))
                equilibria.append((-x_nonzero, y_eq, -z_eq))
            except:
                pass

        return equilibria

    def is_chaotic(self) -> bool:
        """
        判断当前参数是否产生混沌行为

        对于标准参数 m0=-1.143, m1=-0.714，
        当 α > 10 且 β > 20 时产生混沌。

        Returns:
            是否为混沌状态
        """
        return self.alpha > 10.0 and self.beta > 20.0

    def get_bifurcation_type(self) -> str:
        """
        判断当前的分岔类型

        Returns:
            分岔类型描述
        """
        if self.alpha < 5:
            return "stable"
        elif 5 <= self.alpha < 10:
            return "periodic"
        else:
            return "double_scroll_chaotic"

    def generate_near_spiral(
        self,
        spiral_index: int = 0,
        perturbation: float = 0.1,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        在双螺旋的一个螺旋附近生成初始状态

        Chua 吸引子有两个对称的螺旋。

        Args:
            spiral_index: 螺旋索引 (0=正螺旋, 1=负螺旋)
            perturbation: 微扰幅度
            seed: 随机种子

        Returns:
            初始状态向量
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Chua 吸引子螺旋的大致位置
        if spiral_index == 0:
            # 正螺旋
            x_approx = 2.0
            y_approx = 0.0
            z_approx = 0.5
        else:
            # 负螺旋
            x_approx = -2.0
            y_approx = 0.0
            z_approx = -0.5

        z0 = torch.tensor(
            [x_approx, y_approx, z_approx],
            dtype=torch.float32,
            device=self.device
        )
        z0 = z0 + torch.randn(3, device=self.device) * perturbation

        return z0

    def compute_band_width(self, z: torch.Tensor) -> float:
        """
        估算螺旋带宽度

        基于 x 值估算当前所在的螺旋带宽度。

        Args:
            z: 状态向量

        Returns:
            估算的带宽度
        """
        x = z[0].item()
        # 带宽度与非线性函数的转折点位置相关
        return abs(x) * abs(self.m1 - self.m0)

    def __repr__(self) -> str:
        return (
            f"ChuaAttractor(α={self.alpha}, β={self.beta}, "
            f"m0={self.m0}, m1={self.m1}, device='{self.device}')"
        )