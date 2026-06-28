"""
元意识场模块 - Meta Consciousness Field
==========================================

实现元意识场与自指深度计算，构成递归状态监控的核心动力学机制。

核心功能：
- 元意识场（MetaConsciousnessField）：基于状态变化率的指数加权积分
- 自指深度（SelfReferentialDepth）：基于元意识场的阶梯式涌现深度

元意识场公式：
M_pre(t) = ∫_{t-τ_M}^{t} ||dZ/dτ||²_{Σ_M} · w_M(t-τ) dτ

指数遗忘核：
w_M(s) = e^{-s/τ_M}

自指深度公式：
Λ(t) = clip( floor((M_pre - M_0) / ΔM), 0, L_max )
"""

import torch
import math
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class MetaConsciousnessField:
    """
    元意识场 MetaConsciousnessField
    
    基于指数移动平均近似的窗口积分，计算元意识场 M_pre(t)。
    元意识场度量联合状态空间中状态变化的"活跃度"，是自指深度涌现的基础。
    
    使用指数移动平均（EMA）高效近似积分运算，
    保证数值稳定性与非负性公理。
    """
    
    def __init__(
        self,
        window_time: float = 1.0,
        metric_dim: Optional[int] = None,
        device: str = "cpu"
    ):
        """
        初始化元意识场
        
        Args:
            window_time: 窗口时间 τ_M（秒），控制指数遗忘的时间尺度
            metric_dim: 度量矩阵 Σ_M 的维度，初始用单位矩阵近似（普通L2范数）
            device: 计算设备
        """
        self.window_time = window_time
        self.metric_dim = metric_dim
        self.device = device
        
        self._m_pre: float = 0.0
        self._prev_z: Optional[torch.Tensor] = None
        
        logger.info(
            f"MetaConsciousnessField initialized: "
            f"window_time={window_time}s, "
            f"metric_dim={metric_dim}"
        )
    
    def step(self, z_t: torch.Tensor, dt: float) -> float:
        """
        更新元意识场
        
        根据当前联合状态 z_t 和时间步长 dt，
        计算状态变化率并通过指数移动平均更新元意识场。
        
        Args:
            z_t: 当前联合状态向量
            dt: 时间步长（秒）
            
        Returns:
            更新后的元意识场标量值
        """
        z_t = z_t.to(self.device).float()
        
        if self._prev_z is None:
            self._prev_z = z_t.clone()
            return self._m_pre
        
        dz = z_t - self._prev_z
        dz_norm_sq = torch.sum(dz ** 2).item()
        dz_dt_norm_sq = dz_norm_sq / (dt ** 2) if dt > 0 else 0.0
        
        if self.window_time > 0:
            alpha = 1.0 - math.exp(-dt / self.window_time)
        else:
            alpha = 1.0
        
        self._m_pre = alpha * dz_dt_norm_sq + (1.0 - alpha) * self._m_pre
        
        self._m_pre = max(0.0, self._m_pre)
        
        self._prev_z = z_t.clone()
        
        return self._m_pre
    
    def get_value(self) -> float:
        """
        获取当前元意识场标量值
        
        Returns:
            元意识场值 M_pre(t)
        """
        return self._m_pre
    
    def reset(self) -> None:
        """重置元意识场为0，清除历史状态"""
        self._m_pre = 0.0
        self._prev_z = None
        logger.debug("MetaConsciousnessField reset")
    
    def __repr__(self) -> str:
        return (
            f"MetaConsciousnessField("
            f"window_time={self.window_time}s, "
            f"current_value={self._m_pre:.6f})"
        )


class SelfReferentialDepth:
    """
    自指深度 SelfReferentialDepth
    
    基于元意识场的阶梯函数计算自指深度 Λ(t)。
    自指深度是元意识场累积到一定阈值后涌现的离散层级，
    代表递归自指的层数。
    
    涌现单调性保证：当元意识场增大时，深度不会减少
    （允许元意识场减小时深度相应减小）。
    """
    
    def __init__(
        self,
        emergence_threshold: float = 1.0,
        level_spacing: float = 0.5,
        max_depth: int = 3
    ):
        """
        初始化自指深度计算器
        
        Args:
            emergence_threshold: 涌现阈值 M_0，元意识场超过此值才开始有深度
            level_spacing: 层级间距 ΔM，每增加此值深度加1
            max_depth: 最大深度 L_max
        """
        self.emergence_threshold = emergence_threshold
        self.level_spacing = level_spacing
        self.max_depth = max_depth
        
        self._current_depth: int = 0
        self._last_m_pre: float = 0.0
        
        logger.info(
            f"SelfReferentialDepth initialized: "
            f"M_0={emergence_threshold}, "
            f"ΔM={level_spacing}, "
            f"L_max={max_depth}"
        )
    
    def compute(self, m_pre: float) -> int:
        """
        计算当前自指深度
        
        根据元意识场值通过阶梯函数计算离散深度层级，
        同时验证涌现单调性。
        
        Args:
            m_pre: 当前元意识场值 M_pre(t)
            
        Returns:
            自指深度 Λ(t)
        """
        if self.level_spacing > 0:
            raw_depth = math.floor((m_pre - self.emergence_threshold) / self.level_spacing)
        else:
            raw_depth = 0
        
        depth = max(0, min(int(raw_depth), self.max_depth))
        
        if m_pre > self._last_m_pre and depth < self._current_depth:
            logger.warning(
                f"Monotonicity violation detected: "
                f"M_pre increased ({self._last_m_pre:.6f} -> {m_pre:.6f}) "
                f"but depth decreased ({self._current_depth} -> {depth}). "
                f"Enforcing monotonicity."
            )
            depth = self._current_depth
        
        self._current_depth = depth
        self._last_m_pre = m_pre
        
        return depth
    
    def get_depth(self) -> int:
        """
        获取当前自指深度
        
        Returns:
            当前深度值
        """
        return self._current_depth
    
    def reset(self) -> None:
        """重置自指深度为0"""
        self._current_depth = 0
        self._last_m_pre = 0.0
        logger.debug("SelfReferentialDepth reset")
    
    def __repr__(self) -> str:
        return (
            f"SelfReferentialDepth("
            f"depth={self._current_depth}/{self.max_depth}, "
            f"last_M_pre={self._last_m_pre:.6f})"
        )


if __name__ == "__main__":
    import torch
    
    print("=" * 60)
    print("元意识场模块测试")
    print("=" * 60)
    
    print("\n1. 测试 MetaConsciousnessField")
    print("-" * 40)
    
    field = MetaConsciousnessField(window_time=1.0)
    print(f"初始化: {field}")
    print(f"初始值: {field.get_value():.6f}")
    
    dt = 0.01
    z_dim = 8
    
    print(f"\n逐步更新 (dt={dt}s, z_dim={z_dim}):")
    z = torch.zeros(z_dim)
    for i in range(10):
        z = z + 0.1 * torch.randn(z_dim)
        val = field.step(z, dt)
        print(f"  Step {i+1}: M_pre = {val:.6f}")
    
    print(f"\n最终值: {field.get_value():.6f}")
    
    print("\n重置后:")
    field.reset()
    print(f"  重置值: {field.get_value():.6f}")
    
    print("\n2. 测试 SelfReferentialDepth")
    print("-" * 40)
    
    depth_calc = SelfReferentialDepth(
        emergence_threshold=1.0,
        level_spacing=0.5,
        max_depth=3
    )
    print(f"初始化: {depth_calc}")
    
    test_values = [0.0, 0.5, 1.0, 1.2, 1.6, 2.0, 2.7, 3.5, 5.0, 2.0, 1.0]
    print(f"\n测试值序列: {test_values}")
    print("\n深度计算:")
    for m_val in test_values:
        d = depth_calc.compute(m_val)
        print(f"  M_pre={m_val:5.2f}  ->  Λ={d} (max={depth_calc.max_depth})")
    
    print(f"\n当前深度: {depth_calc.get_depth()}")
    
    print("\n3. 集成测试：场 -> 深度联动")
    print("-" * 40)
    
    field2 = MetaConsciousnessField(window_time=0.5)
    depth2 = SelfReferentialDepth(emergence_threshold=2.0, level_spacing=1.0, max_depth=3)
    
    z2 = torch.zeros(4)
    dt2 = 0.05
    
    print(f"\n逐步模拟 (dt={dt2}s):")
    for i in range(20):
        z2 = z2 + 0.15 * torch.randn(4)
        m = field2.step(z2, dt2)
        d = depth2.compute(m)
        if i % 4 == 0 or i == 19:
            print(f"  Step {i+1:2d}: M_pre={m:.4f}, Λ={d}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
