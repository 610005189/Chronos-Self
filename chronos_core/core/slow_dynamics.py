"""
慢变量动力学系统
================

实现小时-天级慢速人格/身份状态演化的动力学系统。

核心功能：
- 演化方程：dE_slow/dt = G_φ(E_slow) + α(t) · Pooling(E_fast) - γ · (E_slow - E_slow_baseline)
- 维度：512
- 快变量池化机制（平均池化、注意力池化）
- 弹性恢复项（baseline回归）
- 慢变量低频更新策略（每100快变量步更新1次）
- 自发的慢变量演化（G_φ）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union, List
import logging
from dataclasses import dataclass, field
import math

from chronos_core.utils.config import DimensionalityConfig, CouplingStabilityConfig, MemoryTemporalConfig
from .neural_ode import DynamicsFunction

logger = logging.getLogger(__name__)


@dataclass
class SlowDynamicsConfig:
    """慢变量动力学配置"""

    # 状态维度
    slow_dim: int = 512
    fast_dim: int = 2048

    # 网络架构
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    activation: str = "tanh"

    # 动力学参数
    elastic_coeff: float = 0.02  # 弹性恢复系数 γ（提高10倍，减少漂移）
    baseline_decay: float = 0.001  # baseline 衰减率
    spontaneous_rate: float = 0.0005  # 自发演化速率（降低以减少自发变化）

    # 池化配置
    pooling_method: str = "average"  # 'average', 'attention', 'max'
    attention_heads: int = 4  # 注意力池化头数

    # 时间尺度
    time_scale: float = 0.01  # 相对时间尺度（相对于快变量）
    slow_update_frequency: int = 100  # 每N快变量步更新1次慢变量

    # 稳定性参数
    max_state_change: float = 0.1  # 最大单步状态变化
    baseline_drift_threshold: float = 0.5  # baseline 漂移阈值

    # 耦合参数
    base_coupling_coeff: float = 0.001  # 基础耦合系数 α_0（降低10倍，减少快变量扰动）
    coupling_adaptation: float = 0.5  # 耦合适应性 β（降低以更稳定）


class PoolingMechanism(nn.Module):
    """
    快变量池化机制

    将高维快变量状态池化为低维慢变量可接收的信号。

    支持三种池化方法：
    1. 平均池化（average pooling）
    2. 注意力池化（attention pooling）
    3. 最大池化（max pooling）
    """

    def __init__(
        self,
        fast_dim: int = 2048,
        slow_dim: int = 512,
        method: str = "average",
        attention_heads: int = 4,
        device: Optional[str] = None
    ):
        """
        初始化池化机制

        Args:
            fast_dim: 快变量维度
            slow_dim: 慢变量维度
            method: 池化方法
            attention_heads: 注意力头数（用于注意力池化）
            device: 计算设备
        """
        super().__init__()

        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.method = method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 根据方法创建相应的池化层
        if method == "average":
            # 平均池化 + 线性投影
            self.pooling_layer = nn.Sequential(
                nn.AdaptiveAvgPool1d(slow_dim),
                nn.Flatten()
            )

        elif method == "attention":
            # 注意力池化
            # 将快变量分割为多个块，使用注意力机制选择重要部分
            self.chunk_size = fast_dim // attention_heads
            self.attention_heads = attention_heads

            # 注意力权重网络
            self.attention_weights = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.chunk_size, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Softmax(dim=-1)
                )
                for _ in range(attention_heads)
            ])

            # 投影层
            self.projection = nn.Linear(fast_dim, slow_dim)

        elif method == "max":
            # 最大池化 + 线性投影
            self.pooling_layer = nn.Sequential(
                nn.AdaptiveMaxPool1d(slow_dim),
                nn.Flatten()
            )

        else:
            # 默认使用平均池化
            self.pooling_layer = nn.Sequential(
                nn.AdaptiveAvgPool1d(slow_dim),
                nn.Flatten()
            )

        # 统计信息
        self.pooling_calls = 0

        logger.debug(
            f"PoolingMechanism created: method={method}, "
            f"fast_dim={fast_dim}, slow_dim={slow_dim}"
        )

    def forward(self, E_fast: torch.Tensor) -> torch.Tensor:
        """
        执行池化

        Args:
            E_fast: 快变量状态 (fast_dim,) 或 (batch_size, fast_dim)

        Returns:
            池化后的信号 (slow_dim,) 或 (batch_size, slow_dim)
        """
        # 确保输入在设备上
        E_fast = E_fast.to(self.device)

        # 处理维度
        if E_fast.dim() == 1:
            # 单样本：添加批次维度
            E_fast = E_fast.unsqueeze(0)

        batch_size = E_fast.shape[0]

        if self.method == "average" or self.method == "max":
            # 使用 PyTorch 的池化层
            # 需要将形状调整为 (batch_size, 1, fast_dim)
            E_fast_reshaped = E_fast.unsqueeze(1)
            pooled = self.pooling_layer(E_fast_reshaped)
            # pooled 形状为 (batch_size, slow_dim)

        elif self.method == "attention":
            # 注意力池化
            # 分割为多个块
            chunks = E_fast.view(batch_size, self.attention_heads, self.chunk_size)

            # 计算每个块的注意力权重
            weighted_chunks = []
            for i, attention_net in enumerate(self.attention_weights):
                chunk = chunks[:, i, :]  # (batch_size, chunk_size)
                weight = attention_net(chunk)  # (batch_size, 1)
                weighted_chunk = chunk * weight
                weighted_chunks.append(weighted_chunk)

            # 合加权块
            weighted_E_fast = torch.cat(weighted_chunks, dim=-1)

            # 投影到慢变量维度
            pooled = self.projection(weighted_E_fast)

        else:
            # 默认平均池化
            pooled = E_fast.mean(dim=-1, keepdim=True).expand(batch_size, self.slow_dim)

        # 如果输入是单样本，恢复形状
        if batch_size == 1 and pooled.shape[0] == 1:
            pooled = pooled.squeeze(0)

        self.pooling_calls += 1

        return pooled

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "pooling_calls": self.pooling_calls
        }


class SpontaneousEvolution(nn.Module):
    """
    自发演化函数 G_φ

    实现慢变量的自发动力学演化（不受快变量直接影响）。

    这反映了人格和身份的内在稳定性与缓慢变化趋势。
    """

    def __init__(
        self,
        slow_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "tanh",
        spontaneous_rate: float = 0.001,
        device: Optional[str] = None
    ):
        """
        初始化自发演化函数

        Args:
            slow_dim: 慢变量维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            activation: 激活函数
            spontaneous_rate: 自发演化速率
            device: 计算设备
        """
        super().__init__()

        self.slow_dim = slow_dim
        self.spontaneous_rate = spontaneous_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 激活函数选择
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.Tanh()

        # 构建网络
        layers = []

        # 输入层
        layers.append(nn.Linear(slow_dim, hidden_dim))
        layers.append(act_fn)

        # 隐藏层
        for _ in range(num_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn)

        # 输出层（小值初始化，确保自发演化缓慢）
        output_layer = nn.Linear(hidden_dim, slow_dim)
        nn.init.constant_(output_layer.weight, 0.001)
        nn.init.constant_(output_layer.bias, 0.0)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

        # 统计
        self.forward_calls = 0

        logger.debug(
            f"SpontaneousEvolution created: slow_dim={slow_dim}, "
            f"rate={spontaneous_rate}"
        )

    def forward(self, E_slow: torch.Tensor) -> torch.Tensor:
        """
        计算自发演化项

        Args:
            E_slow: 慢变量状态

        Returns:
            自发演化贡献 (slow_dim,)
        """
        E_slow = E_slow.to(self.device)

        # 网络输出
        spontaneous = self.network(E_slow)

        # 缩放自发演化速率
        spontaneous = spontaneous * self.spontaneous_rate

        self.forward_calls += 1

        return spontaneous

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "forward_calls": self.forward_calls
        }


class SlowDynamicsFunction(DynamicsFunction):
    """
    慢变量动力学函数

    实现 Neural ODE 的动力学函数接口：
        dE_slow/dt = G_φ(E_slow) + α(t) · Pooling(E_fast) - γ · (E_slow - E_slow_baseline)

    包含三个部分：
    1. 自发演化 G_φ(E_slow)：内在稳定性与变化趋势
    2. 快慢耦合 α(t) · Pooling(E_fast)：快变量对慢变量的影响
    3. 弹性恢复 -γ · (E_slow - E_slow_baseline)：baseline 回归

    Attributes:
        pooling: 池化机制
        spontaneous: 自发演化函数
        elastic_coeff: 弹性恢复系数
        E_slow_baseline: 慢变量 baseline
    """

    def __init__(
        self,
        config: Optional[SlowDynamicsConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        temporal_config: Optional[MemoryTemporalConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化慢变量动力学函数

        Args:
            config: 慢变量动力学配置
            dim_config: 维度配置
            coupling_config: 耦合稳定性配置
            temporal_config: 时间配置
            device: 计算设备
        """
        super().__init__()

        # 合并配置
        self.config = config or SlowDynamicsConfig()

        if dim_config:
            self.config.slow_dim = dim_config.slow_variable_dim
            self.config.fast_dim = dim_config.fast_variable_dim

        if coupling_config:
            self.config.elastic_coeff = coupling_config.elastic_restoration_coeff
            self.config.base_coupling_coeff = coupling_config.coupling_adaptation_coeff

        if temporal_config:
            self.config.slow_update_frequency = temporal_config.slow_update_frequency

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建池化机制
        self.pooling = PoolingMechanism(
            fast_dim=self.config.fast_dim,
            slow_dim=self.config.slow_dim,
            method=self.config.pooling_method,
            attention_heads=self.config.attention_heads,
            device=self.device
        )

        # 创建自发演化函数
        self.spontaneous = SpontaneousEvolution(
            slow_dim=self.config.slow_dim,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_hidden_layers,
            activation=self.config.activation,
            spontaneous_rate=self.config.spontaneous_rate,
            device=self.device
        )

        # 弹性恢复系数
        self.elastic_coeff = self.config.elastic_coeff

        # 慢变量 baseline（初始为零，可在运行期间更新）
        self.E_slow_baseline = torch.zeros(self.config.slow_dim, device=self.device)

        # 耦合系数（自适应）
        self.coupling_coeff = self.config.base_coupling_coeff

        # 统计信息
        self.forward_calls = 0
        self.update_count = 0

        # 将模块移到设备上
        self.to(self.device)

        logger.debug(
            f"SlowDynamicsFunction created: slow_dim={self.config.slow_dim}, "
            f"elastic_coeff={self.elastic_coeff}, device={self.device}"
        )

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        E_fast: Optional[torch.Tensor] = None,
        coupling_coeff: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算慢变量的时间导数

        Args:
            t: 时间标量
            y: 慢变量状态 E_slow (slow_dim,)
            E_fast: 快变量状态（用于池化）
            coupling_coeff: 当前耦合系数（可选，默认使用内部值）

        Returns:
            慢变量的时间导数 dE_slow/dt
        """
        # 确保输入在正确设备上
        y = y.to(self.device)

        # 使用提供的耦合系数或内部值
        alpha = coupling_coeff if coupling_coeff is not None else self.coupling_coeff

        # 计算三部分贡献

        # 1. 自发演化 G_φ(E_slow)
        spontaneous_term = self.spontaneous(y)

        # 2. 快慢耦合 α(t) · Pooling(E_fast)
        if E_fast is not None:
            E_fast = E_fast.to(self.device)
            pooled_fast = self.pooling(E_fast)
            coupling_term = alpha * pooled_fast
        else:
            coupling_term = torch.zeros_like(y)

        # 3. 弹性恢复 -γ · (E_slow - E_slow_baseline)
        elastic_term = -self.elastic_coeff * (y - self.E_slow_baseline)

        # 合并三项
        dydt = spontaneous_term + coupling_term + elastic_term

        # 限制单步变化（防止过大更新）
        dydt_norm = torch.norm(dydt, dim=-1, keepdim=True)
        clip_scale = torch.where(
            dydt_norm > self.config.max_state_change,
            self.config.max_state_change / dydt_norm,
            torch.ones_like(dydt_norm)
        )
        dydt = dydt * clip_scale
        max_norm = dydt_norm.max().item()
        if max_norm > self.config.max_state_change:
            logger.debug(f"Slow state change clipped: max_original_norm={max_norm:.4f}")

        # 统计
        self.forward_calls += 1

        return dydt

    def set_baseline(self, baseline: torch.Tensor) -> None:
        """
        设置慢变量 baseline

        Args:
            baseline: 新的 baseline 值
        """
        self.E_slow_baseline = baseline.to(self.device).clone()
        logger.debug(f"Slow baseline updated: norm={torch.norm(baseline).item():.4f}")

    def update_baseline(self, E_slow: torch.Tensor, rate: float = 0.001) -> None:
        """
        更新 baseline（缓慢漂移）

        Args:
            E_slow: 当前慢变量状态
            rate: 更新速率
        """
        E_slow = E_slow.to(self.device)
        # 缓慢更新 baseline（加权平均）
        self.E_slow_baseline = (
            (1 - rate) * self.E_slow_baseline +
            rate * E_slow
        )

        # 检查 baseline 漂移
        drift = torch.norm(self.E_slow_baseline).item()
        if drift > self.config.baseline_drift_threshold:
            logger.warning(f"Baseline drift too large: {drift:.4f}")

    def set_coupling_coeff(self, coeff: float) -> None:
        """
        设置耦合系数

        Args:
            coeff: 新的耦合系数
        """
        self.coupling_coeff = coeff
        logger.debug(f"Coupling coefficient set to: {coeff:.4f}")

    def update_coupling_coeff(
        self,
        E_fast: torch.Tensor,
        adaptation_coeff: Optional[float] = None
    ) -> None:
        """
        自适应更新耦合系数

        基于快变量方差调整耦合强度：
            α(t) = α_0 · 1/(1 + β · Var(E_fast(t)))

        Args:
            E_fast: 快变量状态
            adaptation_coeff: 适应性系数 β
        """
        beta = adaptation_coeff or self.config.coupling_adaptation

        # 计算快变量方差
        variance = torch.var(E_fast).item()

        # 自适应耦合系数
        self.coupling_coeff = self.config.base_coupling_coeff / (1 + beta * variance)

        # 限制耦合系数范围
        self.coupling_coeff = max(0.001, min(0.1, self.coupling_coeff))

        logger.debug(
            f"Coupling coefficient updated: "
            f"variance={variance:.4f}, alpha={self.coupling_coeff:.4f}"
        )

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "forward_calls": self.forward_calls,
            "update_count": self.update_count,
            "baseline_norm": torch.norm(self.E_slow_baseline).item(),
            "pooling_stats": self.pooling.get_statistics(),
            "spontaneous_stats": self.spontaneous.get_statistics()
        }

    def reset_baseline(self) -> None:
        """重置 baseline"""
        self.E_slow_baseline = torch.zeros(self.config.slow_dim, device=self.device)
        logger.debug("Slow baseline reset")

    def __repr__(self) -> str:
        return (
            f"SlowDynamicsFunction(slow_dim={self.config.slow_dim}, "
            f"elastic_coeff={self.elastic_coeff}, calls={self.forward_calls})"
        )


class SlowDynamicsSystem(nn.Module):
    """
    慢变量动力学系统

    整合动力学函数和低频更新机制，提供完整的慢变量演化系统。

    主要功能：
    1. 创建和管理动力学函数
    2. 执行低频时间积分
    3. 快变量池化和耦合
    4. 弹性恢复和 baseline 维持
    5. 稳定性监测

    使用示例：
        system = SlowDynamicsSystem(config=SlowDynamicsConfig())
        system.initialize()

        # 每N快变量步更新一次慢变量
        if should_update:
            E_slow_new = system.step(E_slow, E_fast, dt_slow)
    """

    def __init__(
        self,
        config: Optional[SlowDynamicsConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        temporal_config: Optional[MemoryTemporalConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化慢变量动力学系统

        Args:
            config: 慢变量动力学配置
            dim_config: 维度配置
            coupling_config: 耦合稳定性配置
            temporal_config: 时间配置
            device: 计算设备
        """
        super().__init__()

        self.config = config or SlowDynamicsConfig()

        if dim_config:
            self.config.slow_dim = dim_config.slow_variable_dim
            self.config.fast_dim = dim_config.fast_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 动力学函数
        self.dynamics_fn: Optional[SlowDynamicsFunction] = None

        # 低频更新计数器
        self.fast_step_counter = 0
        self.slow_update_counter = 0

        # 状态记录
        self.state_history: List[torch.Tensor] = []
        self.baseline_history: List[torch.Tensor] = []

        # 初始化标志
        self._initialized = False

        logger.debug(
            f"SlowDynamicsSystem created: slow_dim={self.config.slow_dim}, "
            f"update_frequency={self.config.slow_update_frequency}"
        )

    def initialize(self) -> None:
        """初始化系统"""
        # 创建动力学函数
        self.dynamics_fn = SlowDynamicsFunction(
            config=self.config,
            device=self.device
        )
        self.add_module('dynamics_fn', self.dynamics_fn)

        self._initialized = True
        logger.debug("SlowDynamicsSystem initialized")

    def should_update_slow(self) -> bool:
        """
        判断是否应该更新慢变量

        根据低频更新策略：每 slow_update_frequency 快变量步更新1次慢变量

        Returns:
            是否应该更新
        """
        self.fast_step_counter += 1

        if self.fast_step_counter >= self.config.slow_update_frequency:
            # 达到更新频率
            self.fast_step_counter = 0
            return True

        return False

    def step(
        self,
        E_slow: torch.Tensor,
        E_fast: Optional[torch.Tensor] = None,
        dt_slow: float = 1.0,
        t: float = 0.0,
        update_baseline: bool = False
    ) -> torch.Tensor:
        """
        执行单步演化

        Args:
            E_slow: 当前慢变量状态
            E_fast: 快变量状态（用于耦合）
            dt_slow: 慢变量时间步长（相对于快变量时间尺度）
            t: 当前时间
            update_baseline: 是否更新 baseline

        Returns:
            新的慢变量状态
        """
        if not self._initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        # 确保状态在设备上
        E_slow = E_slow.to(self.device)

        # 自适应更新耦合系数
        if E_fast is not None:
            self.dynamics_fn.update_coupling_coeff(E_fast)

        # 计算时间导数
        dydt = self.dynamics_fn.forward(
            torch.tensor(t, device=self.device),
            E_slow,
            E_fast=E_fast,
            coupling_coeff=self.dynamics_fn.coupling_coeff
        )

        # 欧拉更新（慢变量使用更大的时间步长）
        E_slow_new = E_slow + dt_slow * dydt

        # 稳定性检查
        self._check_stability(E_slow_new)

        # 更新 baseline（可选）
        if update_baseline:
            self.dynamics_fn.update_baseline(E_slow_new)

        # 更新计数器
        self.slow_update_counter += 1

        return E_slow_new

    def integrate(
        self,
        E_slow: torch.Tensor,
        t_span: torch.Tensor,
        E_fast_trajectory: Optional[List[torch.Tensor]] = None,
        record_history: bool = False
    ) -> torch.Tensor:
        """
        执行多步积分

        Args:
            E_slow: 初始慢变量状态
            E_fast_trajectory: 快变量轨迹（可选，用于耦合）
            t_span: 时间点序列
            record_history: 是否记录历史

        Returns:
            状态轨迹
        """
        if not self._initialized:
            raise ValueError("System not initialized.")

        # 确保输入在设备上
        E_slow = E_slow.to(self.device)
        t_span = t_span.to(self.device)

        num_steps = t_span.shape[0]

        # 初始化轨迹
        trajectory = torch.zeros((num_steps, self.config.slow_dim), device=self.device)
        trajectory[0] = E_slow

        current_state = E_slow
        for i in range(1, num_steps):
            dt_slow = (t_span[i] - t_span[i-1]).item()

            # 获取对应的快变量（如果提供）
            E_fast = None
            if E_fast_trajectory is not None and len(E_fast_trajectory) > i:
                E_fast = E_fast_trajectory[i]

            # 执行单步
            current_state = self.step(
                current_state,
                E_fast=E_fast,
                dt_slow=dt_slow,
                t=t_span[i-1].item()
            )
            trajectory[i] = current_state

            if record_history:
                self.state_history.append(current_state.detach().cpu())
                self.baseline_history.append(
                    self.dynamics_fn.E_slow_baseline.detach().cpu()
                )

        return trajectory

    def _check_stability(self, state: torch.Tensor) -> bool:
        """检查状态稳定性"""
        # 检查 NaN 和 Inf
        if torch.isnan(state).any():
            logger.warning("NaN detected in slow variable state!")
            return False

        if torch.isinf(state).any():
            logger.warning("Inf detected in slow variable state!")
            return False

        # 检查范数（512维变量，范数20以内是合理的）
        norm = torch.norm(state).item()
        if norm > 20.0:
            logger.warning(f"Slow variable norm too large: {norm:.4e}")
            return False

        return True

    def reset_fast_counter(self) -> None:
        """重置快变量计数器"""
        self.fast_step_counter = 0
        logger.debug("Fast step counter reset")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "fast_step_counter": self.fast_step_counter,
            "slow_update_counter": self.slow_update_counter,
            "history_length": len(self.state_history),
            "dynamics_fn_stats": self.dynamics_fn.get_statistics() if self.dynamics_fn else None
        }

    def reset(self) -> None:
        """重置系统"""
        self.fast_step_counter = 0
        self.slow_update_counter = 0
        self.state_history.clear()
        self.baseline_history.clear()
        if self.dynamics_fn:
            self.dynamics_fn.reset_baseline()

        logger.debug("SlowDynamicsSystem reset")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"SlowDynamicsSystem(status={status}, "
            f"slow_dim={self.config.slow_dim}, "
            f"counter={self.fast_step_counter}/{self.config.slow_update_frequency})"
        )


def create_slow_dynamics_from_config(
    dim_config: DimensionalityConfig,
    coupling_config: CouplingStabilityConfig,
    temporal_config: MemoryTemporalConfig,
    device: Optional[str] = None
) -> SlowDynamicsSystem:
    """
    从全局配置创建慢变量动力学系统

    Args:
        dim_config: 维度配置
        coupling_config: 耦合稳定性配置
        temporal_config: 时间配置
        device: 计算设备

    Returns:
        SlowDynamicsSystem 实例
    """
    config = SlowDynamicsConfig(
        slow_dim=dim_config.slow_variable_dim,
        fast_dim=dim_config.fast_variable_dim,
        elastic_coeff=coupling_config.elastic_restoration_coeff,
        baseline_drift_threshold=coupling_config.stability_threshold / 1000,
        base_coupling_coeff=coupling_config.coupling_adaptation_coeff,
        coupling_adaptation=coupling_config.coupling_adaptation_coeff,
        slow_update_frequency=temporal_config.slow_update_frequency
    )

    system = SlowDynamicsSystem(
        config=config,
        dim_config=dim_config,
        coupling_config=coupling_config,
        temporal_config=temporal_config,
        device=device
    )

    system.initialize()
    return system