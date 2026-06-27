"""
快变量动力学系统
================

实现毫秒-秒级快速认知状态演化的动力学系统。

核心功能：
- 演化方程：dE_fast/dt = F_θ(E_fast, E_slow, X_sem, X_log, C(t), B(t), t)
- 维度：2048
- 集成所有输入信号（语义流、物理流、元认知调控、混沌注入）
- 可学习的演化函数 F_θ（使用 MLP 或 Transformer）
- 支持自适应步长积分
- 数值稳定性保障
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union
import logging
from dataclasses import dataclass, field
import math

from chronos_core.utils.config import DimensionalityConfig, MetaCognitiveConfig
from .neural_ode import DynamicsFunction

logger = logging.getLogger(__name__)


@dataclass
class FastDynamicsConfig:
    """快变量动力学配置"""

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512

    # 输入维度
    semantic_dim: int = 512
    physical_dim: int = 512
    fusion_dim: int = 1024
    meta_cognitive_dim: int = 128
    chaos_dim: int = 64  # 核心子空间维度

    # 网络架构
    hidden_dim: int = 1024
    num_hidden_layers: int = 4
    activation: str = "tanh"  # 'relu', 'tanh', 'gelu', 'silu'

    # 动力学参数
    decay_rate: float = 0.85  # 自然衰减率（大幅提高以增强稳定性）
    noise_scale: float = 0.00001  # 内部噪声强度（大幅降低以减少扰动）

    # 稳定性参数
    max_gradient_norm: float = 10.0  # 梯度裁剪
    state_norm_threshold: float = 100.0  # 状态范数阈值

    # 时间尺度
    time_scale: float = 1.0  # 相对时间尺度（相对于慢变量）

    # 是否使用注意力机制
    use_attention: bool = False
    attention_heads: int = 8

    # 傅里叶特征映射配置
    fourier_enabled: bool = True  # 是否启用傅里叶特征映射
    fourier_n_features: int = 1024  # 傅里叶特征数量
    fourier_scale: float = 2.0  # 傅里叶特征缩放因子


class EvolutionFunctionMLP(nn.Module):
    """
    MLP 形式的演化函数

    计算状态的时间导数：
        dE_fast/dt = F_θ(E_fast, E_slow, X_inputs, C(t), B(t), t)

    使用多层感知机实现非线性演化。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 4,
        activation: str = "tanh"
    ):
        """
        初始化 MLP 演化函数

        Args:
            input_dim: 输入维度（所有输入信号 + 状态）
            output_dim: 输出维度（状态导数）
            hidden_dim: 隐藏层维度
            num_hidden_layers: 隐藏层数量
            activation: 激活函数类型
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 激活函数选择
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "silu":
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Tanh()

        # 构建网络层（使用谱归一化）
        layers = []

        # 输入层
        layers.append(nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim)))
        layers.append(act_fn)

        # 隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(act_fn)

        # 输出层（不使用激活函数，允许正负输出）
        layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim, output_dim)))

        # 组合成序列
        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._init_weights()

        logger.info(
            f"EvolutionFunctionMLP created: input_dim={input_dim}, "
            f"output_dim={output_dim}, hidden_dim={hidden_dim}, "
            f"layers={num_hidden_layers}"
        )

    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.network:
            if hasattr(layer, 'weight_orig'):
                # 谱归一化层：只初始化 weight_orig（原始权重）
                # weight_u 和 weight_v 是向量，会在前向传播中自动更新
                nn.init.xavier_uniform_(layer.weight_orig, gain=1.0)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.001)
            elif isinstance(layer, nn.Linear):
                # 普通线性层（如果谱归一化被移除）
                nn.init.xavier_uniform_(layer.weight, gain=1.0)
                # 偏置初始化为小值
                nn.init.constant_(layer.bias, 0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输出

        Args:
            x: 输入向量

        Returns:
            输出向量（状态导数）
        """
        return self.network(x)

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        for i, layer in enumerate(self.network):
            if hasattr(layer, 'weight_u'):  # 检测谱归一化包装（PyTorch使用 weight_u）
                self.network[i] = nn.utils.remove_spectral_norm(layer)
        logger.info("Spectral normalization removed from EvolutionFunctionMLP")


class EvolutionFunctionTransformer(nn.Module):
    """
    Transformer 形式的演化函数

    使用注意力机制处理多源输入，适合复杂的多模态融合场景。
    """

    def __init__(
        self,
        state_dim: int,
        input_dims: Dict[str, int],
        hidden_dim: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        activation: str = "gelu"
    ):
        """
        初始化 Transformer 演化函数

        Args:
            state_dim: 状态维度
            input_dims: 各输入源的维度字典
            hidden_dim: 隐藏维度
            num_heads: 注意力头数
            num_layers: Transformer 层数
            activation: 激活函数
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # 输入投影层（将各输入源投影到统一维度，使用谱归一化）
        self.input_projections = nn.ModuleDict({
            name: nn.utils.spectral_norm(nn.Linear(dim, hidden_dim))
            for name, dim in input_dims.items()
        })

        # 状态投影（使用谱归一化）
        self.state_projection = nn.utils.spectral_norm(nn.Linear(state_dim, hidden_dim))

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 输出层（使用谱归一化）
        self.output_layer = nn.utils.spectral_norm(nn.Linear(hidden_dim, state_dim))

        # 时间编码（使用谱归一化）
        self.time_encoding = nn.utils.spectral_norm(nn.Linear(1, hidden_dim))

        logger.info(
            f"EvolutionFunctionTransformer created: state_dim={state_dim}, "
            f"hidden_dim={hidden_dim}, heads={num_heads}, layers={num_layers}"
        )

    def forward(
        self,
        state: torch.Tensor,
        inputs: Dict[str, torch.Tensor],
        time: torch.Tensor
    ) -> torch.Tensor:
        """
        计算状态导数

        Args:
            state: 当前状态
            inputs: 输入信号字典
            time: 时间标量

        Returns:
            状态导数
        """
        batch_size = state.shape[0] if state.dim() > 1 else 1

        # 投影各输入源
        projected_inputs = []
        for name, projection in self.input_projections.items():
            if name in inputs:
                projected = projection(inputs[name])
                projected_inputs.append(projected)

        # 投影状态
        state_projected = self.state_projection(state)

        # 时间编码
        time_encoded = self.time_encoding(time.unsqueeze(-1))

        # 组合成序列 (batch_size, num_sources, hidden_dim)
        # 包括：状态、时间、各输入源
        sequence = torch.stack([state_projected, time_encoded] + projected_inputs, dim=1)

        # Transformer 处理
        encoded = self.transformer_encoder(sequence)

        # 取状态位置作为输出（或平均池化）
        # 这里使用平均池化
        pooled = encoded.mean(dim=1)

        # 输出状态导数
        output = self.output_layer(pooled)

        return output

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        # 移除输入投影层的谱归一化
        for name, projection in self.input_projections.items():
            if hasattr(projection, 'weight_u'):  # 检测谱归一化包装（PyTorch使用 weight_u）
                self.input_projections[name] = nn.utils.remove_spectral_norm(projection)

        # 移除其他层的谱归一化
        if hasattr(self.state_projection, 'weight_u'):
            self.state_projection = nn.utils.remove_spectral_norm(self.state_projection)
        if hasattr(self.output_layer, 'weight_u'):
            self.output_layer = nn.utils.remove_spectral_norm(self.output_layer)
        if hasattr(self.time_encoding, 'weight_u'):
            self.time_encoding = nn.utils.remove_spectral_norm(self.time_encoding)

        logger.info("Spectral normalization removed from EvolutionFunctionTransformer")


class FastDynamicsFunction(DynamicsFunction):
    """
    快变量动力学函数

    实现 Neural ODE 的动力学函数接口：
        dE_fast/dt = F_θ(E_fast, E_slow, X_inputs, C(t), B(t), t)

    集成所有输入信号：
    - E_fast: 快变量状态（2048维）
    - E_slow: 慢变量状态（512维）
    - X_sem: 语义流（512维）
    - X_log: 物理流（512维）
    - X_fused: 融合表征（1024维）
    - C(t): 元认知调控信号（128维）
    - B(t): 混沌注入信号（64维核心子空间）

    使用 MLP 或 Transformer 实现可学习的演化函数 F_θ。
    """

    def __init__(
        self,
        config: Optional[FastDynamicsConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化快变量动力学函数

        Args:
            config: 快变量动力学配置
            dim_config: 维度配置（来自全局配置）
            meta_config: 元认知配置（来自全局配置）
            device: 计算设备
        """
        super().__init__()

        # 合并配置
        self.config = config or FastDynamicsConfig()

        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim
            self.config.semantic_dim = dim_config.semantic_dim
            self.config.physical_dim = dim_config.physical_dim
            self.config.fusion_dim = dim_config.fusion_dim
            self.config.chaos_dim = dim_config.core_subspace_dim

        if meta_config:
            self.config.meta_cognitive_dim = meta_config.control_output_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化傅里叶特征映射
        if self.config.fourier_enabled:
            # 创建随机固定矩阵 B，用于傅里叶特征映射
            # B 的形状: [n_features, fusion_dim]
            fourier_B = torch.randn(
                self.config.fourier_n_features,
                self.config.fusion_dim
            ) * self.config.fourier_scale
            # 注册为 buffer（不参与训练，但会随模型移动到设备）
            self.register_buffer('fourier_B', fourier_B)
            logger.info(
                f"Fourier feature mapping enabled: n_features={self.config.fourier_n_features}, "
                f"scale={self.config.fourier_scale}"
            )
        else:
            self.fourier_B = None

        # 计算总输入维度
        # E_fast + E_slow + X_sem + X_log + X_fused(or fourier_features) + C(t) + B(t) + t
        if self.config.fourier_enabled:
            # 使用傅里叶特征映射时，fusion_dim 被 2*n_features 替代
            fusion_input_dim = self.config.fourier_n_features * 2
        else:
            fusion_input_dim = self.config.fusion_dim

        self.total_input_dim = (
            self.config.fast_dim +
            self.config.slow_dim +
            self.config.semantic_dim +
            self.config.physical_dim +
            fusion_input_dim +
            self.config.meta_cognitive_dim +
            self.config.chaos_dim +
            1  # 时间维度
        )

        # 创建演化函数网络
        if self.config.use_attention:
            # 使用 Transformer
            input_dims = {
                'semantic': self.config.semantic_dim,
                'physical': self.config.physical_dim,
                'fusion': self.config.fusion_dim,
                'meta_cognitive': self.config.meta_cognitive_dim,
                'chaos': self.config.chaos_dim,
                'slow': self.config.slow_dim
            }
            self.evolution_fn = EvolutionFunctionTransformer(
                state_dim=self.config.fast_dim,
                input_dims=input_dims,
                hidden_dim=self.config.hidden_dim,
                num_heads=self.config.attention_heads,
                num_layers=2,
                activation=self.config.activation
            )
        else:
            # 使用 MLP
            self.evolution_fn = EvolutionFunctionMLP(
                input_dim=self.total_input_dim,
                output_dim=self.config.fast_dim,
                hidden_dim=self.config.hidden_dim,
                num_hidden_layers=self.config.num_hidden_layers,
                activation=self.config.activation
            )

        # 衰减项（自然衰减，使用谱归一化）
        self.decay_layer = nn.utils.spectral_norm(
            nn.Linear(self.config.fast_dim, self.config.fast_dim, bias=False)
        )
        # 初始化衰减层的 weight_orig（原始权重）
        # 设置为负值，产生衰减效果
        # weight_u 和 weight_v 会在前向传播中自动更新
        nn.init.constant_(self.decay_layer.weight_orig, -self.config.decay_rate)

        # 将网络移到设备上
        self.to(self.device)

        # 统计信息
        self.forward_calls = 0

        logger.info(
            f"FastDynamicsFunction created: fast_dim={self.config.fast_dim}, "
            f"input_dim={self.total_input_dim}, device={self.device}"
        )

    def forward(
        self,
        t: torch.Tensor,
        y: torch.Tensor,
        E_slow: Optional[torch.Tensor] = None,
        X_sem: Optional[torch.Tensor] = None,
        X_log: Optional[torch.Tensor] = None,
        X_fused: Optional[torch.Tensor] = None,
        C_meta: Optional[torch.Tensor] = None,
        B_chaos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算快变量的时间导数

        Args:
            t: 时间标量
            y: 快变量状态 E_fast (fast_dim,) 或 (batch_size, fast_dim)
            E_slow: 慢变量状态
            X_sem: 语义流输入
            X_log: 物理流输入
            X_fused: 融合表征
            C_meta: 元认知调控信号
            B_chaos: 混沌注入信号

        Returns:
            快变量的时间导数 dE_fast/dt
        """
        # 确保输入在正确设备上
        y = y.to(self.device)

        # 处理缺失输入（用零填充）
        batch_size = y.shape[0] if y.dim() > 1 else 1

        # 扩展时间到批次维度
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.dim() == 0 else t[0].item()
        else:
            t_val = t

        # 构建输入向量
        input_parts = []

        # 1. 快变量状态
        input_parts.append(y)

        # 2. 慢变量状态
        if E_slow is not None:
            input_parts.append(E_slow.to(self.device))
        else:
            input_parts.append(torch.zeros(batch_size, self.config.slow_dim, device=self.device))

        # 3. 语义流
        if X_sem is not None:
            input_parts.append(X_sem.to(self.device))
        else:
            input_parts.append(torch.zeros(batch_size, self.config.semantic_dim, device=self.device))

        # 4. 物理流
        if X_log is not None:
            input_parts.append(X_log.to(self.device))
        else:
            input_parts.append(torch.zeros(batch_size, self.config.physical_dim, device=self.device))

        # 5. 融合表征（应用傅里叶特征映射）
        if X_fused is not None:
            X_fused = X_fused.to(self.device)
            if self.config.fourier_enabled:
                # 应用傅里叶特征映射
                # X_fused: [batch_size, fusion_dim]
                # fourier_B: [n_features, fusion_dim]
                # B @ X_fused.T: [n_features, batch_size]
                # 转置后: [batch_size, n_features]
                projected = torch.matmul(self.fourier_B, X_fused.T).T  # [batch_size, n_features]
                gamma_X = torch.cat([
                    torch.cos(projected),
                    torch.sin(projected)
                ], dim=-1)  # [batch_size, 2*n_features]
                input_parts.append(gamma_X)
            else:
                input_parts.append(X_fused)
        else:
            # 缺失输入用零填充
            if self.config.fourier_enabled:
                input_parts.append(torch.zeros(batch_size, self.config.fourier_n_features * 2, device=self.device))
            else:
                input_parts.append(torch.zeros(batch_size, self.config.fusion_dim, device=self.device))

        # 6. 元认知调控信号
        if C_meta is not None:
            input_parts.append(C_meta.to(self.device))
        else:
            input_parts.append(torch.zeros(batch_size, self.config.meta_cognitive_dim, device=self.device))

        # 7. 混沌注入信号
        if B_chaos is not None:
            # 确保混沌信号维度匹配
            if B_chaos.shape[-1] < self.config.chaos_dim:
                # 扩展维度
                padding = torch.zeros(
                    batch_size,
                    self.config.chaos_dim - B_chaos.shape[-1],
                    device=self.device
                )
                B_chaos = torch.cat([B_chaos.to(self.device), padding], dim=-1)
            elif B_chaos.shape[-1] > self.config.chaos_dim:
                # 截断（支持批量）
                B_chaos = B_chaos[..., :self.config.chaos_dim].to(self.device)
            input_parts.append(B_chaos)
        else:
            input_parts.append(torch.zeros(batch_size, self.config.chaos_dim, device=self.device))

        # 8. 时间
        t_tensor = torch.tensor([[t_val]], device=self.device).expand(batch_size, -1)
        input_parts.append(t_tensor)

        # 拼接所有输入
        if batch_size == 1 and y.dim() == 1:
            # 单样本情况
            input_vector = torch.cat([part.flatten() if part.dim() > 1 else part for part in input_parts], dim=0)
        else:
            # 批量情况
            input_vector = torch.cat(input_parts, dim=-1)

        # 计算演化函数输出（非线性部分）
        F_output = self.evolution_fn(input_vector)

        # 计算衰减项（线性部分）
        decay_output = self.decay_layer(y)

        # 合并输出
        # dE_fast/dt = F_θ(...) + decay * E_fast
        dydt = F_output + decay_output

        # 添加范数依赖衰减（更强的稳定化）
        # 当范数超过阈值的一半时，添加额外的衰减
        norm = torch.norm(y, dim=-1, keepdim=True)
        threshold_half = self.config.state_norm_threshold * 0.5
        extra_decay_factor = torch.where(
            norm > threshold_half,
            (norm - threshold_half) / threshold_half,
            torch.zeros_like(norm)
        )
        extra_decay_factor = torch.clamp(extra_decay_factor, min=0.0, max=2.0)
        dydt = dydt - extra_decay_factor * y

        # 添加内部噪声（可选）
        if self.config.noise_scale > 0:
            noise = torch.randn_like(dydt) * self.config.noise_scale
            dydt = dydt + noise

        # 梯度裁剪（防止过大导数）
        dydt_norm = torch.norm(dydt, dim=-1, keepdim=True)
        clip_scale = torch.where(
            dydt_norm > self.config.max_gradient_norm,
            self.config.max_gradient_norm / dydt_norm,
            torch.ones_like(dydt_norm)
        )
        dydt = dydt * clip_scale
        max_norm = dydt_norm.max().item()
        if max_norm > self.config.max_gradient_norm:
            logger.debug(f"Gradient clipped: max_original_norm={max_norm:.4f}")

        # 统计
        self.forward_calls += 1

        return dydt

    def set_inputs(
        self,
        E_slow: torch.Tensor,
        X_sem: Optional[torch.Tensor] = None,
        X_log: Optional[torch.Tensor] = None,
        X_fused: Optional[torch.Tensor] = None,
        C_meta: Optional[torch.Tensor] = None,
        B_chaos: Optional[torch.Tensor] = None
    ) -> None:
        """
        设置外部输入（用于连续积分期间的输入更新）

        Args:
            E_slow: 慢变量状态
            X_sem: 语义流
            X_log: 物理流
            X_fused: 融合表征
            C_meta: 元认知调控
            B_chaos: 混沌注入
        """
        # 存储输入（将在 forward 时使用）
        self._cached_E_slow = E_slow.to(self.device)
        self._cached_X_sem = X_sem.to(self.device) if X_sem is not None else None
        self._cached_X_log = X_log.to(self.device) if X_log is not None else None
        self._cached_X_fused = X_fused.to(self.device) if X_fused is not None else None
        self._cached_C_meta = C_meta.to(self.device) if C_meta is not None else None
        self._cached_B_chaos = B_chaos.to(self.device) if B_chaos is not None else None

    def forward_with_cached_inputs(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        使用缓存的输入计算导数

        Args:
            t: 时间
            y: 状态

        Returns:
            状态导数
        """
        return self.forward(
            t, y,
            E_slow=self._cached_E_slow,
            X_sem=self._cached_X_sem,
            X_log=self._cached_X_log,
            X_fused=self._cached_X_fused,
            C_meta=self._cached_C_meta,
            B_chaos=self._cached_B_chaos
        )

    def reset_cache(self) -> None:
        """重置缓存的输入"""
        self._cached_E_slow = None
        self._cached_X_sem = None
        self._cached_X_log = None
        self._cached_X_fused = None
        self._cached_C_meta = None
        self._cached_B_chaos = None

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        # 移除演化函数中的谱归一化
        if hasattr(self.evolution_fn, 'remove_spectral_norm'):
            self.evolution_fn.remove_spectral_norm()

        # 移除衰减层的谱归一化
        if hasattr(self.decay_layer, 'weight_u'):  # 检测谱归一化包装（PyTorch使用 weight_u）
            self.decay_layer = nn.utils.remove_spectral_norm(self.decay_layer)

        logger.info("Spectral normalization removed from FastDynamicsFunction")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "forward_calls": self.forward_calls,
            "fast_dim": self.config.fast_dim,
            "total_input_dim": self.total_input_dim,
            "decay_rate": self.config.decay_rate,
            "noise_scale": self.config.noise_scale,
            "fourier_enabled": self.config.fourier_enabled,
            "device": self.device
        }

        # 添加傅里叶特征映射信息
        if self.config.fourier_enabled:
            stats["fourier_n_features"] = self.config.fourier_n_features
            stats["fourier_scale"] = self.config.fourier_scale

        # 网络参数统计
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stats["total_parameters"] = total_params
        stats["trainable_parameters"] = trainable_params

        return stats

    def __repr__(self) -> str:
        return (
            f"FastDynamicsFunction(fast_dim={self.config.fast_dim}, "
            f"input_dim={self.total_input_dim}, calls={self.forward_calls})"
        )


class FastDynamicsSystem(nn.Module):
    """
    快变量动力学系统

    整合动力学函数和 ODE 求解器，提供完整的快变量演化系统。

    主要功能：
    1. 创建和管理动力学函数
    2. 执行时间积分
    3. 处理外部输入
    4. 稳定性监测
    5. 状态记录

    使用示例：
        system = FastDynamicsSystem(config=FastDynamicsConfig())
        system.initialize()

        # 单步更新
        E_fast_new = system.step(E_fast, inputs, dt)

        # 多步积分
        trajectory = system.integrate(E_fast, inputs, t_span)
    """

    def __init__(
        self,
        config: Optional[FastDynamicsConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化快变量动力学系统

        Args:
            config: 快变量动力学配置
            dim_config: 维度配置
            meta_config: 元认知配置
            device: 计算设备
        """
        super().__init__()

        self.config = config or FastDynamicsConfig()

        self.dim_config = dim_config
        self.meta_config = meta_config

        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim
            self.config.semantic_dim = dim_config.semantic_dim
            self.config.physical_dim = dim_config.physical_dim
            self.config.fusion_dim = dim_config.fusion_dim
            self.config.chaos_dim = dim_config.core_subspace_dim

        if meta_config:
            self.config.meta_cognitive_dim = meta_config.control_output_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 动力学函数
        self.dynamics_fn: Optional[FastDynamicsFunction] = None

        # 状态记录
        self.state_history: list = []
        self.time_history: list = []

        # 初始化标志
        self._initialized = False

        # 日志频率控制
        self._clip_log_interval = 100  # 每100步输出一次范数裁剪日志
        self._clip_log_counter = 0
        self._stability_log_interval = 50  # 每50步输出一次稳定性警告日志
        self._stability_log_counter = 0

        logger.info(
            f"FastDynamicsSystem created: fast_dim={self.config.fast_dim}, "
            f"device={self.device}"
        )

    def initialize(self) -> None:
        """初始化系统"""
        # 创建动力学函数
        self.dynamics_fn = FastDynamicsFunction(
            config=self.config,
            device=self.device
        )
        self.add_module('dynamics_fn', self.dynamics_fn)

        self._initialized = True
        logger.info("FastDynamicsSystem initialized")

    def step(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        dt: float = 0.01,
        t: float = 0.0
    ) -> torch.Tensor:
        """
        执行单步演化

        Args:
            E_fast: 当前快变量状态
            E_slow: 慢变量状态
            inputs: 输入信号字典
            dt: 时间步长
            t: 当前时间

        Returns:
            新的快变量状态
        """
        if not self._initialized:
            raise ValueError("System not initialized. Call initialize() first.")

        # 确保状态在设备上
        E_fast = E_fast.to(self.device)
        E_slow = E_slow.to(self.device)

        # 处理输入
        if inputs is None:
            inputs = {}

        # 使用简单的欧拉法积分（或可以改用更高级方法）
        # dE_fast/dt = F(E_fast, E_slow, inputs, t)
        dydt = self.dynamics_fn.forward(
            torch.tensor(t, device=self.device),
            E_fast,
            E_slow=E_slow,
            X_sem=inputs.get('X_sem'),
            X_log=inputs.get('X_log'),
            X_fused=inputs.get('X_fused'),
            C_meta=inputs.get('C_meta'),
            B_chaos=inputs.get('B_chaos')
        )

        # 欧拉更新
        E_fast_new = E_fast + dt * dydt

        # 状态范数裁剪（防止发散）
        norm = torch.norm(E_fast_new).item()
        if norm > self.config.state_norm_threshold:
            scale = self.config.state_norm_threshold / norm
            E_fast_new = E_fast_new * scale
            # 限制日志输出频率
            self._clip_log_counter += 1
            if self._clip_log_counter >= self._clip_log_interval:
                logger.info(f"State norm clipped: {norm:.4e} -> {self.config.state_norm_threshold}")
                self._clip_log_counter = 0

        # 稳定性检查
        self._check_stability(E_fast_new)

        return E_fast_new

    def integrate(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        t_span: torch.Tensor,
        inputs: Optional[Dict[str, torch.Tensor]] = None,
        record_history: bool = False
    ) -> torch.Tensor:
        """
        执行多步积分

        Args:
            E_fast: 初始快变量状态
            E_slow: 慢变量状态（在整个积分期间固定）
            inputs: 输入信号字典
            t_span: 时间点序列
            record_history: 是否记录历史

        Returns:
            状态轨迹
        """
        if not self._initialized:
            raise ValueError("System not initialized.")

        # 确保输入在设备上
        E_fast = E_fast.to(self.device)
        E_slow = E_slow.to(self.device)
        t_span = t_span.to(self.device)

        # 设置缓存输入
        if inputs:
            self.dynamics_fn.set_inputs(
                E_slow=E_slow,
                X_sem=inputs.get('X_sem'),
                X_log=inputs.get('X_log'),
                X_fused=inputs.get('X_fused'),
                C_meta=inputs.get('C_meta'),
                B_chaos=inputs.get('B_chaos')
            )
        else:
            self.dynamics_fn.set_inputs(E_slow=E_slow)

        # 手动积分（因为 torchdiffeq 的接口需要专门的动力学函数）
        num_steps = t_span.shape[0]
        trajectory = torch.zeros((num_steps, self.config.fast_dim), device=self.device)
        trajectory[0] = E_fast

        current_state = E_fast
        for i in range(1, num_steps):
            dt = (t_span[i] - t_span[i-1]).item()
            current_state = self.step(
                current_state,
                E_slow,
                inputs,
                dt,
                t_span[i-1].item()
            )
            trajectory[i] = current_state

            if record_history:
                self.state_history.append(current_state.detach().cpu())
                self.time_history.append(t_span[i].item())

        return trajectory

    def _check_stability(self, state: torch.Tensor) -> bool:
        """检查状态稳定性"""
        # 检查 NaN 和 Inf（始终输出）
        if torch.isnan(state).any():
            logger.warning("NaN detected in fast variable state!")
            return False

        if torch.isinf(state).any():
            logger.warning("Inf detected in fast variable state!")
            return False

        # 检查范数（限制日志频率）
        norm = torch.norm(state).item()
        if norm > self.config.state_norm_threshold:
            self._stability_log_counter += 1
            if self._stability_log_counter >= self._stability_log_interval:
                logger.warning(f"Fast variable norm too large: {norm:.4e}")
                self._stability_log_counter = 0
            return False

        return True

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = {
            "initialized": self._initialized,
            "fast_dim": self.config.fast_dim,
            "slow_dim": self.config.slow_dim,
            "history_length": len(self.state_history),
            "dynamics_fn_stats": self.dynamics_fn.get_statistics() if self.dynamics_fn else None
        }

        return stats

    def reset(self) -> None:
        """重置系统"""
        self.state_history.clear()
        self.time_history.clear()
        if self.dynamics_fn:
            self.dynamics_fn.reset_cache()

        # 重置日志计数器
        self._clip_log_counter = 0
        self._stability_log_counter = 0

        logger.info("FastDynamicsSystem reset")

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        if self.dynamics_fn and hasattr(self.dynamics_fn, 'remove_spectral_norm'):
            self.dynamics_fn.remove_spectral_norm()
        logger.info("Spectral normalization removed from FastDynamicsSystem")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"FastDynamicsSystem(status={status}, "
            f"fast_dim={self.config.fast_dim})"
        )


def create_fast_dynamics_from_config(
    dim_config: DimensionalityConfig,
    meta_config: MetaCognitiveConfig,
    device: Optional[str] = None
) -> FastDynamicsSystem:
    """
    从全局配置创建快变量动力学系统

    Args:
        dim_config: 维度配置
        meta_config: 元认知配置
        device: 计算设备

    Returns:
        FastDynamicsSystem 实例
    """
    config = FastDynamicsConfig(
        fast_dim=dim_config.fast_variable_dim,
        slow_dim=dim_config.slow_variable_dim,
        semantic_dim=dim_config.semantic_dim,
        physical_dim=dim_config.physical_dim,
        fusion_dim=dim_config.fusion_dim,
        meta_cognitive_dim=meta_config.l2_hidden_dim,
        chaos_dim=dim_config.core_subspace_dim
    )

    system = FastDynamicsSystem(
        config=config,
        dim_config=dim_config,
        meta_config=meta_config,
        device=device
    )

    system.initialize()
    return system