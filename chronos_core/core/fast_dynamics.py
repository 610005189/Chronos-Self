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
from typing import Optional, Dict, Tuple, Union, List
import logging
from dataclasses import dataclass, field
import math
from enum import Enum

from chronos_core.utils.config import DimensionalityConfig, MetaCognitiveConfig
from .neural_ode import DynamicsFunction
from .state_controller import StateController, StateMode, StateParameters

logger = logging.getLogger(__name__)


@dataclass
class FastDynamicsConfig:
    """快变量动力学配置"""

    # 状态模式
    state_mode: str = "work"  # 可选值: "rest", "work", "explore"

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
    damping_coeff: float = 0.0  # 显式阻尼系数（方案B，已废弃）
    gamma: float = 0.5  # 线性耗散系数（专家建议：与约束Lipschitz的MLP配合使用）
    dynamics_scale: float = 1.0  # 演化函数输出缩放因子（方案C：直接控制混沌强度）
    noise_scale: float = 0.00001  # 内部噪声强度（大幅降低以减少扰动）

    # E/I 平衡网络参数（方案 A）
    ei_ratio: float = 4.0  # E维数 / I维数，默认 1638:410（2048/5）
    alpha: float = 0.1  # 抑制反馈增益
    e_target: float = 0.0  # 兴奋目标均值
    use_ei_balance: bool = False  # 开关（默认关闭，保持兼容性）

    # 稳定性参数
    max_gradient_norm: float = 10.0  # 梯度裁剪
    state_norm_threshold: float = 100.0  # 状态范数阈值
    state_norm_clip: float = 0.0  # 状态范数截断（>0时启用，每步将状态范数截断到此值）

    # 逐层谱约束参数（方案C）
    target_spectral_norm: float = 1.9  # 每层权重的目标谱范数（控制Lipschitz常数）

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

    支持两种限速机制：
        1. 激活函数斜率限制（方案B）：scaled_tanh(x) = tanh(x/(1+gamma))
        2. 逐层谱约束（方案C）：每层权重谱范数约束到 target_spectral_norm

    方案C确保每层的 Lipschitz 常数 ≤ target_spectral_norm，
    解决全局 gamma 缩放无法均匀控制各层谱范数的问题。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        num_hidden_layers: int = 4,
        activation: str = "tanh",
        target_spectral_norm: float = 1.0
    ):
        """
        初始化 MLP 演化函数

        Args:
            input_dim: 输入维度（所有输入信号 + 状态）
            output_dim: 输出维度（状态导数）
            hidden_dim: 隐藏层维度
            num_hidden_layers: 隐藏层数量
            activation: 激活函数类型
            target_spectral_norm: 目标谱范数（默认1.0，确保每层Lipschitz≤1）
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.target_spectral_norm = target_spectral_norm

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

        self.activation = act_fn

        # 分离线性层和激活层（便于在 forward 中插入缩放）
        self.linear_layers = nn.ModuleList()

        # 输入层
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))

        # 隐藏层
        for _ in range(num_hidden_layers):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出层（不使用激活函数，允许正负输出）
        self.linear_layers.append(nn.Linear(hidden_dim, output_dim))

        # gamma 参数（激活函数斜率限制系数）
        # gamma = 0 时无缩放，gamma > 0 时激活函数斜率上限为 1/(1+gamma)
        self.gamma = 0.0

        # 逐层谱约束缩放因子（缓存）
        # layer_scale[i] = 1.0 / spectral_norm_i if spectral_norm_i > target else 1.0
        self.layer_scales: Optional[List[float]] = None

        # 初始化权重
        self._init_weights()

        # 计算每层谱范数并生成缩放因子
        self._compute_layer_scales()

        logger.debug(
            f"EvolutionFunctionMLP created: input_dim={input_dim}, "
            f"output_dim={output_dim}, hidden_dim={hidden_dim}, "
            f"layers={num_hidden_layers}, target_spectral_norm={target_spectral_norm}"
        )

    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.linear_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1.4)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.001)

    def _compute_layer_scales(self):
        """
        计算每层权重矩阵的谱范数并生成缩放因子
        
        使用幂迭代法估计谱范数，然后生成缩放因子：
        - 如果谱范数 > target_spectral_norm，缩放因子 = target / spectral_norm
        - 否则，缩放因子 = 1.0（不缩放）
        
        缩放因子缓存到 self.layer_scales，在 forward 中应用。
        """
        self.layer_scales = []
        
        for i, layer in enumerate(self.linear_layers):
            if isinstance(layer, nn.Linear):
                # 使用幂迭代法估计谱范数（增加迭代次数提高精度）
                weight = layer.weight.data
                spectral_norm = self._power_iteration_spectral_norm(weight, num_iterations=20)
                
                # 计算缩放因子：约束谱范数到目标值（加小容差）
                tolerance = 1e-4
                if spectral_norm > self.target_spectral_norm - tolerance:
                    # 确保缩放后谱范数 <= target
                    scale = self.target_spectral_norm / (spectral_norm + 1e-6)
                else:
                    scale = 1.0
                
                self.layer_scales.append(scale)
                
                # 直接对权重应用缩放（原地修改）
                with torch.no_grad():
                    layer.weight.data *= scale
                
                # 验证缩放后的谱范数
                scaled_norm = self._power_iteration_spectral_norm(layer.weight.data, num_iterations=10)
                
                logger.debug(
                    f"Layer {i}: original_norm={spectral_norm:.4f}, "
                    f"scale={scale:.4f}, scaled_norm={scaled_norm:.4f}"
                )
        
        # 计算总 Lipschitz 上限（所有层谱范数的乘积 * 激活函数斜率上限）
        total_lipschitz = 1.0
        for scale in self.layer_scales:
            total_lipschitz *= self.target_spectral_norm  # 缩放后谱范数≤target
        
        # 激活函数斜率上限（tanh 最大斜率为1）
        # 如果 gamma > 0，斜率上限为 1/(1+gamma)
        act_slope_limit = 1.0 / (1.0 + self.gamma) if self.gamma > 0 else 1.0
        num_act_layers = len(self.linear_layers) - 1  # 输出层无激活
        total_lipschitz *= act_slope_limit ** num_act_layers
        
        logger.info(
            f"Layer spectral constraints applied: total_Lipschitz≈{total_lipschitz:.4f}, "
            f"num_layers={len(self.linear_layers)}, target_spectral_norm={self.target_spectral_norm}"
        )

    def _power_iteration_spectral_norm(self, weight: torch.Tensor, num_iterations: int = 5) -> float:
        """
        使用幂迭代法估计权重矩阵的谱范数（最大奇异值）
        
        Args:
            weight: 权重矩阵 (out_features, in_features)
            num_iterations: 幂迭代次数
            
        Returns:
            估计的谱范数
        """
        # 初始化随机向量
        v = torch.randn(weight.shape[1], device=weight.device)
        v = v / torch.norm(v)
        
        for _ in range(num_iterations):
            # v -> u: W @ v
            u = weight @ v
            u_norm = torch.norm(u)
            if u_norm < 1e-10:
                return 0.0
            u = u / u_norm
            
            # u -> v: W.T @ u
            v = weight.t() @ u
            v_norm = torch.norm(v)
            if v_norm < 1e-10:
                return 0.0
            v = v / v_norm
        
        # 谱范数 = ||W @ v|| = ||W.T @ u||
        spectral_norm = torch.norm(weight @ v).item()
        return spectral_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        计算输出（支持两种限速机制）

        方案C（逐层谱约束）已在初始化时应用：
            - 每层权重已缩放，谱范数 ≤ target_spectral_norm
            - 确保各层 Lipschitz 常数均匀控制

        方案B（激活函数斜率限制）在 forward 中应用：
            - scaled_tanh(x) = tanh(x / (1 + gamma))
            - 进一步限制激活函数的有效斜率

        组合效果：
            - 总 Lipschitz ≈ (target_spectral_norm)^n * (1/(1+gamma))^n
            - 逐层谱约束解决"闸门效应"
            - gamma 参数提供额外的连续调节能力

        Args:
            x: 输入向量

        Returns:
            输出向量（状态导数）
        """
        # 计算缩放因子：激活函数输入除以 (1+gamma)
        # 这样 tanh(x/(1+gamma)) 的斜率上限为 1/(1+gamma)
        # 总 Lipschitz 约为 L_total ≈ (1/(1+gamma))^n * product(W_i)
        act_scale = 1.0 / (1.0 + self.gamma) if self.gamma > 0 else 1.0

        # 前向传播：每层激活前应用输入缩放
        h = x

        # 输入层
        h = self.linear_layers[0](h)
        h = self.activation(h * act_scale)

        # 隐藏层
        for i in range(1, len(self.linear_layers) - 1):
            h = self.linear_layers[i](h)
            h = self.activation(h * act_scale)

        # 输出层（不应用激活）
        output = self.linear_layers[-1](h)

        return output

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        for i, layer in enumerate(self.linear_layers):
            if hasattr(layer, 'weight_u'):
                self.linear_layers[i] = nn.utils.remove_spectral_norm(layer)
        logger.debug("Spectral normalization removed from EvolutionFunctionMLP")


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

        logger.debug(
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

        logger.debug("Spectral normalization removed from EvolutionFunctionTransformer")


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
            logger.debug(
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
                activation=self.config.activation,
                target_spectral_norm=self.config.target_spectral_norm
            )
            # 设置内部限速 gamma 参数
            if hasattr(self.evolution_fn, 'gamma'):
                self.evolution_fn.gamma = self.config.gamma

        # 衰减项（自然衰减，不使用谱归一化，因为需要保持精确的衰减值）
        self.decay_layer = nn.Linear(self.config.fast_dim, self.config.fast_dim, bias=False)
        # 初始化为对角矩阵，产生衰减效果（每个维度只受自身衰减影响）
        with torch.no_grad():
            self.decay_layer.weight.zero_()
            self.decay_layer.weight.diagonal().fill_(-self.config.decay_rate)

        # 将网络移到设备上
        self.to(self.device)

        # 统计信息
        self.forward_calls = 0

        logger.debug(
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

        # E/I 平衡网络分支
        # 当 alpha 极小时，退化为非 E/I 模式以保证一致性
        use_ei_effective = self.config.use_ei_balance and self.config.alpha > 1e-6
        if use_ei_effective:
            # === E/I 平衡网络动力学 ===
            # 计算兴奋和抑制群体的维度
            d_E = int(self.config.fast_dim * self.config.ei_ratio / (self.config.ei_ratio + 1))
            d_I = self.config.fast_dim - d_E

            # 分割状态
            if y.dim() == 1:
                y_E = y[:d_E]
                y_I = y[d_E:]
            else:
                y_E = y[:, :d_E]
                y_I = y[:, d_E:]

            # 计算 E 群体均值和抑制反馈
            E_mean = y_E.mean(dim=-1, keepdim=True)
            inhibition = self.config.alpha * (E_mean - self.config.e_target)

            # 构建 E 群体的输入向量
            input_parts_E = []
            input_parts_E.append(y_E)  # E 群体状态

            # 共享输入（慢变量、语义流、物理流、融合表征、元认知调控）
            if E_slow is not None:
                input_parts_E.append(E_slow.to(self.device))
            else:
                input_parts_E.append(torch.zeros(batch_size, self.config.slow_dim, device=self.device))

            if X_sem is not None:
                input_parts_E.append(X_sem.to(self.device))
            else:
                input_parts_E.append(torch.zeros(batch_size, self.config.semantic_dim, device=self.device))

            if X_log is not None:
                input_parts_E.append(X_log.to(self.device))
            else:
                input_parts_E.append(torch.zeros(batch_size, self.config.physical_dim, device=self.device))

            if X_fused is not None:
                X_fused = X_fused.to(self.device)
                if self.config.fourier_enabled:
                    projected = torch.matmul(self.fourier_B, X_fused.T).T
                    gamma_X = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
                    input_parts_E.append(gamma_X)
                else:
                    input_parts_E.append(X_fused)
            else:
                if self.config.fourier_enabled:
                    input_parts_E.append(torch.zeros(batch_size, self.config.fourier_n_features * 2, device=self.device))
                else:
                    input_parts_E.append(torch.zeros(batch_size, self.config.fusion_dim, device=self.device))

            if C_meta is not None:
                input_parts_E.append(C_meta.to(self.device))
            else:
                input_parts_E.append(torch.zeros(batch_size, self.config.meta_cognitive_dim, device=self.device))

            # 混沌注入仅施加于 E 群体
            if B_chaos is not None:
                if B_chaos.shape[-1] < self.config.chaos_dim:
                    padding = torch.zeros(
                        batch_size,
                        self.config.chaos_dim - B_chaos.shape[-1],
                        device=self.device
                    )
                    B_chaos = torch.cat([B_chaos.to(self.device), padding], dim=-1)
                elif B_chaos.shape[-1] > self.config.chaos_dim:
                    B_chaos = B_chaos[..., :self.config.chaos_dim].to(self.device)
                input_parts_E.append(B_chaos)
            else:
                input_parts_E.append(torch.zeros(batch_size, self.config.chaos_dim, device=self.device))

            # 时间
            t_tensor = torch.tensor([[t_val]], device=self.device).expand(batch_size, -1)
            input_parts_E.append(t_tensor)

            # 构建 I 群体的输入向量
            input_parts_I = []
            input_parts_I.append(y_I)  # I 群体状态

            # I 群体也接收共享输入（但不接收混沌注入）
            if E_slow is not None:
                input_parts_I.append(E_slow.to(self.device))
            else:
                input_parts_I.append(torch.zeros(batch_size, self.config.slow_dim, device=self.device))

            if X_sem is not None:
                input_parts_I.append(X_sem.to(self.device))
            else:
                input_parts_I.append(torch.zeros(batch_size, self.config.semantic_dim, device=self.device))

            if X_log is not None:
                input_parts_I.append(X_log.to(self.device))
            else:
                input_parts_I.append(torch.zeros(batch_size, self.config.physical_dim, device=self.device))

            if X_fused is not None:
                X_fused = X_fused.to(self.device)
                if self.config.fourier_enabled:
                    projected = torch.matmul(self.fourier_B, X_fused.T).T
                    gamma_X = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
                    input_parts_I.append(gamma_X)
                else:
                    input_parts_I.append(X_fused)
            else:
                if self.config.fourier_enabled:
                    input_parts_I.append(torch.zeros(batch_size, self.config.fourier_n_features * 2, device=self.device))
                else:
                    input_parts_I.append(torch.zeros(batch_size, self.config.fusion_dim, device=self.device))

            if C_meta is not None:
                input_parts_I.append(C_meta.to(self.device))
            else:
                input_parts_I.append(torch.zeros(batch_size, self.config.meta_cognitive_dim, device=self.device))

            # I 群体不接收混沌注入
            input_parts_I.append(torch.zeros(batch_size, self.config.chaos_dim, device=self.device))

            # 时间
            input_parts_I.append(t_tensor)

            # 拼接输入
            if batch_size == 1 and y.dim() == 1:
                input_vector_E = torch.cat([part.flatten() if part.dim() > 1 else part for part in input_parts_E], dim=0)
                input_vector_I = torch.cat([part.flatten() if part.dim() > 1 else part for part in input_parts_I], dim=0)
            else:
                input_vector_E = torch.cat(input_parts_E, dim=-1)
                input_vector_I = torch.cat(input_parts_I, dim=-1)

            # 计算 E 群体动力学
            # 需要创建临时的完整输入以使用现有的演化函数
            # 注意：这里我们需要重新思考架构，因为演化函数期望完整的输入维度

            # 为 E 群体计算导数（使用部分输入）
            # 由于架构限制，我们采用简化方案：
            # 1. 使用完整的输入向量调用演化函数
            # 2. 然后分割输出并应用 E/I 特定的调节

            # 构建完整的输入向量（用于演化函数）
            # 注意：在 E/I 平衡模式下，混沌注入不作为演化函数的输入
            # 混沌注入将直接添加到 E 群体的动力学方程中
            input_parts = []
            input_parts.append(y)  # 完整状态

            if E_slow is not None:
                input_parts.append(E_slow.to(self.device))
            else:
                input_parts.append(torch.zeros(batch_size, self.config.slow_dim, device=self.device))

            if X_sem is not None:
                input_parts.append(X_sem.to(self.device))
            else:
                input_parts.append(torch.zeros(batch_size, self.config.semantic_dim, device=self.device))

            if X_log is not None:
                input_parts.append(X_log.to(self.device))
            else:
                input_parts.append(torch.zeros(batch_size, self.config.physical_dim, device=self.device))

            if X_fused is not None:
                X_fused = X_fused.to(self.device)
                if self.config.fourier_enabled:
                    projected = torch.matmul(self.fourier_B, X_fused.T).T
                    gamma_X = torch.cat([torch.cos(projected), torch.sin(projected)], dim=-1)
                    input_parts.append(gamma_X)
                else:
                    input_parts.append(X_fused)
            else:
                if self.config.fourier_enabled:
                    input_parts.append(torch.zeros(batch_size, self.config.fourier_n_features * 2, device=self.device))
                else:
                    input_parts.append(torch.zeros(batch_size, self.config.fusion_dim, device=self.device))

            if C_meta is not None:
                input_parts.append(C_meta.to(self.device))
            else:
                input_parts.append(torch.zeros(batch_size, self.config.meta_cognitive_dim, device=self.device))

            # E/I 平衡模式：混沌注入不作为演化函数输入
            # 使用零向量替代
            input_parts.append(torch.zeros(batch_size, self.config.chaos_dim, device=self.device))

            input_parts.append(t_tensor)

            if batch_size == 1 and y.dim() == 1:
                input_vector = torch.cat([part.flatten() if part.dim() > 1 else part for part in input_parts], dim=0)
            else:
                input_vector = torch.cat(input_parts, dim=-1)

            # 计算演化函数输出
            F_output = self.evolution_fn(input_vector)
            F_output = F_output * self.config.dynamics_scale

            # 分割输出
            if F_output.dim() == 1:
                F_E = F_output[:d_E]
                F_I = F_output[d_E:]
            else:
                F_E = F_output[:, :d_E]
                F_I = F_output[:, d_E:]

            # 计算衰减项
            decay_output = self.decay_layer(y)
            if decay_output.dim() == 1:
                decay_E = decay_output[:d_E]
                decay_I = decay_output[d_E:]
            else:
                decay_E = decay_output[:, :d_E]
                decay_I = decay_output[:, d_E:]

            # E 群体动力学：dx_E/dt = MLP(x_E) + g(E, I) - gamma * x_E + chaos_injection
            # g(E, I) = -inhibition
            # 混沌注入仅作用于 E 群体（需要扩展维度）
            if B_chaos is not None:
                # 处理混沌注入维度匹配
                if B_chaos.shape[-1] < self.config.chaos_dim:
                    padding = torch.zeros(
                        batch_size,
                        self.config.chaos_dim - B_chaos.shape[-1],
                        device=self.device
                    )
                    B_chaos_E = torch.cat([B_chaos.to(self.device), padding], dim=-1)
                elif B_chaos.shape[-1] > self.config.chaos_dim:
                    B_chaos_E = B_chaos[..., :self.config.chaos_dim].to(self.device)
                else:
                    B_chaos_E = B_chaos.to(self.device)

                # 扩展混沌注入到 d_E 维度（使用重复策略）
                # 将 chaos_dim 维度的混沌信号扩展到 d_E 维度
                if B_chaos_E.dim() == 1:
                    # 单样本情况
                    chaos_repeated = B_chaos_E.repeat(d_E // self.config.chaos_dim + 1)[:d_E]
                else:
                    # 批量情况
                    repeat_factor = d_E // self.config.chaos_dim + 1
                    chaos_repeated = B_chaos_E.repeat(1, repeat_factor)[:, :d_E]

                dydt_E = F_E - inhibition + decay_E + chaos_repeated
            else:
                dydt_E = F_E - inhibition + decay_E

            # I 群体动力学：dx_I/dt = MLP(x_I) - alpha * (E_mean - e_target)
            # I 群体不接收混沌注入，gamma 已在 MLP 内部应用
            dydt_I = F_I - inhibition + decay_I

            # 合并导数
            if y.dim() == 1:
                dydt = torch.cat([dydt_E, dydt_I], dim=0)
            else:
                dydt = torch.cat([dydt_E, dydt_I], dim=-1)

            # 添加范数依赖衰减
            norm = torch.norm(y, dim=-1, keepdim=True)
            threshold_half = self.config.state_norm_threshold * 0.5
            extra_decay_factor = torch.where(
                norm > threshold_half,
                (norm - threshold_half) / threshold_half,
                torch.zeros_like(norm)
            )
            extra_decay_factor = torch.clamp(extra_decay_factor, min=0.0, max=2.0)
            dydt = dydt - extra_decay_factor * y

            # 添加内部噪声
            if self.config.noise_scale > 0:
                noise = torch.randn_like(dydt) * self.config.noise_scale
                dydt = dydt + noise

        else:
            # === 原有的完整动力学（向后兼容）===
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
                    # 处理单样本（1D）和批量（2D）的情况
                    if B_chaos.dim() == 1:
                        # 单样本：1D 张量
                        padding = torch.zeros(
                            self.config.chaos_dim - B_chaos.shape[-1],
                            device=self.device
                        )
                        B_chaos = torch.cat([B_chaos.to(self.device), padding], dim=-1)
                    else:
                        # 批量：2D 张量
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

            # 方案C：缩放演化函数输出以控制混沌强度
            F_output = F_output * self.config.dynamics_scale

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

            # 添加额外阻尼项（方案B：显式阻尼，已废弃，改用 gamma）
            if hasattr(self.config, 'damping_coeff') and self.config.damping_coeff > 0:
                dydt = dydt - self.config.damping_coeff * y

            # gamma 耗散项已移至 MLP 内部（内部限速机制）
            # 不再需要外部的 -gamma * y 项

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

        logger.debug("Spectral normalization removed from FastDynamicsFunction")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            "forward_calls": self.forward_calls
        }

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

        # 状态控制器（用于状态切换和参数过渡）
        initial_mode = StateMode.WORK if self.config.state_mode == "work" else (
            StateMode.REST if self.config.state_mode == "rest" else StateMode.EXPLORE
        )
        self.state_controller = StateController(initial_mode=initial_mode)

        # 日志频率控制
        self._clip_log_interval = 100  # 每100步输出一次范数裁剪日志
        self._clip_log_counter = 0
        self._stability_log_interval = 50  # 每50步输出一次稳定性警告日志
        self._stability_log_counter = 0

        logger.debug(
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

        # 执行状态过渡一步（如果有正在进行的过渡）
        params, transition_complete = self.state_controller.step_transition()

        # 应用当前状态参数到动力学系统
        self._apply_state_params(params)

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

        # 状态范数主动截断（方案D：范数控制，防止持续增长）
        # 当 state_norm_clip > 0 时启用，每步将状态范数截断到此值
        if self.config.state_norm_clip > 0:
            norm = torch.norm(E_fast_new)
            if norm > self.config.state_norm_clip:
                E_fast_new = E_fast_new * (self.config.state_norm_clip / norm)

        # 状态范数裁剪（防止发散）- 使用当前状态参数
        norm = torch.norm(E_fast_new).item()
        threshold = params.state_norm_threshold
        if norm > threshold:
            scale = threshold / norm
            E_fast_new = E_fast_new * scale
            # 限制日志输出频率
            self._clip_log_counter += 1
            if self._clip_log_counter >= self._clip_log_interval:
                logger.debug(f"State norm clipped: {norm:.4e} -> {threshold}")
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
        return {
            "history_length": len(self.state_history),
            "dynamics_fn_stats": self.dynamics_fn.get_statistics() if self.dynamics_fn else None
        }

    def reset(self) -> None:
        """重置系统"""
        self.state_history.clear()
        self.time_history.clear()
        if self.dynamics_fn:
            self.dynamics_fn.reset_cache()

        # 重置日志计数器
        self._clip_log_counter = 0
        self._stability_log_counter = 0

        logger.debug("FastDynamicsSystem reset")

    def remove_spectral_norm(self) -> None:
        """
        移除所有谱归一化包装（用于调试或推理优化）

        警告：移除谱归一化后，权重将不再受Lipschitz约束
        """
        if self.dynamics_fn and hasattr(self.dynamics_fn, 'remove_spectral_norm'):
            self.dynamics_fn.remove_spectral_norm()
        logger.debug("Spectral normalization removed from FastDynamicsSystem")

    def _apply_state_params(self, params: StateParameters) -> None:
        """
        应用状态参数到动力学系统

        Args:
            params: 状态参数配置
        """
        # 更新配置参数
        self.config.decay_rate = params.decay_rate
        self.config.gamma = params.gamma
        self.config.dynamics_scale = params.dynamics_scale
        self.config.noise_scale = params.noise_scale
        self.config.ei_ratio = params.ei_ratio
        self.config.alpha = params.alpha
        self.config.e_target = params.e_target
        self.config.state_norm_threshold = params.state_norm_threshold
        self.config.state_norm_clip = params.state_norm_clip
        self.config.max_gradient_norm = params.max_gradient_norm
        self.config.target_spectral_norm = params.target_spectral_norm

        # 更新动力学函数中的参数（如果已初始化）
        if self.dynamics_fn is not None:
            self.dynamics_fn.config.decay_rate = params.decay_rate
            self.dynamics_fn.config.gamma = params.gamma
            self.dynamics_fn.config.dynamics_scale = params.dynamics_scale
            self.dynamics_fn.config.noise_scale = params.noise_scale
            self.dynamics_fn.config.state_norm_threshold = params.state_norm_threshold
            self.dynamics_fn.config.state_norm_clip = params.state_norm_clip
            self.dynamics_fn.config.max_gradient_norm = params.max_gradient_norm
            self.dynamics_fn.config.target_spectral_norm = params.target_spectral_norm
            
            # 更新衰减层权重（对角矩阵）
            if hasattr(self.dynamics_fn, 'decay_layer'):
                if hasattr(self.dynamics_fn.decay_layer, 'weight_u'):
                    self.dynamics_fn.decay_layer = torch.nn.utils.remove_spectral_norm(
                        self.dynamics_fn.decay_layer
                    )
                with torch.no_grad():
                    self.dynamics_fn.decay_layer.weight.zero_()
                    self.dynamics_fn.decay_layer.weight.diagonal().fill_(-params.decay_rate)
                self.dynamics_fn.decay_layer.bias = None
            
            # 更新 gamma 参数（内部限速模式）
            if hasattr(self.dynamics_fn.evolution_fn, 'gamma'):
                self.dynamics_fn.evolution_fn.gamma = params.gamma

    def switch_state(
        self,
        target_mode: StateMode,
        transition_steps: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        切换系统状态模式

        Args:
            target_mode: 目标状态模式
            transition_steps: 过渡步数（None 使用默认值 50）
            force: 是否强制切换（立即完成）

        Returns:
            是否成功开始切换
        """
        return self.state_controller.switch_state(target_mode, transition_steps, force)

    def get_current_state_mode(self) -> StateMode:
        """获取当前状态模式"""
        return self.state_controller.get_current_mode()

    def get_state_controller_stats(self) -> Dict[str, Any]:
        """获取状态控制器统计信息"""
        return self.state_controller.get_statistics()

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