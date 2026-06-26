"""
L1 自我状态层 - Self-State Layer
=================================

Task 13: 实现L1自我状态层，负责完整认知积分和状态管理。

核心功能：
- 完整认知积分（包含 SelfState 的快变量和慢变量）
- 数据引用接口（从 L0 获取感知数据）
- 工作记忆管理（与 WorkingMemory 集成）
- 注意焦点管理（基于重要性权重）
- 状态演化（通过 IntegrationEngine）
- 状态完整性验证

关键特性：
- L1 包含完整的自我状态（SelfState）
- 接收 L0 的感知数据进行状态演化
- 工作记忆管理组块和激活强度
- 注意焦点动态调整
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import time

from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine, IntegrationEngineConfig
from chronos_core.core.external_input import ExternalInput
from chronos_core.memory.work_memory import WorkingMemory, ChunkType
from chronos_core.utils.config import (
    DimensionalityConfig,
    MetaCognitiveConfig,
    MemoryTemporalConfig,
    ChronosConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SelfStateLayerConfig:
    """L1 自我状态层配置"""
    
    # 状态维度
    fast_dim: int = 2048         # 快变量维度
    slow_dim: int = 512          # 慢变量维度
    perception_dim: int = 1024   # L0 感知输入维度
    
    # 工作记忆参数
    working_memory_capacity: int = 7        # 米勒定律容量
    working_memory_decay_time: float = 10.0  # 激活衰减时间常数
    working_memory_chunk_dim: int = 256     # 组块维度
    
    # 注意焦点参数
    attention_heads: int = 8                 # 注意力头数
    attention_dropout: float = 0.1          # 注意力 dropout
    attention_temperature: float = 1.0      # 注意力温度
    
    # 积分引擎参数
    integration_dt: float = 0.01            # 默认积分步长
    slow_update_frequency: int = 100        # 慢变量更新频率
    
    # 状态演化参数
    state_evolution_rate: float = 0.001     # 状态演化速率
    stability_threshold: float = 1e6        # 稳定性阈值
    
    # L0 数据引用参数
    perception_update_interval: float = 1.0  # L0 数据更新间隔
    
    # 设备参数
    device: str = "cpu"


class AttentionFocusManager(nn.Module):
    """
    注意焦点管理器
    
    功能：
    - 基于重要性权重调整注意焦点
    - 多头注意力机制
    - 注意力分配优化
    - 动态焦点调整
    
    Attributes:
        attention_heads: 注意力头数
        attention_dropout: Dropout 概率
        attention_temperature: 注意力温度参数
    """
    
    def __init__(
        self,
        state_dim: int = 2048,
        perception_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
        temperature: float = 1.0,
        device: str = "cpu"
    ):
        """
        初始化注意焦点管理器
        
        Args:
            state_dim: 状态维度
            perception_dim: 感知输入维度
            num_heads: 注意力头数
            dropout: Dropout 概率
            temperature: 注意力温度
            device: 计算设备
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.perception_dim = perception_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.temperature = temperature
        self.device = device
        
        # 确保维度可以被注意力头数整除
        assert state_dim % num_heads == 0, f"state_dim {state_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = state_dim // num_heads
        
        # 状态注意力投影
        self.state_q_proj = nn.Linear(state_dim, state_dim)
        self.state_k_proj = nn.Linear(state_dim, state_dim)
        self.state_v_proj = nn.Linear(state_dim, state_dim)
        
        # 感知输入注意力投影
        self.perception_q_proj = nn.Linear(perception_dim, state_dim)
        self.perception_k_proj = nn.Linear(perception_dim, state_dim)
        self.perception_v_proj = nn.Linear(perception_dim, state_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(state_dim, state_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 重要性权重缓存
        self._importance_weights: Optional[torch.Tensor] = None
        self._attention_history: List[torch.Tensor] = []
        
        self.to(device)
        
        logger.info(
            f"AttentionFocusManager initialized: "
            f"state_dim={state_dim}, perception_dim={perception_dim}, "
            f"num_heads={num_heads}, temperature={temperature}"
        )
    
    def forward(
        self,
        state: torch.Tensor,
        perception_data: Optional[torch.Tensor] = None,
        importance_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算注意焦点
        
        Args:
            state: 当前状态
            perception_data: L0 感知数据
            importance_weights: 重要性权重
        
        Returns:
            注意焦点调整后的状态
        """
        state = state.to(self.device)
        
        # 处理重要性权重
        if importance_weights is None:
            importance_weights = torch.ones(self.state_dim, device=self.device)
        else:
            importance_weights = importance_weights.to(self.device)
            # 确保维度匹配
            if importance_weights.shape[0] != self.state_dim:
                importance_weights = importance_weights[:self.state_dim]
        
        self._importance_weights = importance_weights
        
        # 状态自注意力
        Q = self.state_q_proj(state)  # (state_dim,)
        K = self.state_k_proj(state)
        V = self.state_v_proj(state)
        
        # 如果有感知数据，计算交叉注意力
        if perception_data is not None:
            perception_data = perception_data.to(self.device)
            
            # 确保感知数据维度匹配
            if perception_data.shape[0] != self.perception_dim:
                if perception_data.shape[0] < self.perception_dim:
                    padding = torch.zeros(
                        self.perception_dim - perception_data.shape[0],
                        device=self.device
                    )
                    perception_data = torch.cat([perception_data, padding])
                else:
                    perception_data = perception_data[:self.perception_dim]
            
            # 感知数据查询
            perception_Q = self.perception_q_proj(perception_data)
            perception_K = self.perception_k_proj(perception_data)
            perception_V = self.perception_v_proj(perception_data)
            
            # 合并查询
            Q = Q + perception_Q[:self.state_dim] * 0.5
            K = K + perception_K[:self.state_dim] * 0.5
            V = V + perception_V[:self.state_dim] * 0.5
        
        # 多头注意力计算
        # 重塑为多头形式: (num_heads, head_dim)
        Q_heads = Q.view(self.num_heads, self.head_dim)
        K_heads = K.view(self.num_heads, self.head_dim)
        V_heads = V.view(self.num_heads, self.head_dim)
        
        # 计算注意力分数
        # (num_heads, 1)
        scores = torch.sum(Q_heads * K_heads, dim=-1) / (self.head_dim ** 0.5)
        scores = scores / self.temperature
        
        # 应用重要性权重
        weights_heads = importance_weights.view(self.num_heads, self.head_dim)
        scores = scores * weights_heads.mean(dim=-1)
        
        # Softmax
        attention_weights = torch.softmax(scores, dim=0)
        
        # 应用注意力权重
        attended_heads = attention_weights.unsqueeze(-1) * V_heads
        
        # 合合多头输出
        attended = attended_heads.view(self.state_dim)
        
        # Dropout
        attended = self.dropout_layer(attended)
        
        # 输出投影
        output = self.out_proj(attended)
        
        # 记录注意力历史
        self._attention_history.append(attention_weights.detach().clone())
        
        return output
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        获取当前注意力权重
        
        Returns:
            注意力权重张量
        """
        if self._attention_history:
            return self._attention_history[-1]
        return torch.ones(self.num_heads, device=self.device)
    
    def get_importance_weights(self) -> Optional[torch.Tensor]:
        """
        获取重要性权重
        
        Returns:
            重要性权重张量
        """
        return self._importance_weights
    
    def update_importance_weights(
        self,
        weights: torch.Tensor,
        smoothing_factor: float = 0.1
    ):
        """
        更新重要性权重（平滑更新）
        
        Args:
            weights: 新的重要性权重
            smoothing_factor: 平滑因子
        """
        weights = weights.to(self.device)
        
        if self._importance_weights is None:
            self._importance_weights = weights
        else:
            # 平滑更新
            self._importance_weights = (
                smoothing_factor * weights +
                (1 - smoothing_factor) * self._importance_weights
            )
    
    def reset_attention_history(self):
        """重置注意力历史"""
        self._attention_history.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "temperature": self.temperature,
            "attention_history_length": len(self._attention_history),
        }
        
        if self._attention_history:
            recent_attention = self._attention_history[-1]
            stats["recent_attention_mean"] = recent_attention.mean().item()
            stats["recent_attention_std"] = recent_attention.std().item()
            stats["recent_attention_max"] = recent_attention.max().item()
        
        return stats


class WorkingMemoryIntegrator:
    """
    工作记忆整合器
    
    功能：
    - 与 WorkingMemory 集成
    - 组块更新与激活强度管理
    - 工作记忆输出整合
    - 状态与工作记忆的协同
    
    Attributes:
        working_memory: WorkingMemory 实例
    """
    
    def __init__(
        self,
        working_memory: Optional[WorkingMemory] = None,
        fast_dim: int = 2048,
        chunk_dim: int = 256,
        capacity: int = 7,
        decay_time_constant: float = 10.0,
        min_activation: float = 0.01,
        device: str = "cpu"
    ):
        """
        初始化工作记忆整合器
        
        Args:
            working_memory: WorkingMemory 实例（可选，会自动创建）
            fast_dim: 快变量维度
            chunk_dim: 组块维度
            capacity: 工作记忆容量
            decay_time_constant: 衰减时间常数
            min_activation: 最小激活阈值
            device: 计算设备
        """
        self.fast_dim = fast_dim
        self.chunk_dim = chunk_dim
        self.device = device
        
        # 创建或使用提供的工作记忆
        if working_memory is None:
            self.working_memory = WorkingMemory(
                capacity=capacity,
                fast_dim=fast_dim,
                chunk_dim=chunk_dim,
                decay_time_constant=decay_time_constant,
                min_activation=min_activation,
                device=device
            )
        else:
            self.working_memory = working_memory
        
        logger.info(
            f"WorkingMemoryIntegrator initialized: "
            f"capacity={capacity}, chunk_dim={chunk_dim}"
        )
    
    def update_from_state(
        self,
        state: torch.Tensor,
        delta_time: float,
        importance_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        从状态更新工作记忆
        
        Args:
            state: 快变量状态
            delta_time: 时间间隔
            importance_weights: 重要性权重
        
        Returns:
            工作记忆输出
        """
        state = state.to(self.device)
        
        # 计算输入驱动力（基于状态变化）
        input_drive = {}
        
        # 基于重要性权重生成组块
        if importance_weights is None:
            importance_weights = torch.ones(state.shape[0], device=self.device)
        
        # 创建新组块或更新现有组块
        # 简化实现：定期创建新组块
        if len(self.working_memory.get_all_chunks()) < self.working_memory.capacity:
            chunk = self.working_memory.create_chunk(
                source_state=state,
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=importance_weights.mean().item(),
            )
            logger.debug(f"Created new chunk: {chunk.chunk_id}")
        
        # 更新激活强度（衰减）
        self.working_memory.update_activations(delta_time)
        
        # 计算工作记忆输出
        wm_output = self.working_memory.compute_working_memory_output()
        
        return wm_output
    
    def integrate_with_state(
        self,
        state: torch.Tensor,
        wm_output: torch.Tensor,
        integration_weight: float = 0.1
    ) -> torch.Tensor:
        """
        整合工作记忆输出与状态
        
        Args:
            state: 状态张量
            wm_output: 工作记忆输出
            integration_weight: 整合权重
        
        Returns:
            整合后的状态
        """
        state = state.to(self.device)
        wm_output = wm_output.to(self.device)
        
        # 确保维度匹配
        if wm_output.shape[0] != state.shape[0]:
            if wm_output.shape[0] < state.shape[0]:
                # 扩展工作记忆输出
                padding = torch.zeros(
                    state.shape[0] - wm_output.shape[0],
                    device=self.device
                )
                wm_output = torch.cat([wm_output, padding])
            else:
                # 截断工作记忆输出
                wm_output = wm_output[:state.shape[0]]
        
        # 加权整合
        integrated_state = state * (1 - integration_weight) + wm_output * integration_weight
        
        return integrated_state
    
    def get_working_memory_output(self) -> torch.Tensor:
        """
        获取工作记忆输出
        
        Returns:
            工作记忆输出张量
        """
        return self.working_memory.compute_working_memory_output()
    
    def get_active_chunks_info(self) -> List[Dict[str, Any]]:
        """
        获取激活组块信息
        
        Returns:
            组块信息列表
        """
        active_chunks = self.working_memory.get_active_chunks()
        
        chunks_info = []
        for chunk in active_chunks:
            activation = self.working_memory.activation_strength.get_activation(chunk.chunk_id)
            chunks_info.append({
                "chunk_id": chunk.chunk_id,
                "chunk_type": chunk.chunk_type.value,
                "activation": activation,
                "content_norm": chunk.get_content_norm(),
            })
        
        return chunks_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        wm_stats = self.working_memory.get_statistics()
        
        stats = {
            "working_memory_stats": wm_stats,
            "fast_dim": self.fast_dim,
            "chunk_dim": self.chunk_dim,
        }
        
        return stats
    
    def reset(self):
        """重置工作记忆整合器"""
        self.working_memory.clear()
        logger.info("WorkingMemoryIntegrator reset")


class SelfStateLayer(nn.Module):
    """
    L1 自我状态层
    
    Task 13 完整实现
    
    功能：
    - 完整认知积分（包含 SelfState）
    - 数据引用接口（从 L0 获取感知数据）
    - 工作记忆管理（WorkingMemoryIntegrator）
    - 注意焦点管理（AttentionFocusManager）
    - 状态演化（IntegrationEngine）
    - 状态完整性验证
    
    特性：
    - L1 包含完整的自我状态和自我模型
    - 接收 L0 的感知数据，驱动状态演化
    - 工作记忆管理组块和激活强度
    - 注意焦点基于重要性权重动态调整
    - 与 L2 交互，接收调控信号
    """
    
    def __init__(
        self,
        config: Optional[SelfStateLayerConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        memory_config: Optional[MemoryTemporalConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化 L1 自我状态层
        
        Args:
            config: 自我状态层配置
            dim_config: 维度配置
            meta_config: 元认知配置
            memory_config: 内存配置
            global_config: 全局配置
            device: 计算设备
        """
        super().__init__()
        
        # 合并配置
        self.config = config or SelfStateLayerConfig()
        
        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim
            self.config.perception_dim = dim_config.fusion_dim
            self.config.working_memory_chunk_dim = dim_config.working_memory_dim
        
        if meta_config:
            self.config.attention_heads = max(1, dim_config.fast_variable_dim // 256)
            self.config.attention_temperature = 1.0
        
        if memory_config:
            self.config.working_memory_capacity = memory_config.working_memory_chunks
            self.config.working_memory_decay_time = 10.0  # 默认值
            self.config.slow_update_frequency = memory_config.slow_update_frequency
        
        self.device = device or self.config.device
        
        # 创建自我状态
        self.self_state: Optional[SelfState] = None
        self._initialize_self_state()
        
        # 创建积分引擎
        self.integration_engine: Optional[IntegrationEngine] = None
        if global_config:
            self.integration_engine = IntegrationEngine(
                config=global_config,
                device=self.device
            )
        else:
            # 使用简化配置
            engine_config = IntegrationEngineConfig(
                fast_dim=self.config.fast_dim,
                slow_dim=self.config.slow_dim,
            )
            # 创建简化的积分引擎（不初始化，避免依赖缺失）
            self.integration_engine = None
            logger.info("IntegrationEngine not initialized (simplified mode)")
        
        # 创建注意焦点管理器
        self.attention_focus_manager = AttentionFocusManager(
            state_dim=self.config.fast_dim,
            perception_dim=self.config.perception_dim,
            num_heads=self.config.attention_heads,
            dropout=self.config.attention_dropout,
            temperature=self.config.attention_temperature,
            device=self.device
        )
        
        # 创建工作记忆整合器
        self.working_memory_integrator = WorkingMemoryIntegrator(
            fast_dim=self.config.fast_dim,
            chunk_dim=self.config.working_memory_chunk_dim,
            capacity=self.config.working_memory_capacity,
            decay_time_constant=self.config.working_memory_decay_time,
            device=self.device
        )
        
        # L0 数据引用缓存
        self._l0_perception_cache: Optional[torch.Tensor] = None
        self._l0_update_time: float = 0.0
        
        # L2 调控信号缓存
        self._l2_control_signal: Optional[torch.Tensor] = None
        
        # 状态演化历史
        self._state_history: List[SelfState] = []
        
        # 统计信息
        self._stats = {
            "total_steps": 0,
            "l0_updates": 0,
            "l2_signals_received": 0,
            "attention_adjustments": 0,
            "wm_updates": 0,
        }
        
        self.to(self.device)
        
        logger.info(
            f"SelfStateLayer (L1) initialized: "
            f"fast_dim={self.config.fast_dim}, "
            f"slow_dim={self.config.slow_dim}, "
            f"perception_dim={self.config.perception_dim}, "
            f"wm_capacity={self.config.working_memory_capacity}"
        )
    
    def _initialize_self_state(self):
        """初始化自我状态"""
        self.self_state = SelfState(
            E_fast=torch.zeros(self.config.fast_dim),
            E_slow=torch.zeros(self.config.slow_dim),
            timestamp=0.0
        )
        logger.debug("SelfState initialized with zeros")
    
    def forward(
        self,
        l0_perception_data: Optional[torch.Tensor] = None,
        l2_control_signal: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        update_working_memory: bool = True
    ) -> SelfState:
        """
        执行状态演化
        
        流程：
        1. 接收 L0 感知数据
        2. 接收 L2 调控信号
        3. 注意焦点调整
        4. 工作记忆更新
        5. 状态演化（积分）
        6. 状态完整性验证
        
        Args:
            l0_perception_data: L0 感知数据
            l2_control_signal: L2 调控信号
            dt: 时间步长
            update_working_memory: 是否更新工作记忆
        
        Returns:
            新的自我状态
        """
        dt = dt or self.config.integration_dt
        
        # 1. 接收 L0 感知数据
        if l0_perception_data is not None:
            self._update_l0_perception(l0_perception_data)
        
        # 2. 接收 L2 调控信号
        if l2_control_signal is not None:
            self._update_l2_control_signal(l2_control_signal)
        
        # 3. 注意焦点调整
        importance_weights = self._compute_importance_weights()
        
        E_fast_attended = self.attention_focus_manager(
            self.self_state.E_fast,
            perception_data=self._l0_perception_cache,
            importance_weights=importance_weights
        )
        self._stats["attention_adjustments"] += 1
        
        # 4. 工作记忆更新
        if update_working_memory:
            wm_output = self.working_memory_integrator.update_from_state(
                E_fast_attended,
                dt,
                importance_weights
            )
            
            # 整合工作记忆输出
            E_fast_attended = self.working_memory_integrator.integrate_with_state(
                E_fast_attended,
                wm_output,
                integration_weight=0.1
            )
            self._stats["wm_updates"] += 1
        
        # 5. 状态演化（积分）
        # 使用简化的演化（如果没有积分引擎）
        if self.integration_engine and self.integration_engine._initialized:
            # 使用完整积分引擎
            external_input = self._build_external_input()
            new_state = self.integration_engine.step(
                self.self_state,
                external_input,
                meta_cognitive_signal=self._l2_control_signal,
                dt=dt
            )
        else:
            # 使用简化演化
            new_state = self._simple_state_evolution(E_fast_attended, dt)
        
        # 更新自我状态
        self.self_state = new_state
        
        # 6. 状态完整性验证
        is_valid, errors = self.verify_state_integrity()
        if not is_valid:
            logger.warning(f"State integrity validation failed: {errors}")
        
        # 记录历史
        self._state_history.append(self.self_state.copy())
        
        # 更新统计
        self._stats["total_steps"] += 1
        
        return self.self_state
    
    def _update_l0_perception(self, perception_data: torch.Tensor):
        """
        更新 L0 感知数据缓存
        
        Args:
            perception_data: L0 感知数据
        """
        perception_data = perception_data.to(self.device)
        
        # 确保维度匹配
        if perception_data.shape[0] != self.config.perception_dim:
            if perception_data.shape[0] < self.config.perception_dim:
                padding = torch.zeros(
                    self.config.perception_dim - perception_data.shape[0],
                    device=self.device
                )
                perception_data = torch.cat([perception_data, padding])
            else:
                perception_data = perception_data[:self.config.perception_dim]
        
        self._l0_perception_cache = perception_data.clone()
        self._l0_update_time = time.time()
        self._stats["l0_updates"] += 1
        
        logger.debug(f"L0 perception data updated: shape={perception_data.shape}")
    
    def _update_l2_control_signal(self, control_signal: torch.Tensor):
        """
        更新 L2 调控信号缓存
        
        Args:
            control_signal: L2 调控信号
        """
        control_signal = control_signal.to(self.device)
        self._l2_control_signal = control_signal.clone()
        self._stats["l2_signals_received"] += 1
        
        logger.debug(f"L2 control signal updated: shape={control_signal.shape}")
    
    def _compute_importance_weights(self) -> torch.Tensor:
        """
        计算重要性权重
        
        Returns:
            重要性权重张量
        """
        # 基于状态范数和感知数据计算重要性
        state_norm = self.self_state.get_fast_norm()
        
        # 默认权重
        weights = torch.ones(self.config.fast_dim, device=self.device)
        
        # 如果有感知数据，基于感知数据调整权重
        if self._l0_perception_cache is not None:
            # 计算感知数据的影响权重
            perception_norm = torch.norm(self._l0_perception_cache).item()
            influence_factor = min(1.0, perception_norm / (state_norm + 1e-8))
            
            # 加权调整
            weights = weights * (1 + influence_factor)
        
        # 如果有 L2 调控信号，基于调控信号调整权重
        if self._l2_control_signal is not None:
            # L2 调控信号可以提供重要性权重建议
            # 简化实现：使用调控信号范数作为权重因子
            control_norm = torch.norm(self._l2_control_signal).item()
            weights = weights * (1 + control_norm * 0.1)
        
        # 归一化权重
        weights = weights / weights.sum()
        
        return weights
    
    def _build_external_input(self) -> ExternalInput:
        """
        构建外部输入对象
        
        Returns:
            ExternalInput 实例
        """
        if self._l0_perception_cache is None:
            # 创建空输入
            return ExternalInput(
                X_sem=torch.zeros(self.config.perception_dim // 2),
                X_log=torch.zeros(self.config.perception_dim // 2)
            )
        
        # 分割感知数据为语义和物理流
        split_dim = self.config.perception_dim // 2
        X_sem = self._l0_perception_cache[:split_dim]
        X_log = self._l0_perception_cache[split_dim:]
        
        return ExternalInput(
            X_sem=X_sem,
            X_log=X_log,
            timestamp=self.self_state.timestamp
        )
    
    def _simple_state_evolution(
        self,
        E_fast_attended: torch.Tensor,
        dt: float
    ) -> SelfState:
        """
        简化状态演化（当没有积分引擎时使用）
        
        Args:
            E_fast_attended: 注意调整后的快变量
            dt: 时间步长
        
        Returns:
            新的自我状态
        """
        # 简化的演化方程
        # E_fast_new = E_fast_attended + noise + perception_influence
        # E_slow_new = E_slow + decay
        
        E_fast = self.self_state.E_fast.to(self.device)
        E_slow = self.self_state.E_slow.to(self.device)
        
        # 快变量演化
        noise = torch.randn_like(E_fast) * 0.01  # 小噪声
        perception_influence = torch.zeros_like(E_fast)
        
        if self._l0_perception_cache is not None:
            # 感知数据投影到快变量维度
            if self._l0_perception_cache.shape[0] < E_fast.shape[0]:
                # 重复填充
                repeat_factor = E_fast.shape[0] // self._l0_perception_cache.shape[0]
                perception_influence = self._l0_perception_cache.repeat(repeat_factor)[:E_fast.shape[0]]
            else:
                perception_influence = self._l0_perception_cache[:E_fast.shape[0]]
        
        E_fast_new = E_fast_attended + noise + perception_influence * 0.01
        
        # 检查是否需要更新慢变量
        should_update_slow = self._stats["total_steps"] % self.config.slow_update_frequency == 0
        
        if should_update_slow:
            # 慢变量演化（简化）
            decay = -self.config.state_evolution_rate * E_slow
            E_slow_new = E_slow + decay * dt
        else:
            E_slow_new = E_slow
        
        # 创建新状态
        new_state = SelfState(
            E_fast=E_fast_new.detach().cpu(),
            E_slow=E_slow_new.detach().cpu(),
            timestamp=self.self_state.timestamp + dt,
            history=self.self_state.history.copy(),
            metadata={
                "step": self._stats["total_steps"],
                "l0_updated": self._l0_update_time > 0,
                "l2_signal": self._l2_control_signal is not None,
            }
        )
        
        return new_state
    
    def verify_state_integrity(self) -> Tuple[bool, List[str]]:
        """
        验证状态完整性
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        if self.self_state is None:
            return False, ["SelfState is None"]
        
        # 使用 SelfState 的验证方法
        is_valid, errors = self.self_state.validate()
        
        # 附加检查
        # 1. 检查快慢变量维度
        if self.self_state.E_fast.shape[0] != self.config.fast_dim:
            errors.append(
                f"E_fast dimension mismatch: expected {self.config.fast_dim}, "
                f"got {self.self_state.E_fast.shape[0]}"
            )
        
        if self.self_state.E_slow.shape[0] != self.config.slow_dim:
            errors.append(
                f"E_slow dimension mismatch: expected {self.config.slow_dim}, "
                f"got {self.self_state.E_slow.shape[0]}"
            )
        
        # 2. 检查状态稳定性
        fast_norm = self.self_state.get_fast_norm()
        slow_norm = self.self_state.get_slow_norm()
        
        if fast_norm > self.config.stability_threshold:
            errors.append(f"Fast variable norm too large: {fast_norm:.4e}")
        
        if slow_norm > self.config.stability_threshold:
            errors.append(f"Slow variable norm too large: {slow_norm:.4e}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_state(self) -> SelfState:
        """
        获取当前自我状态
        
        Returns:
            SelfState 实例
        """
        return self.self_state
    
    def set_state(self, state: SelfState):
        """
        设置自我状态
        
        Args:
            state: 新的自我状态
        """
        self.self_state = state.copy()
        logger.info(f"SelfState set: timestamp={state.timestamp:.2f}")
    
    def receive_from_l0(self) -> Optional[torch.Tensor]:
        """
        数据引用接口：从 L0 获取感知数据
        
        Returns:
            L0 感知数据
        """
        return self._l0_perception_cache
    
    def send_to_l2(self) -> torch.Tensor:
        """
        将状态数据发送给 L2
        
        Returns:
            状态数据张量（L2 将接收这个数据）
        """
        # 合合快变量和慢变量
        E_fast = self.self_state.E_fast.to(self.device)
        E_slow = self.self_state.E_slow.to(self.device)
        
        # 拼合为完整状态向量
        state_vector = torch.cat([E_fast, E_slow], dim=0)
        
        return state_vector
    
    def receive_l2_control(self) -> Optional[torch.Tensor]:
        """
        接收 L2 调控信号
        
        Returns:
            L2 调控信号
        """
        return self._l2_control_signal
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        获取当前注意力权重
        
        Returns:
            注意力权重张量
        """
        return self.attention_focus_manager.get_attention_weights()
    
    def get_working_memory_info(self) -> Dict[str, Any]:
        """
        获取工作记忆信息
        
        Returns:
            工作记忆统计信息
        """
        return self.working_memory_integrator.get_statistics()
    
    def get_active_chunks(self) -> List[Dict[str, Any]]:
        """
        获取激活组块信息
        
        Returns:
            激活组块信息列表
        """
        return self.working_memory_integrator.get_active_chunks_info()
    
    def reset(self):
        """重置自我状态层"""
        # 重置自我状态
        self._initialize_self_state()
        
        # 重置积分引擎
        if self.integration_engine:
            self.integration_engine.reset()
        
        # 重置注意力
        self.attention_focus_manager.reset_attention_history()
        
        # 重置工作记忆
        self.working_memory_integrator.reset()
        
        # 清空缓存
        self._l0_perception_cache = None
        self._l0_update_time = 0.0
        self._l2_control_signal = None
        
        # 清空历史
        self._state_history.clear()
        
        # 重置统计
        self._stats = {
            "total_steps": 0,
            "l0_updates": 0,
            "l2_signals_received": 0,
            "attention_adjustments": 0,
            "wm_updates": 0,
        }
        
        logger.info("SelfStateLayer reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        # 添加状态信息
        if self.self_state:
            stats["state_info"] = {
                "timestamp": self.self_state.timestamp,
                "fast_norm": self.self_state.get_fast_norm(),
                "slow_norm": self.self_state.get_slow_norm(),
                "history_length": len(self.self_state.history),
            }
        
        # 添加注意力信息
        stats["attention_stats"] = self.attention_focus_manager.get_statistics()
        
        # 添加工作记忆信息
        stats["working_memory_stats"] = self.working_memory_integrator.get_statistics()
        
        # 添加配置信息
        stats["config"] = {
            "fast_dim": self.config.fast_dim,
            "slow_dim": self.config.slow_dim,
            "perception_dim": self.config.perception_dim,
            "wm_capacity": self.config.working_memory_capacity,
        }
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证自我状态层
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证状态完整性
        is_valid, state_errors = self.verify_state_integrity()
        errors.extend(state_errors)
        
        # 验证配置
        if self.config.fast_dim <= 0:
            errors.append(f"Invalid fast_dim: {self.config.fast_dim}")
        if self.config.slow_dim <= 0:
            errors.append(f"Invalid slow_dim: {self.config.slow_dim}")
        if self.config.perception_dim <= 0:
            errors.append(f"Invalid perception_dim: {self.config.perception_dim}")
        
        # 验证工作记忆容量
        if not 5 <= self.config.working_memory_capacity <= 9:
            errors.append(
                f"WM capacity {self.config.working_memory_capacity} "
                f"violates Miller's law (5-9)"
            )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __repr__(self) -> str:
        state_info = ""
        if self.self_state:
            state_info = (
                f"timestamp={self.self_state.timestamp:.2f}, "
                f"fast_norm={self.self_state.get_fast_norm():.4f}"
            )
        
        return (
            f"SelfStateLayer(L1, "
            f"fast_dim={self.config.fast_dim}, "
            f"slow_dim={self.config.slow_dim}, "
            f"wm_capacity={self.config.working_memory_capacity}, "
            f"{state_info})"
        )


def create_self_state_layer_from_config(
    dim_config: DimensionalityConfig,
    meta_config: MetaCognitiveConfig,
    memory_config: MemoryTemporalConfig,
    global_config: Optional[ChronosConfig] = None,
    device: Optional[str] = None
) -> SelfStateLayer:
    """
    从配置创建自我状态层
    
    Args:
        dim_config: 维度配置
        meta_config: 元认知配置
        memory_config: 内存配置
        global_config: 全局配置
        device: 计算设备
    
    Returns:
        SelfStateLayer 实例
    """
    config = SelfStateLayerConfig(
        fast_dim=dim_config.fast_variable_dim,
        slow_dim=dim_config.slow_variable_dim,
        perception_dim=dim_config.fusion_dim,
        working_memory_capacity=memory_config.working_memory_chunks,
        working_memory_chunk_dim=dim_config.working_memory_dim,
        slow_update_frequency=memory_config.slow_update_frequency,
    )
    
    layer = SelfStateLayer(
        config=config,
        dim_config=dim_config,
        meta_config=meta_config,
        memory_config=memory_config,
        global_config=global_config,
        device=device
    )
    
    return layer