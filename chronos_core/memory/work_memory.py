"""
工作记忆系统 - Working Memory System
=====================================

实现基于米勒定律的工作记忆机制（7±2组块）

核心组件：
- Chunk: 组块数据类，表示聚合的信息单元
- ActivationStrength: 激活强度向量管理
- WorkingMemory: 完整工作记忆系统

核心公式：
1. 组块形成: C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
2. 激活衰减: a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a) + InputDrive_k(t)
3. 容量约束: 仅激活强度排名前N（N=7）的组块具备完整计算权重
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
import copy

logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """组块类型枚举"""
    SEMANTIC = "semantic"       # 语义类组块（目标、意图）
    EMOTIONAL = "emotional"     # 情感类组块（情绪状态）
    PHYSICAL = "physical"       # 物理类组块（环境状态）
    PROPRI0CEPTIVE = "proprioceptive"  # 本体感觉类组块（内部状态）
    CAUSAL = "causal"          # 因果类组块（因果链条）
    HYBRID = "hybrid"          # 混合类组块（多类型融合）
    TEMPORARY = "temporary"    # 临时组块（短期处理）
    UNKNOWN = "unknown"        # 未知类型


class ChunkStatus(Enum):
    """组块状态枚举"""
    ACTIVE = "active"          # 激活状态（参与计算）
    DORMANT = "dormant"        # 休眠状态（保留但不计算）
    RECOVERING = "recovering"  # 恢复状态（正在被重新激活）
    MERGING = "merging"        # 合并状态（与其他组块合并）
    DECAYING = "decaying"      # 衰减状态（正在衰减）
    REMOVED = "removed"        # 已移除状态


@dataclass
class Chunk:
    """
    组块数据类
    
    组块定义：将相关维度聚合为一个可操作的信息单元
    
    组块形成公式：
        C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
    
    其中：
        - C_k: 第k个组块的内容向量
        - α_ki: 注意力权重，绑定快变量的不同维度
        - E_fast^(i): 快变量的第i维度子空间
    
    Attributes:
        chunk_id: 组块唯一标识符
        content: 组块内容向量（聚合后的信息单元）
        attention_weights: 注意力权重向量，绑定快变量维度
        chunk_type: 组块类型
        status: 组块状态
        creation_time: 创建时间戳
        last_update_time: 最后更新时间戳
        source_dimensions: 来源维度索引集合
        metadata: 额外元数据
    """
    
    chunk_id: str
    content: torch.Tensor = field(default=None)
    attention_weights: torch.Tensor = field(default=None)
    chunk_type: ChunkType = field(default=ChunkType.UNKNOWN)
    status: ChunkStatus = field(default=ChunkStatus.ACTIVE)
    creation_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    source_dimensions: Set[int] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 默认组块维度
    CHUNK_DIM: int = field(default=256, init=False, repr=False)
    
    def __post_init__(self):
        """初始化后处理，确保张量维度正确"""
        if self.content is None:
            self.content = torch.zeros(self.CHUNK_DIM)
        elif not isinstance(self.content, torch.Tensor):
            self.content = torch.tensor(self.content, dtype=torch.float32)
        
        # 获取实际内容维度
        actual_dim = self.content.shape[0]
        
        if self.attention_weights is None:
            # 使用内容维度初始化注意力权重
            self.attention_weights = torch.ones(actual_dim) / actual_dim
        elif not isinstance(self.attention_weights, torch.Tensor):
            self.attention_weights = torch.tensor(self.attention_weights, dtype=torch.float32)
        
        # 确保注意力权重维度与内容维度匹配
        if self.attention_weights.shape[0] != actual_dim:
            # 调整注意力权重维度
            if self.attention_weights.shape[0] > actual_dim:
                self.attention_weights = self.attention_weights[:actual_dim]
            else:
                self.attention_weights = torch.cat([
                    self.attention_weights,
                    torch.zeros(actual_dim - self.attention_weights.shape[0])
                ])
            # 重新归一化
            if self.attention_weights.sum() > 0:
                self.attention_weights = self.attention_weights / self.attention_weights.sum()
            else:
                self.attention_weights = torch.ones(actual_dim) / actual_dim
        
        # 确保注意力权重是有效的概率分布
        if self.attention_weights.sum() > 0:
            self.attention_weights = self.attention_weights / self.attention_weights.sum()
        
        # 确保张量是连续的
        self.content = self.content.contiguous()
        self.attention_weights = self.attention_weights.contiguous()
    
    def update_content(
        self,
        new_content: torch.Tensor,
        new_attention_weights: Optional[torch.Tensor] = None,
    ):
        """
        更新组块内容和注意力权重
        
        Args:
            new_content: 新的内容向量
            new_attention_weights: 新的注意力权重（可选）
        """
        self.content = new_content.contiguous()
        if new_attention_weights is not None:
            self.attention_weights = new_attention_weights.contiguous()
            if self.attention_weights.sum() > 0:
                self.attention_weights = self.attention_weights / self.attention_weights.sum()
        self.last_update_time = time.time()
    
    def compute_weighted_content(self) -> torch.Tensor:
        """
        计算加权后的组块内容
        
        Returns:
            加权后的内容向量
        """
        return self.content * self.attention_weights
    
    def get_content_norm(self) -> float:
        """获取组块内容的 L2 范数"""
        return torch.norm(self.content).item()
    
    def get_attention_entropy(self) -> float:
        """
        计算注意力权重的熵
        
        高熵表示均匀分布（广泛注意力）
        低熵表示集中分布（窄注意力）
        
        Returns:
            注意力熵值
        """
        # 避免数值问题
        weights = self.attention_weights.clamp(min=1e-10)
        entropy = -torch.sum(weights * torch.log(weights)).item()
        return entropy
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "chunk_id": self.chunk_id,
            "content": self.content.detach().cpu().numpy().tolist(),
            "attention_weights": self.attention_weights.detach().cpu().numpy().tolist(),
            "chunk_type": self.chunk_type.value,
            "status": self.status.value,
            "creation_time": self.creation_time,
            "last_update_time": self.last_update_time,
            "source_dimensions": list(self.source_dimensions),
            "metadata": self.metadata.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        """从字典反序列化"""
        content = torch.tensor(data["content"], dtype=torch.float32)
        attention_weights = torch.tensor(data["attention_weights"], dtype=torch.float32)
        
        return cls(
            chunk_id=data["chunk_id"],
            content=content,
            attention_weights=attention_weights,
            chunk_type=ChunkType(data.get("chunk_type", "unknown")),
            status=ChunkStatus(data.get("status", "active")),
            creation_time=data.get("creation_time", time.time()),
            last_update_time=data.get("last_update_time", time.time()),
            source_dimensions=set(data.get("source_dimensions", [])),
            metadata=data.get("metadata", {}),
        )
    
    def copy(self) -> "Chunk":
        """深拷贝组块"""
        return Chunk(
            chunk_id=self.chunk_id,
            content=self.content.clone().detach(),
            attention_weights=self.attention_weights.clone().detach(),
            chunk_type=self.chunk_type,
            status=self.status,
            creation_time=self.creation_time,
            last_update_time=self.last_update_time,
            source_dimensions=self.source_dimensions.copy(),
            metadata=copy.deepcopy(self.metadata),
        )
    
    def is_active(self) -> bool:
        """判断组块是否处于激活状态"""
        return self.status == ChunkStatus.ACTIVE
    
    def is_dormant(self) -> bool:
        """判断组块是否处于休眠状态"""
        return self.status == ChunkStatus.DORMANT
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证组块的有效性
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 检查数值稳定性
        if torch.isnan(self.content).any():
            errors.append("Content contains NaN values")
        if torch.isinf(self.content).any():
            errors.append("Content contains Inf values")
        if torch.isnan(self.attention_weights).any():
            errors.append("Attention weights contain NaN values")
        if torch.isinf(self.attention_weights).any():
            errors.append("Attention weights contain Inf values")
        
        # 检查注意力权重有效性
        if self.attention_weights.sum() <= 0:
            errors.append("Attention weights sum is zero or negative")
        if (self.attention_weights < 0).any():
            errors.append("Attention weights contain negative values")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"Chunk(id={self.chunk_id}, type={self.chunk_type.value}, "
            f"status={self.status.value}, content_norm={self.get_content_norm():.4f}, "
            f"attention_entropy={self.get_attention_entropy():.4f})"
        )


class ActivationStrength:
    """
    激活强度向量管理类
    
    定义：A(t) = {a_1(t), a_2(t), ..., a_M(t)}
    
    每个组块有一个激活强度值（0.0-1.0），表示该组块在工作记忆中的活跃程度
    
    核心功能：
    1. 激活强度向量管理
    2. 激活衰减机制：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a)
    3. 输入驱动激活增强
    4. 数值稳定性维护
    
    Attributes:
        activations: 激活强度张量
        chunk_ids: 组块ID列表（与激活强度对应）
        decay_time_constant: 衰减时间常数 τ_a（秒）
        min_activation: 最小激活阈值 ε（历史信息保留）
        max_activation: 最大激活阈值（1.0）
    """
    
    def __init__(
        self,
        decay_time_constant: float = 10.0,  # τ_a: 衰减时间常数（秒）
        min_activation: float = 0.01,       # ε: 最小激活阈值
        max_activation: float = 1.0,        # 最大激活阈值
        device: str = "cpu",
    ):
        """
        初始化激活强度管理器
        
        Args:
            decay_time_constant: 衰减时间常数 τ_a（秒），控制衰减速度
            min_activation: 最小激活阈值 ε，用于历史信息保留
            max_activation: 最大激活阈值
            device: 计算设备
        """
        self.decay_time_constant = decay_time_constant
        self.min_activation = min_activation
        self.max_activation = max_activation
        self.device = device
        
        # 激活强度张量和对应的组块ID
        self._activations: Dict[str, float] = {}
        
        # 时间戳记录（用于衰减计算）
        self._last_update_times: Dict[str, float] = {}
        
        logger.info(
            f"ActivationStrength initialized: τ_a={decay_time_constant}s, "
            f"ε={min_activation}, device={device}"
        )
    
    def add_chunk(self, chunk_id: str, initial_activation: float = 1.0):
        """
        为新组块添加激活强度
        
        Args:
            chunk_id: 组块ID
            initial_activation: 初始激活强度（默认为1.0）
        """
        # 裁剪到有效范围
        activation = max(self.min_activation, min(self.max_activation, initial_activation))
        
        self._activations[chunk_id] = activation
        self._last_update_times[chunk_id] = time.time()
        
        logger.debug(
            f"Added activation for chunk {chunk_id}: {activation:.4f}"
        )
    
    def remove_chunk(self, chunk_id: str):
        """
        移除组块的激活强度
        
        Args:
            chunk_id: 组块ID
        """
        if chunk_id in self._activations:
            del self._activations[chunk_id]
            del self._last_update_times[chunk_id]
            logger.debug(f"Removed activation for chunk {chunk_id}")
    
    def get_activation(self, chunk_id: str) -> float:
        """
        获取指定组块的当前激活强度
        
        Args:
            chunk_id: 组块ID
        
        Returns:
            当前激活强度值
        """
        return self._activations.get(chunk_id, 0.0)
    
    def set_activation(self, chunk_id: str, activation: float, clamp: bool = True):
        """
        直接设置组块的激活强度
        
        Args:
            chunk_id: 组块ID
            activation: 新的激活强度值
            clamp: 是否裁剪到有效范围
        """
        if chunk_id not in self._activations:
            self.add_chunk(chunk_id, activation)
            return
        
        if clamp:
            activation = max(self.min_activation, min(self.max_activation, activation))
        
        self._activations[chunk_id] = activation
        self._last_update_times[chunk_id] = time.time()
    
    def decay_activations(self, delta_time: float) -> Dict[str, float]:
        """
        执行激活衰减
        
        衰减公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a)
        
        Args:
            delta_time: 时间间隔 Δt（秒）
        
        Returns:
            衰减后的激活强度字典
        """
        decay_factor = np.exp(-delta_time / self.decay_time_constant)
        
        decayed_activations = {}
        
        for chunk_id, activation in self._activations.items():
            # 应用衰减
            new_activation = activation * decay_factor
            
            # 保持最小激活阈值（历史信息保留）
            new_activation = max(self.min_activation, new_activation)
            
            # 裁剪到有效范围
            new_activation = min(self.max_activation, new_activation)
            
            decayed_activations[chunk_id] = new_activation
            self._activations[chunk_id] = new_activation
        
        logger.debug(
            f"Decayed activations: Δt={delta_time}s, "
            f"decay_factor={decay_factor:.4f}"
        )
        
        return decayed_activations
    
    def apply_input_drive(
        self,
        input_drive: Dict[str, float],
    ) -> Dict[str, float]:
        """
        应用输入驱动的激活增强
        
        公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a) + InputDrive_k(t)
        
        Args:
            input_drive: 输入驱动力字典，key为组块ID，value为驱动力强度
        
        Returns:
            更新后的激活强度字典
        """
        updated_activations = {}
        
        for chunk_id, drive in input_drive.items():
            if chunk_id not in self._activations:
                # 新组块，添加激活强度
                self.add_chunk(chunk_id, drive)
                updated_activations[chunk_id] = drive
                continue
            
            # 应用输入驱动
            current_activation = self._activations[chunk_id]
            new_activation = current_activation + drive
            
            # 裁剪到有效范围
            new_activation = max(self.min_activation, min(self.max_activation, new_activation))
            
            self._activations[chunk_id] = new_activation
            updated_activations[chunk_id] = new_activation
        
        logger.debug(
            f"Applied input drive to {len(input_drive)} chunks"
        )
        
        return updated_activations
    
    def decay_and_drive(
        self,
        delta_time: float,
        input_drive: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        组合衰减和输入驱动
        
        完整公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a) + InputDrive_k(t)
        
        Args:
            delta_time: 时间间隔 Δt（秒）
            input_drive: 输入驱动力字典（可选）
        
        Returns:
            更新后的激活强度字典
        """
        # 先衰减
        self.decay_activations(delta_time)
        
        # 再应用输入驱动
        if input_drive:
            self.apply_input_drive(input_drive)
        
        return self.get_all_activations()
    
    def get_all_activations(self) -> Dict[str, float]:
        """
        获取所有组块的激活强度
        
        Returns:
            激活强度字典
        """
        return self._activations.copy()
    
    def get_top_n_chunks(self, n: int) -> List[Tuple[str, float]]:
        """
        获取激活强度排名前N的组块
        
        Args:
            n: Top-N数量
        
        Returns:
            排名前N的 (chunk_id, activation) 元组列表
        """
        # 按激活强度排序
        sorted_chunks = sorted(
            self._activations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_chunks[:n]
    
    def get_bottom_chunks(self, threshold: float = None) -> List[Tuple[str, float]]:
        """
        获取低激活组块（低于阈值）
        
        Args:
            threshold: 激活阈值（默认使用 min_activation）
        
        Returns:
            低激活组块的 (chunk_id, activation) 元组列表
        """
        if threshold is None:
            threshold = self.min_activation
        
        low_activation_chunks = [
            (chunk_id, activation)
            for chunk_id, activation in self._activations.items()
            if activation <= threshold
        ]
        
        return low_activation_chunks
    
    def normalize_activations(self):
        """
        归一化激活强度（使总和为1）
        
        注意：这主要用于注意力分配，而不是替代衰减机制
        """
        total = sum(self._activations.values())
        if total > 0:
            for chunk_id in self._activations:
                self._activations[chunk_id] = self._activations[chunk_id] / total
    
    def batch_update(self, updates: Dict[str, float]):
        """
        批量更新激活强度
        
        Args:
            updates: 更新字典，key为组块ID，value为新激活强度
        """
        for chunk_id, activation in updates.items():
            self.set_activation(chunk_id, activation)
    
    def to_tensor(self) -> torch.Tensor:
        """
        将激活强度转换为张量
        
        Returns:
            激活强度张量
        """
        if not self._activations:
            return torch.tensor([], device=self.device)
        
        activations_list = list(self._activations.values())
        return torch.tensor(activations_list, dtype=torch.float32, device=self.device)
    
    def clear(self):
        """清空所有激活强度"""
        self._activations.clear()
        self._last_update_times.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取激活强度统计信息
        
        Returns:
            统计信息字典
        """
        if not self._activations:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "max": 0.0,
                "min": 0.0,
            }
        
        activations_tensor = self.to_tensor()
        return {
            "count": len(self._activations),
            "mean": activations_tensor.mean().item(),
            "std": activations_tensor.std().item(),
            "max": activations_tensor.max().item(),
            "min": activations_tensor.min().item(),
        }
    
    def __len__(self) -> int:
        """获取组块数量"""
        return len(self._activations)
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (
            f"ActivationStrength(count={stats['count']}, "
            f"mean={stats['mean']:.4f}, "
            f"τ_a={self.decay_time_constant}s, "
            f"ε={self.min_activation})"
        )


class WorkingMemory:
    """
    工作记忆系统
    
    完整工作记忆类，管理所有组块和激活强度
    
    核心机制：
    1. 组块动态生成（注意力绑定）
    2. 激活强度向量管理
    3. 激活衰减与更新机制
    4. Top-N组块选择（容量约束，米勒定律）
    5. 低激活组块信息保留
    6. 快速恢复机制
    
    Attributes:
        capacity: 工作记忆容量 N（默认7，米勒定律）
        chunks: 组块字典，key为chunk_id
        activation_strength: 激活强度管理器
        fast_dim: 快变量维度
        chunk_dim: 组块维度
        min_activation: 最小激活阈值 ε
        decay_time_constant: 衰减时间常数 τ_a
    """
    
    def __init__(
        self,
        capacity: int = 7,                    # 米勒定律: N=7±2
        fast_dim: int = 2048,                 # 快变量维度
        chunk_dim: int = 256,                 # 组块维度
        decay_time_constant: float = 10.0,    # 衰减时间常数 τ_a
        min_activation: float = 0.01,         # 最小激活阈值 ε
        device: str = "cpu",
    ):
        """
        初始化工作记忆系统
        
        Args:
            capacity: 工作记忆容量 N（5-9范围）
            fast_dim: 快变量维度
            chunk_dim: 组块维度
            decay_time_constant: 衰减时间常数 τ_a
            min_activation: 最小激活阈值 ε
            device: 计算设备
        """
        # 验证容量范围（米勒定律）
        if not 5 <= capacity <= 9:
            logger.warning(
                f"Capacity {capacity} outside Miller's law range (5-9), "
                f"clamping to valid range"
            )
            capacity = max(5, min(9, capacity))
        
        self.capacity = capacity
        self.fast_dim = fast_dim
        self.chunk_dim = chunk_dim
        self.device = device
        
        # 组块存储
        self._chunks: Dict[str, Chunk] = {}
        
        # 激活强度管理器
        self.activation_strength = ActivationStrength(
            decay_time_constant=decay_time_constant,
            min_activation=min_activation,
            device=device,
        )
        
        # 历史组块存储（用于快速恢复）
        self._history_chunks: Dict[str, Chunk] = {}
        
        # 统计信息
        self._stats = {
            "total_chunks_created": 0,
            "total_chunks_removed": 0,
            "total_chunks_restored": 0,
            "current_active_count": 0,
            "current_dormant_count": 0,
        }
        
        logger.info(
            f"WorkingMemory initialized: capacity={capacity}, "
            f"fast_dim={fast_dim}, chunk_dim={chunk_dim}, "
            f"τ_a={decay_time_constant}s, ε={min_activation}"
        )
    
    # ==================== 组块管理方法 ====================
    
    def create_chunk(
        self,
        source_state: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        chunk_type: ChunkType = ChunkType.TEMPORARY,
        initial_activation: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Chunk:
        """
        创建新组块（动态生成）
        
        组块形成公式：C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
        
        Args:
            source_state: 快变量张量 E_fast
            attention_weights: 注意力权重向量（可选，会自动生成）
            chunk_type: 组块类型
            initial_activation: 初始激活强度
            metadata: 额外元数据
        
        Returns:
            新创建的组块
        """
        # 生成唯一组块ID
        chunk_id = f"chunk_{self._stats['total_chunks_created']}"
        self._stats['total_chunks_created'] += 1
        
        # 如果没有提供注意力权重，自动生成
        if attention_weights is None:
            attention_weights = self._generate_attention_weights(
                source_state, chunk_type
            )
        
        # 计算组块内容：C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
        # 将快变量投影到组块维度
        content = self._compute_chunk_content(source_state, attention_weights)
        
        # 确定来源维度
        source_dimensions = self._determine_source_dimensions(attention_weights)
        
        # 创建组块
        chunk = Chunk(
            chunk_id=chunk_id,
            content=content,
            attention_weights=attention_weights,
            chunk_type=chunk_type,
            status=ChunkStatus.ACTIVE,
            creation_time=time.time(),
            last_update_time=time.time(),
            source_dimensions=source_dimensions,
            metadata=metadata or {},
        )
        
        # 添加到存储
        self._chunks[chunk_id] = chunk
        
        # 添加激活强度
        self.activation_strength.add_chunk(chunk_id, initial_activation)
        
        # 更新统计
        self._stats['current_active_count'] += 1
        
        # 检查容量约束
        self._enforce_capacity_constraint()
        
        logger.debug(
            f"Created chunk {chunk_id}: type={chunk_type.value}, "
            f"activation={initial_activation:.4f}"
        )
        
        return chunk
    
    def _generate_attention_weights(
        self,
        source_state: torch.Tensor,
        chunk_type: ChunkType,
    ) -> torch.Tensor:
        """
        自动生成注意力权重（注意力绑定机制）
        
        策略：
        1. 语义类组块：关注语义相关维度
        2. 情感类组块：关注情感相关维度
        3. 物理类组块：关注物理相关维度
        4. 其他类型：均匀分布
        
        Args:
            source_state: 快变量张量
            chunk_type: 组块类型
        
        Returns:
            注意力权重向量
        """
        # 将快变量投影到组块维度
        if source_state.shape[0] > self.chunk_dim:
            # 使用简单的线性投影
            projection_matrix = torch.randn(self.chunk_dim, source_state.shape[0])
            projection_matrix = projection_matrix / projection_matrix.norm(dim=1, keepdim=True)
            projected_state = projection_matrix @ source_state
        else:
            projected_state = source_state[:self.chunk_dim] if source_state.shape[0] >= self.chunk_dim else torch.cat([source_state, torch.zeros(self.chunk_dim - source_state.shape[0])])
        
        # 根据组块类型生成注意力权重
        if chunk_type == ChunkType.SEMANTIC:
            # 关注高方差维度（语义信息通常变化较大）
            weights = projected_state.abs() + 1e-6
        elif chunk_type == ChunkType.EMOTIONAL:
            # 关注极端值维度（情感状态通常表现为极端值）
            weights = projected_state.abs() ** 2 + 1e-6
        elif chunk_type == ChunkType.PHYSICAL:
            # 基于物理维度分配权重
            weights = torch.ones(self.chunk_dim)
            # 前半部分权重较高（假设物理信息集中在前面）
            weights[:self.chunk_dim // 2] = 2.0
        elif chunk_type == ChunkType.PROPRI0CEPTIVE:
            # 基于本体感觉维度分配权重
            weights = torch.ones(self.chunk_dim)
            # 后半部分权重较高（假设本体感觉集中在后面）
            weights[self.chunk_dim // 2:] = 2.0
        else:
            # 均匀分布
            weights = torch.ones(self.chunk_dim)
        
        # 归一化为概率分布
        weights = weights / weights.sum()
        
        return weights
    
    def _compute_chunk_content(
        self,
        source_state: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算组块内容
        
        公式：C_k(t) = Σ_i α_ki(t) · E_fast^(i)(t)
        
        Args:
            source_state: 快变量张量
            attention_weights: 注意力权重（已投影到组块维度）
        
        Returns:
            组块内容向量（维度为 chunk_dim）
        """
        # 如果快变量维度大于组块维度，需要投影
        if source_state.shape[0] > self.chunk_dim:
            # 使用线性投影将快变量压缩到组块维度
            # 这里使用简单的平均池化或随机投影
            # 为了数值稳定性，使用均匀分割
            chunk_size = source_state.shape[0] // self.chunk_dim
            remainder = source_state.shape[0] % self.chunk_dim
            
            # 创建分组并平均池化
            projected_state = torch.zeros(self.chunk_dim)
            for i in range(self.chunk_dim):
                start_idx = i * chunk_size + min(i, remainder)
                end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
                if end_idx <= source_state.shape[0]:
                    projected_state[i] = source_state[start_idx:end_idx].mean()
            
            source_state = projected_state
        elif source_state.shape[0] < self.chunk_dim:
            # 扩展快变量到组块维度
            source_state = torch.cat([
                source_state,
                torch.zeros(self.chunk_dim - source_state.shape[0])
            ])
        
        # 确保注意力权重维度匹配
        if attention_weights.shape[0] != self.chunk_dim:
            # 调整注意力权重维度
            if attention_weights.shape[0] > self.chunk_dim:
                # 压缩注意力权重
                attention_weights = attention_weights[:self.chunk_dim]
            else:
                # 扩展注意力权重
                attention_weights = torch.cat([
                    attention_weights,
                    torch.zeros(self.chunk_dim - attention_weights.shape[0])
                ])
            # 重新归一化
            if attention_weights.sum() > 0:
                attention_weights = attention_weights / attention_weights.sum()
            else:
                attention_weights = torch.ones(self.chunk_dim) / self.chunk_dim
        
        # 计算加权内容
        content = source_state * attention_weights
        
        return content
    
    def _determine_source_dimensions(
        self,
        attention_weights: torch.Tensor,
    ) -> Set[int]:
        """
        确定组块的来源维度（注意力聚焦的维度）
        
        Args:
            attention_weights: 注意力权重
        
        Returns:
            来源维度索引集合
        """
        # 选择权重较高的维度
        threshold = attention_weights.mean().item()
        source_dims = {
            i for i, w in enumerate(attention_weights.tolist())
            if w > threshold
        }
        return source_dims
    
    def update_chunk(
        self,
        chunk_id: str,
        new_content: Optional[torch.Tensor] = None,
        new_attention_weights: Optional[torch.Tensor] = None,
        new_activation: Optional[float] = None,
    ) -> Optional[Chunk]:
        """
        更新组块
        
        Args:
            chunk_id: 组块ID
            new_content: 新内容（可选）
            new_attention_weights: 新注意力权重（可选）
            new_activation: 新激活强度（可选）
        
        Returns:
            更新后的组块，如果组块不存在则返回 None
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            logger.warning(f"Chunk {chunk_id} not found for update")
            return None
        
        # 更新内容和注意力权重
        if new_content is not None or new_attention_weights is not None:
            content = new_content if new_content is not None else chunk.content
            attention_weights = new_attention_weights if new_attention_weights is not None else chunk.attention_weights
            chunk.update_content(content, attention_weights)
        
        # 更新激活强度
        if new_activation is not None:
            self.activation_strength.set_activation(chunk_id, new_activation)
            if new_activation > self.activation_strength.min_activation:
                chunk.status = ChunkStatus.ACTIVE
                self._stats['current_active_count'] += 1
                self._stats['current_dormant_count'] -= 1
            else:
                chunk.status = ChunkStatus.DORMANT
        
        return chunk
    
    def remove_chunk(self, chunk_id: str, save_to_history: bool = True):
        """
        移除组块
        
        Args:
            chunk_id: 组块ID
            save_to_history: 是否保存到历史存储（用于快速恢复）
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            logger.warning(f"Chunk {chunk_id} not found for removal")
            return
        
        # 保存到历史存储
        if save_to_history:
            self._history_chunks[chunk_id] = chunk.copy()
            logger.debug(f"Saved chunk {chunk_id} to history storage")
        
        # 从存储中移除
        del self._chunks[chunk_id]
        
        # 移除激活强度
        self.activation_strength.remove_chunk(chunk_id)
        
        # 更新统计
        self._stats['total_chunks_removed'] += 1
        if chunk.status == ChunkStatus.ACTIVE:
            self._stats['current_active_count'] -= 1
        elif chunk.status == ChunkStatus.DORMANT:
            self._stats['current_dormant_count'] -= 1
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """
        获取指定组块
        
        Args:
            chunk_id: 组块ID
        
        Returns:
            组块对象，如果不存在则返回 None
        """
        return self._chunks.get(chunk_id)
    
    def get_all_chunks(self) -> Dict[str, Chunk]:
        """
        获取所有组块
        
        Returns:
            组块字典
        """
        return self._chunks.copy()
    
    # ==================== 容量约束与选择 ====================
    
    def _enforce_capacity_constraint(self):
        """
        执行容量约束
        
        Top-N选择机制：仅激活强度排名前N的组块保持激活状态
        低激活组块转为休眠状态（信息保留）
        """
        # 获取Top-N组块
        top_n_chunks = self.activation_strength.get_top_n_chunks(self.capacity)
        top_n_ids = {chunk_id for chunk_id, _ in top_n_chunks}
        
        # 标记激活和休眠组块
        for chunk_id, chunk in self._chunks.items():
            if chunk_id in top_n_ids:
                chunk.status = ChunkStatus.ACTIVE
            else:
                chunk.status = ChunkStatus.DORMANT
        
        # 更新统计
        active_count = sum(1 for c in self._chunks.values() if c.status == ChunkStatus.ACTIVE)
        dormant_count = sum(1 for c in self._chunks.values() if c.status == ChunkStatus.DORMANT)
        self._stats['current_active_count'] = active_count
        self._stats['current_dormant_count'] = dormant_count
        
        logger.debug(
            f"Enforced capacity constraint: {active_count} active, "
            f"{dormant_count} dormant"
        )
    
    def get_active_chunks(self) -> List[Chunk]:
        """
        获取当前激活的组块（Top-N）
        
        Returns:
            激活组块列表
        """
        active_chunks = [
            chunk for chunk in self._chunks.values()
            if chunk.status == ChunkStatus.ACTIVE
        ]
        
        # 按激活强度排序
        active_chunks.sort(
            key=lambda c: self.activation_strength.get_activation(c.chunk_id),
            reverse=True
        )
        
        return active_chunks
    
    def get_dormant_chunks(self) -> List[Chunk]:
        """
        获取休眠的组块（历史信息保留）
        
        Returns:
            休眠组块列表
        """
        return [
            chunk for chunk in self._chunks.values()
            if chunk.status == ChunkStatus.DORMANT
        ]
    
    def select_top_n_chunks(self, n: Optional[int] = None) -> List[Tuple[Chunk, float]]:
        """
        选择Top-N组块
        
        Args:
            n: 选择数量（默认使用容量配置）
        
        Returns:
            (chunk, activation) 元组列表
        """
        if n is None:
            n = self.capacity
        
        top_n = self.activation_strength.get_top_n_chunks(n)
        
        result = []
        for chunk_id, activation in top_n:
            chunk = self._chunks.get(chunk_id)
            if chunk is not None:
                result.append((chunk, activation))
        
        return result
    
    # ==================== 激活衰减与更新 ====================
    
    def update_activations(
        self,
        delta_time: float,
        input_drive: Optional[Dict[str, float]] = None,
    ):
        """
        更新所有组块的激活强度
        
        完整公式：a_k(t+Δt) = a_k(t) · e^(-Δt/τ_a) + InputDrive_k(t)
        
        Args:
            delta_time: 时间间隔 Δt（秒）
            input_drive: 输入驱动力字典（可选）
        """
        # 执行衰减和输入驱动
        self.activation_strength.decay_and_drive(delta_time, input_drive)
        
        # 更新组块状态
        for chunk_id, chunk in self._chunks.items():
            activation = self.activation_strength.get_activation(chunk_id)
            
            if activation <= self.activation_strength.min_activation:
                chunk.status = ChunkStatus.DORMANT
            elif chunk.status == ChunkStatus.DORMANT and activation > self.activation_strength.min_activation:
                chunk.status = ChunkStatus.RECOVERING
        
        # 执行容量约束
        self._enforce_capacity_constraint()
        
        logger.debug(
            f"Updated activations: Δt={delta_time}s, "
            f"input_drive_count={len(input_drive) if input_drive else 0}"
        )
    
    def boost_chunk_activation(
        self,
        chunk_id: str,
        boost_amount: float = 0.5,
    ):
        """
        增强指定组块的激活强度
        
        Args:
            chunk_id: 组块ID
            boost_amount: 增强量
        """
        current_activation = self.activation_strength.get_activation(chunk_id)
        new_activation = current_activation + boost_amount
        self.activation_strength.set_activation(chunk_id, new_activation)
        
        # 如果组块是休眠状态，触发恢复
        chunk = self._chunks.get(chunk_id)
        if chunk and chunk.status == ChunkStatus.DORMANT:
            chunk.status = ChunkStatus.RECOVERING
        
        logger.debug(
            f"Boosted chunk {chunk_id} activation: "
            f"{current_activation:.4f} -> {new_activation:.4f}"
        )
    
    # ==================== 快速恢复机制 ====================
    
    def restore_chunk(self, chunk_id: str, initial_activation: float = 0.5) -> Optional[Chunk]:
        """
        从历史存储恢复组块
        
        Args:
            chunk_id: 组块ID
            initial_activation: 恢复后的初始激活强度
        
        Returns:
            恢复的组块，如果不存在则返回 None
        """
        # 检查当前是否已存在
        if chunk_id in self._chunks:
            logger.warning(f"Chunk {chunk_id} already exists, boosting instead")
            self.boost_chunk_activation(chunk_id, initial_activation)
            return self._chunks[chunk_id]
        
        # 从历史存储获取
        chunk = self._history_chunks.get(chunk_id)
        if chunk is None:
            logger.warning(f"Chunk {chunk_id} not found in history storage")
            return None
        
        # 恢复组块
        restored_chunk = chunk.copy()
        restored_chunk.status = ChunkStatus.RECOVERING
        restored_chunk.last_update_time = time.time()
        
        self._chunks[chunk_id] = restored_chunk
        self.activation_strength.add_chunk(chunk_id, initial_activation)
        
        # 更新统计
        self._stats['total_chunks_restored'] += 1
        
        # 执行容量约束
        self._enforce_capacity_constraint()
        
        logger.info(
            f"Restored chunk {chunk_id} with activation {initial_activation:.4f}"
        )
        
        return restored_chunk
    
    def restore_chunks_by_pattern(
        self,
        pattern_type: ChunkType,
        max_restore: int = 3,
        initial_activation: float = 0.3,
    ) -> List[Chunk]:
        """
        按类型模式恢复组块
        
        Args:
            pattern_type: 组块类型
            max_restore: 最大恢复数量
            initial_activation: 初始激活强度
        
        Returns:
            恢复的组块列表
        """
        # 从历史存储中找到匹配类型的组块
        matching_chunks = [
            chunk for chunk in self._history_chunks.values()
            if chunk.chunk_type == pattern_type
        ]
        
        # 按创建时间排序（最近创建的优先）
        matching_chunks.sort(key=lambda c: c.creation_time, reverse=True)
        
        restored = []
        for chunk in matching_chunks[:max_restore]:
            restored_chunk = self.restore_chunk(chunk.chunk_id, initial_activation)
            if restored_chunk:
                restored.append(restored_chunk)
        
        return restored
    
    def get_history_chunks(self) -> Dict[str, Chunk]:
        """
        获取历史存储中的组块
        
        Returns:
            历史组块字典
        """
        return self._history_chunks.copy()
    
    # ==================== 组块合并与分解 ====================
    
    def merge_chunks(
        self,
        chunk_ids: List[str],
        merged_chunk_type: ChunkType = ChunkType.HYBRID,
    ) -> Optional[Chunk]:
        """
        合并多个组块
        
        Args:
            chunk_ids: 要合并的组块ID列表
            merged_chunk_type: 合并后的组块类型
        
        Returns:
            合合后的新组块
        """
        if len(chunk_ids) < 2:
            logger.warning("Need at least 2 chunks to merge")
            return None
        
        # 获取所有要合并的组块
        chunks_to_merge = []
        for chunk_id in chunk_ids:
            chunk = self._chunks.get(chunk_id)
            if chunk is None:
                logger.warning(f"Chunk {chunk_id} not found for merging")
                continue
            chunks_to_merge.append(chunk)
        
        if len(chunks_to_merge) < 2:
            logger.warning("Not enough valid chunks to merge")
            return None
        
        # 合合内容和注意力权重
        merged_content = torch.stack([c.content for c in chunks_to_merge]).mean(dim=0)
        merged_attention = torch.stack([c.attention_weights for c in chunks_to_merge]).mean(dim=0)
        merged_attention = merged_attention / merged_attention.sum()
        
        # 合合激活强度（加权平均）
        total_activation = sum(
            self.activation_strength.get_activation(c.chunk_id)
            for c in chunks_to_merge
        )
        merged_activation = total_activation / len(chunks_to_merge)
        
        # 合合来源维度
        merged_dimensions = set()
        for c in chunks_to_merge:
            merged_dimensions.update(c.source_dimensions)
        
        # 创建新组块
        merged_chunk = self.create_chunk(
            source_state=merged_content,
            attention_weights=merged_attention,
            chunk_type=merged_chunk_type,
            initial_activation=merged_activation,
            metadata={
                "merged_from": chunk_ids,
                "merge_time": time.time(),
            },
        )
        
        # 移除原组块（保存到历史）
        for chunk_id in chunk_ids:
            if chunk_id != merged_chunk.chunk_id:
                self.remove_chunk(chunk_id, save_to_history=True)
        
        logger.info(f"Merged {len(chunks_to_merge)} chunks into {merged_chunk.chunk_id}")
        
        return merged_chunk
    
    def split_chunk(
        self,
        chunk_id: str,
        split_pattern: Optional[List[torch.Tensor]] = None,
    ) -> List[Chunk]:
        """
        分解组块
        
        Args:
            chunk_id: 要分解的组块ID
            split_pattern: 分解模式（可选）
        
        Returns:
            分解后的组块列表
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            logger.warning(f"Chunk {chunk_id} not found for splitting")
            return []
        
        # 默认分解为两部分
        if split_pattern is None:
            # 基于注意力权重分解
            # 确保 content 和 attention_weights 维度匹配
            actual_dim = chunk.content.shape[0]
            attention_weights = chunk.attention_weights
            
            # 如果维度不匹配，调整 attention_weights
            if attention_weights.shape[0] != actual_dim:
                if attention_weights.shape[0] > actual_dim:
                    attention_weights = attention_weights[:actual_dim]
                else:
                    attention_weights = torch.cat([
                        attention_weights,
                        torch.ones(actual_dim - attention_weights.shape[0]) / (actual_dim - attention_weights.shape[0])
                    ])
                attention_weights = attention_weights / attention_weights.sum()
            
            high_attention_indices = attention_weights > attention_weights.mean()
            low_attention_indices = ~high_attention_indices
            
            split_pattern = [
                chunk.content * high_attention_indices.float(),
                chunk.content * low_attention_indices.float(),
            ]
        
        # 创建分解后的组块
        split_chunks = []
        original_activation = self.activation_strength.get_activation(chunk_id)
        split_activation = original_activation / len(split_pattern)
        
        for i, pattern in enumerate(split_pattern):
            new_chunk = self.create_chunk(
                source_state=pattern,
                chunk_type=chunk.chunk_type,
                initial_activation=split_activation,
                metadata={
                    "split_from": chunk_id,
                    "split_index": i,
                    "split_time": time.time(),
                },
            )
            split_chunks.append(new_chunk)
        
        # 移除原组块
        self.remove_chunk(chunk_id, save_to_history=True)
        
        logger.info(f"Split chunk {chunk_id} into {len(split_chunks)} chunks")
        
        return split_chunks
    
    # ==================== 状态保存与加载 ====================
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化工作记忆状态
        
        Returns:
            状态字典
        """
        return {
            "capacity": self.capacity,
            "fast_dim": self.fast_dim,
            "chunk_dim": self.chunk_dim,
            "decay_time_constant": self.activation_strength.decay_time_constant,
            "min_activation": self.activation_strength.min_activation,
            "chunks": {
                chunk_id: chunk.to_dict()
                for chunk_id, chunk in self._chunks.items()
            },
            "activations": self.activation_strength.get_all_activations(),
            "history_chunks": {
                chunk_id: chunk.to_dict()
                for chunk_id, chunk in self._history_chunks.items()
            },
            "stats": self._stats.copy(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkingMemory":
        """
        从字典恢复工作记忆状态
        
        Args:
            data: 状态字典
        
        Returns:
            工作记忆实例
        """
        wm = cls(
            capacity=data.get("capacity", 7),
            fast_dim=data.get("fast_dim", 2048),
            chunk_dim=data.get("chunk_dim", 256),
            decay_time_constant=data.get("decay_time_constant", 10.0),
            min_activation=data.get("min_activation", 0.01),
        )
        
        # 恢复组块
        for chunk_id, chunk_data in data.get("chunks", {}).items():
            chunk = Chunk.from_dict(chunk_data)
            wm._chunks[chunk_id] = chunk
        
        # 恢复激活强度
        for chunk_id, activation in data.get("activations", {}).items():
            wm.activation_strength.add_chunk(chunk_id, activation)
        
        # 恢复历史组块
        for chunk_id, chunk_data in data.get("history_chunks", {}).items():
            chunk = Chunk.from_dict(chunk_data)
            wm._history_chunks[chunk_id] = chunk
        
        # 恢复统计
        wm._stats = data.get("stats", wm._stats)
        
        return wm
    
    def save_state(self, filepath: str):
        """
        保存状态到文件
        
        Args:
            filepath: 文件路径
        """
        import json
        from pathlib import Path
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        state_dict = self.to_dict()
        
        # 转换张量为列表
        for chunk_id, chunk_data in state_dict["chunks"].items():
            chunk_data["content"] = [float(x) for x in chunk_data["content"]]
            chunk_data["attention_weights"] = [float(x) for x in chunk_data["attention_weights"]]
        
        for chunk_id, chunk_data in state_dict["history_chunks"].items():
            chunk_data["content"] = [float(x) for x in chunk_data["content"]]
            chunk_data["attention_weights"] = [float(x) for x in chunk_data["attention_weights"]]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved working memory state to {filepath}")
    
    def load_state(self, filepath: str):
        """
        从文件加载状态
        
        Args:
            filepath: 文件路径
        """
        import json
        
        with open(filepath, 'r', encoding='utf-8') as f:
            state_dict = json.load(f)
        
        # 恢复状态
        restored_wm = WorkingMemory.from_dict(state_dict)
        
        # 复制恢复的状态
        self.capacity = restored_wm.capacity
        self._chunks = restored_wm._chunks
        self.activation_strength = restored_wm.activation_strength
        self._history_chunks = restored_wm._history_chunks
        self._stats = restored_wm._stats
        
        logger.info(f"Loaded working memory state from {filepath}")
    
    # ==================== 统计与验证 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取工作记忆统计信息
        
        Returns:
            统计信息字典
        """
        activation_stats = self.activation_strength.get_statistics()
        
        return {
            "capacity": self.capacity,
            "current_chunks": len(self._chunks),
            "active_chunks": self._stats['current_active_count'],
            "dormant_chunks": self._stats['current_dormant_count'],
            "history_chunks": len(self._history_chunks),
            "activation_stats": activation_stats,
            "creation_stats": {
                "total_created": self._stats['total_chunks_created'],
                "total_removed": self._stats['total_chunks_removed'],
                "total_restored": self._stats['total_chunks_restored'],
            },
        }
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证工作记忆系统
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证容量约束
        active_count = self._stats['current_active_count']
        if active_count > self.capacity:
            errors.append(
                f"Active chunk count {active_count} exceeds capacity {self.capacity}"
            )
        
        # 验证米勒定律范围
        if not 5 <= self.capacity <= 9:
            errors.append(
                f"Capacity {self.capacity} violates Miller's law (5-9 range)"
            )
        
        # 验证每个组块
        for chunk_id, chunk in self._chunks.items():
            is_valid, chunk_errors = chunk.validate()
            if not is_valid:
                errors.extend([f"Chunk {chunk_id}: {e}" for e in chunk_errors])
        
        # 验证激活强度一致性
        for chunk_id in self._chunks:
            if chunk_id not in self.activation_strength._activations:
                errors.append(f"Chunk {chunk_id} missing activation strength")
        
        for chunk_id in self.activation_strength._activations:
            if chunk_id not in self._chunks and chunk_id not in self._history_chunks:
                errors.append(f"Activation for unknown chunk {chunk_id}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_capacity_constraint(self) -> Tuple[bool, Dict[str, Any]]:
        """
        验证容量约束的正确性
        
        Returns:
            (is_valid, details): 是否满足米勒定律以及详细信息
        """
        active_chunks = self.get_active_chunks()
        
        details = {
            "capacity": self.capacity,
            "active_count": len(active_chunks),
            "satisfies_miller_law": 5 <= self.capacity <= 9,
            "active_within_capacity": len(active_chunks) <= self.capacity,
            "chunk_types": {
                chunk_type.value: sum(
                    1 for c in active_chunks if c.chunk_type == chunk_type
                )
                for chunk_type in ChunkType
            },
        }
        
        is_valid = (
            details["satisfies_miller_law"] and
            details["active_within_capacity"]
        )
        
        return is_valid, details
    
    # ==================== 辅助方法 ====================
    
    def compute_working_memory_output(self) -> torch.Tensor:
        """
        计算工作记忆的输出（用于积分引擎）
        
        Returns:
            工作记忆输出张量（维度为 chunk_dim）
        """
        # 获取激活组块
        active_chunks = self.get_active_chunks()
        
        if not active_chunks:
            return torch.zeros(self.chunk_dim, device=self.device)
        
        # 计算加权输出
        output = torch.zeros(self.chunk_dim, device=self.device)
        
        for chunk in active_chunks:
            activation = self.activation_strength.get_activation(chunk.chunk_id)
            weighted_content = chunk.compute_weighted_content()
            
            # 确保 weighted_content 维度与 chunk_dim 匹配
            if weighted_content.shape[0] != self.chunk_dim:
                if weighted_content.shape[0] > self.chunk_dim:
                    # 压缩到 chunk_dim
                    weighted_content = weighted_content[:self.chunk_dim]
                else:
                    # 扩展到 chunk_dim
                    weighted_content = torch.cat([
                        weighted_content,
                        torch.zeros(self.chunk_dim - weighted_content.shape[0], device=self.device)
                    ])
            
            output = output + weighted_content * activation
        
        return output
    
    def clear(self):
        """清空工作记忆"""
        self._chunks.clear()
        self._history_chunks.clear()
        self.activation_strength.clear()
        self._stats = {
            "total_chunks_created": 0,
            "total_chunks_removed": 0,
            "total_chunks_restored": 0,
            "current_active_count": 0,
            "current_dormant_count": 0,
        }
        
        logger.info("Working memory cleared")
    
    def __len__(self) -> int:
        """获取当前组块数量"""
        return len(self._chunks)
    
    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_statistics()
        return (
            f"WorkingMemory(capacity={stats['capacity']}, "
            f"chunks={stats['current_chunks']}, "
            f"active={stats['active_chunks']}, "
            f"dormant={stats['dormant_chunks']}, "
            f"history={stats['history_chunks']})"
        )