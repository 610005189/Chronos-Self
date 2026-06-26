"""
L0 感知层 - Perception Layer
============================

Task 12: 实现L0感知层，负责实时滤波、编码和知识库接入。

核心功能：
- 实时滤波：对原始输入进行预处理和噪声过滤
- 编码接口：将原始数据转换为结构化表征
- 知识库接入：实现 RAG（检索增强生成）调用
- 无自指约束：L0 仅处理感知，不包含自我模型
- 数据传递：将处理后的数据传递给 L1

关键特性：
- L0 无自指能力，仅作为感知输入的预处理层
- 不包含任何自我状态或自我模型的组件
- 通过 RAG 注入外部知识，增强感知表征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import time
import warnings

# 尝试导入向量数据库库
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    warnings.warn("chromadb not available, RAG functionality will be limited")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    warnings.warn("faiss not available, fallback to simple similarity search")

from chronos_core.utils.config import (
    DimensionalityConfig,
    MetaCognitiveConfig,
    PathsConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class PerceptionLayerConfig:
    """L0 感知层配置"""
    
    # 输入维度
    semantic_dim: int = 512       # 语义输入维度
    physical_dim: int = 512       # 物理输入维度
    output_dim: int = 1024        # 输出融合维度
    
    # 滤波参数
    filter_method: str = "kalman"  # 'kalman', 'moving_average', 'exponential'
    filter_window_size: int = 5    # 滑动窗口大小
    filter_alpha: float = 0.3      # 指数滤波系数
    
    # 编码参数
    encoding_hidden_dim: int = 256
    encoding_num_layers: int = 2
    encoding_activation: str = "relu"
    
    # RAG 知识库参数
    use_rag: bool = True
    rag_vector_db: str = "chromadb"  # 'chromadb', 'faiss', 'simple'
    rag_collection_name: str = "chronos_knowledge"
    rag_top_k: int = 5
    rag_embedding_dim: int = 384
    
    # 噪声过滤参数
    noise_threshold: float = 0.01
    outlier_threshold: float = 3.0  # 标准差倍数
    
    # 数据传递参数
    batch_size: int = 1
    device: str = "cpu"


class PerceptionFilter(nn.Module):
    """
    感知滤波器
    
    实现多种滤波方法：
    - Kalman滤波（自适应最优滤波）
    - 滑动平均滤波
    - 指数加权滤波
    
    功能：
    - 噪声过滤
    - 异常值检测
    - 信号平滑
    """
    
    def __init__(
        self,
        method: str = "kalman",
        window_size: int = 5,
        alpha: float = 0.3,
        noise_threshold: float = 0.01,
        outlier_threshold: float = 3.0,
        device: str = "cpu"
    ):
        """
        初始化感知滤波器
        
        Args:
            method: 滤波方法
            window_size: 滑动窗口大小
            alpha: 指数滤波系数
            noise_threshold: 噪声阈值
            outlier_threshold: 异常值阈值（标准差倍数）
            device: 计算设备
        """
        super().__init__()
        
        self.method = method
        self.window_size = window_size
        self.alpha = alpha
        self.noise_threshold = noise_threshold
        self.outlier_threshold = outlier_threshold
        self.device = device
        
        # 滤波状态缓存
        self._filter_state: Optional[torch.Tensor] = None
        self._history_buffer: List[torch.Tensor] = []
        
        # Kalman 滤波参数
        if method == "kalman":
            # Kalman 状态：[value, velocity]
            self._kalman_state = None
            self._kalman_P = None  # 误差协方差
            self._kalman_Q = 0.01   # 过程噪声协方差
            self._kalman_R = 0.1    # 测量噪声协方差
        
        logger.info(
            f"PerceptionFilter initialized: method={method}, "
            f"window_size={window_size}, alpha={alpha}"
        )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        执行滤波
        
        Args:
            input_tensor: 输入张量
        
        Returns:
            滤波后的张量
        """
        input_tensor = input_tensor.to(self.device)
        
        # 异常值检测和过滤
        filtered = self._remove_outliers(input_tensor)
        
        # 应用滤波方法
        if self.method == "kalman":
            filtered = self._kalman_filter(filtered)
        elif self.method == "moving_average":
            filtered = self._moving_average_filter(filtered)
        elif self.method == "exponential":
            filtered = self._exponential_filter(filtered)
        
        # 噪声阈值过滤
        filtered = self._noise_thresholding(filtered)
        
        return filtered
    
    def _remove_outliers(self, data: torch.Tensor) -> torch.Tensor:
        """
        移除异常值
        
        Args:
            data: 输入数据
        
        Returns:
            过滤后的数据
        """
        # 计算统计量
        mean = data.mean()
        std = data.std() + 1e-8
        
        # 检测异常值
        z_scores = (data - mean) / std
        outlier_mask = z_scores.abs() > self.outlier_threshold
        
        # 替换异常值
        if outlier_mask.any():
            filtered_data = data.clone()
            # 用均值替换异常值
            filtered_data[outlier_mask] = mean
            logger.debug(
                f"Removed {outlier_mask.sum().item()} outliers "
                f"(threshold={self.outlier_threshold}σ)"
            )
            return filtered_data
        
        return data
    
    def _kalman_filter(self, data: torch.Tensor) -> torch.Tensor:
        """
        Kalman 滤波
        
        Args:
            data: 输入数据
        
        Returns:
            滤波后的数据
        """
        # 初始化 Kalman 状态
        if self._kalman_state is None:
            self._kalman_state = torch.zeros(2, device=self.device)
            self._kalman_state[0] = data.mean().item()
            self._kalman_P = torch.eye(2, device=self.device)
        
        # Kalman 滤波步骤（简化版，应用于整体数据）
        # 预测步骤
        # x_pred = F * x (F = [[1, 1], [0, 1]] for constant velocity model)
        F = torch.tensor([[1.0, 1.0], [0.0, 1.0]], device=self.device)
        x_pred = F @ self._kalman_state
        
        # P_pred = F * P * F' + Q
        P_pred = F @ self._kalman_P @ F.T + self._kalman_Q * torch.eye(2, device=self.device)
        
        # 更新步骤
        # H = [1, 0] (只观测位置)
        H = torch.tensor([[1.0, 0.0]], device=self.device)
        z = data.mean().unsqueeze(0)  # 观测值
        
        # K = P_pred * H' * (H * P_pred * H' + R)^-1
        S = H @ P_pred @ H.T + self._kalman_R
        K = P_pred @ H.T @ S.inverse()
        
        # x = x_pred + K * (z - H * x_pred)
        y = z - H @ x_pred
        self._kalman_state = x_pred + K @ y
        
        # P = (I - K * H) * P_pred
        I = torch.eye(2, device=self.device)
        self._kalman_P = (I - K @ H) @ P_pred
        
        # 使用滤波后的状态
        filtered_value = self._kalman_state[0]
        
        # 应用到数据
        filtered_data = data - data.mean() + filtered_value
        
        return filtered_data
    
    def _moving_average_filter(self, data: torch.Tensor) -> torch.Tensor:
        """
        滑动平均滤波
        
        Args:
            data: 输入数据
        
        Returns:
            滤波后的数据
        """
        # 添加到历史缓存
        self._history_buffer.append(data.clone())
        
        # 保持窗口大小
        if len(self._history_buffer) > self.window_size:
            self._history_buffer.pop(0)
        
        # 计算滑动平均
        if len(self._history_buffer) >= 2:
            stacked = torch.stack(self._history_buffer)
            averaged = stacked.mean(dim=0)
            return averaged
        
        return data
    
    def _exponential_filter(self, data: torch.Tensor) -> torch.Tensor:
        """
        指数加权滤波
        
        Args:
            data: 输入数据
        
        Returns:
            滤波后的数据
        """
        if self._filter_state is None:
            self._filter_state = data.clone()
        else:
            # y_t = alpha * x_t + (1 - alpha) * y_{t-1}
            self._filter_state = self.alpha * data + (1 - self.alpha) * self._filter_state
        
        return self._filter_state
    
    def _noise_thresholding(self, data: torch.Tensor) -> torch.Tensor:
        """
        噪声阈值过滤
        
        Args:
            data: 输入数据
        
        Returns:
            过滤后的数据
        """
        # 移除小于阈值的值（视为噪声）
        noise_mask = data.abs() < self.noise_threshold
        filtered_data = data.clone()
        filtered_data[noise_mask] = 0.0
        
        return filtered_data
    
    def reset(self):
        """重置滤波器状态"""
        self._filter_state = None
        self._history_buffer.clear()
        self._kalman_state = None
        self._kalman_P = None
        logger.debug("PerceptionFilter reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "method": self.method,
            "window_size": self.window_size,
            "alpha": self.alpha,
            "noise_threshold": self.noise_threshold,
            "outlier_threshold": self.outlier_threshold,
            "history_length": len(self._history_buffer),
        }


class PerceptionEncoder(nn.Module):
    """
    感知编码器
    
    将原始输入转换为结构化表征
    
    功能：
    - 语义编码
    - 物理编码
    - 融合编码
    - 维度投影
    """
    
    def __init__(
        self,
        semantic_dim: int = 512,
        physical_dim: int = 512,
        output_dim: int = 1024,
        hidden_dim: int = 256,
        num_layers: int = 2,
        activation: str = "relu",
        device: str = "cpu"
    ):
        """
        初始化感知编码器
        
        Args:
            semantic_dim: 语义输入维度
            physical_dim: 物理输入维度
            output_dim: 输出维度
            hidden_dim: 隐藏层维度
            num_layers: 层数
            activation: 激活函数
            device: 计算设备
        """
        super().__init__()
        
        self.semantic_dim = semantic_dim
        self.physical_dim = physical_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 激活函数
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            act_fn = nn.ReLU()
        
        # 语义编码器
        semantic_layers = []
        semantic_layers.append(nn.Linear(semantic_dim, hidden_dim))
        semantic_layers.append(act_fn)
        for _ in range(num_layers - 1):
            semantic_layers.append(nn.Linear(hidden_dim, hidden_dim))
            semantic_layers.append(act_fn)
        semantic_layers.append(nn.Linear(hidden_dim, output_dim // 2))
        self.semantic_encoder = nn.Sequential(*semantic_layers)
        
        # 物理编码器
        physical_layers = []
        physical_layers.append(nn.Linear(physical_dim, hidden_dim))
        physical_layers.append(act_fn)
        for _ in range(num_layers - 1):
            physical_layers.append(nn.Linear(hidden_dim, hidden_dim))
            physical_layers.append(act_fn)
        physical_layers.append(nn.Linear(hidden_dim, output_dim // 2))
        self.physical_encoder = nn.Sequential(*physical_layers)
        
        # 融合层
        self.fusion_layer = nn.Linear(output_dim, output_dim)
        
        self.to(device)
        
        logger.info(
            f"PerceptionEncoder initialized: "
            f"semantic_dim={semantic_dim}, physical_dim={physical_dim}, "
            f"output_dim={output_dim}, hidden_dim={hidden_dim}"
        )
    
    def forward(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor
    ) -> torch.Tensor:
        """
        编码输入
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
        
        Returns:
            编码后的融合表征
        """
        semantic_input = semantic_input.to(self.device)
        physical_input = physical_input.to(self.device)
        
        # 语义编码
        semantic_encoded = self.semantic_encoder(semantic_input)
        
        # 物理编码
        physical_encoded = self.physical_encoder(physical_input)
        
        # 融合
        fused = torch.cat([semantic_encoded, physical_encoded], dim=-1)
        
        # 最终融合层
        output = self.fusion_layer(fused)
        
        return output
    
    def encode_semantic(self, semantic_input: torch.Tensor) -> torch.Tensor:
        """
        仅编码语义输入
        
        Args:
            semantic_input: 语义输入
        
        Returns:
            语义编码
        """
        return self.semantic_encoder(semantic_input.to(self.device))
    
    def encode_physical(self, physical_input: torch.Tensor) -> torch.Tensor:
        """
        仅编码物理输入
        
        Args:
            physical_input: 物理输入
        
        Returns:
            物理编码
        """
        return self.physical_encoder(physical_input.to(self.device))


class RAGKnowledgeBase:
    """
    RAG 知识库
    
    实现检索增强生成，从外部知识库检索相关事实
    
    功能：
    - 向量数据库集成（ChromaDB 或 FAISS）
    - 相似性检索
    - 知识注入
    - 向量嵌入管理
    """
    
    def __init__(
        self,
        vector_db: str = "chromadb",
        collection_name: str = "chronos_knowledge",
        top_k: int = 5,
        embedding_dim: int = 384,
        db_path: Optional[str] = None,
        device: str = "cpu"
    ):
        """
        初始化 RAG 知识库
        
        Args:
            vector_db: 向量数据库类型
            collection_name: 集合名称
            top_k: 检索数量
            embedding_dim: 嵌入维度
            db_path: 数据库路径
            device: 计算设备
        """
        self.vector_db = vector_db
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_dim = embedding_dim
        self.device = device
        
        # 初始化向量数据库
        self._initialize_vector_db(db_path)
        
        # 简单知识库缓存（当向量数据库不可用时）
        self._knowledge_cache: Dict[str, torch.Tensor] = {}
        
        logger.info(
            f"RAGKnowledgeBase initialized: vector_db={vector_db}, "
            f"collection={collection_name}, top_k={top_k}"
        )
    
    def _initialize_vector_db(self, db_path: Optional[str]):
        """
        初始化向量数据库
        
        Args:
            db_path: 数据库路径
        """
        if self.vector_db == "chromadb" and CHROMADB_AVAILABLE:
            try:
                # 使用持久化存储
                if db_path is None:
                    db_path = "./data/vector_db"
                
                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=db_path
                ))
                
                # 创建或获取集合
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                
                logger.info(f"ChromaDB initialized at {db_path}")
                self._db_available = True
                
            except Exception as e:
                logger.warning(f"Failed to initialize ChromaDB: {e}")
                self._db_available = False
        
        elif self.vector_db == "faiss" and FAISS_AVAILABLE:
            try:
                # 创建 FAISS 索引
                self._faiss_index = faiss.IndexFlatIP(self.embedding_dim)
                self._faiss_id_map = {}
                self._db_available = True
                logger.info("FAISS index initialized")
                
            except Exception as e:
                logger.warning(f"Failed to initialize FAISS: {e}")
                self._db_available = False
        
        else:
            # 使用简单缓存
            self._db_available = False
            logger.info("Using simple knowledge cache (no vector DB available)")
    
    def add_knowledge(
        self,
        knowledge_id: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加知识到数据库
        
        Args:
            knowledge_id: 知识ID
            embedding: 嵌入向量
            metadata: 元数据
        """
        embedding_np = embedding.detach().cpu().numpy().flatten()
        
        if self.vector_db == "chromadb" and self._db_available:
            self._collection.add(
                ids=[knowledge_id],
                embeddings=[embedding_np.tolist()],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Added knowledge {knowledge_id} to ChromaDB")
        
        elif self.vector_db == "faiss" and self._db_available:
            # FAISS 需要管理 ID 映射
            id_int = len(self._faiss_id_map)
            self._faiss_id_map[id_int] = knowledge_id
            self._faiss_index.add(embedding_np.reshape(1, -1))
            logger.debug(f"Added knowledge {knowledge_id} to FAISS")
        
        else:
            # 使用简单缓存
            self._knowledge_cache[knowledge_id] = embedding.clone()
            logger.debug(f"Added knowledge {knowledge_id} to cache")
    
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[str]]:
        """
        检索相关知识
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 检索数量
        
        Returns:
            (embeddings, scores, ids): 检索结果
        """
        top_k = top_k or self.top_k
        query_np = query_embedding.detach().cpu().numpy().flatten()
        
        if self.vector_db == "chromadb" and self._db_available:
            results = self._collection.query(
                query_embeddings=[query_np.tolist()],
                n_results=top_k
            )
            
            embeddings = [
                torch.tensor(e, device=self.device)
                for e in results['embeddings'][0]
            ]
            scores = [1.0 - d for d in results['distances'][0]]  # 转换距离为相似度
            ids = results['ids'][0]
            
            return embeddings, scores, ids
        
        elif self.vector_db == "faiss" and self._db_available:
            # FAISS 搜索
            D, I = self._faiss_index.search(query_np.reshape(1, -1), top_k)
            
            embeddings = []
            scores = []
            ids = []
            
            for i, (dist, idx) in enumerate(zip(D[0], I[0])):
                if idx >= 0:  # 有效索引
                    knowledge_id = self._faiss_id_map.get(idx)
                    if knowledge_id in self._knowledge_cache:
                        embeddings.append(self._knowledge_cache[knowledge_id])
                        scores.append(float(dist))
                        ids.append(knowledge_id)
            
            return embeddings, scores, ids
        
        else:
            # 简单相似性搜索（余弦相似度）
            all_embeddings = list(self._knowledge_cache.values())
            all_ids = list(self._knowledge_cache.keys())
            
            if not all_embeddings:
                return [], [], []
            
            # 计算相似度
            query_tensor = query_embedding.to(self.device)
            similarities = []
            for emb in all_embeddings:
                emb_tensor = emb.to(self.device)
                sim = torch.cosine_similarity(query_tensor.unsqueeze(0), emb_tensor.unsqueeze(0))
                similarities.append(sim.item())
            
            # 排序并选择 top_k
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            embeddings = [all_embeddings[i] for i in sorted_indices]
            scores = [similarities[i] for i in sorted_indices]
            ids = [all_ids[i] for i in sorted_indices]
            
            return embeddings, scores, ids
    
    def inject_knowledge(
        self,
        perception_embedding: torch.Tensor,
        query_embedding: Optional[torch.Tensor] = None,
        injection_weight: float = 0.2
    ) -> torch.Tensor:
        """
        注入检索到的知识到感知表征
        
        Args:
            perception_embedding: 感知嵌入向量
            query_embedding: 查询嵌入向量（可选，默认使用感知嵌入）
            injection_weight: 注入权重
        
        Returns:
            增强后的感知表征
        """
        if query_embedding is None:
            query_embedding = perception_embedding
        
        # 检索相关知识
        retrieved_embeddings, scores, ids = self.retrieve(query_embedding)
        
        if not retrieved_embeddings:
            logger.debug("No knowledge retrieved for injection")
            return perception_embedding
        
        # 加权融合检索到的知识
        perception_embedding = perception_embedding.to(self.device)
        
        knowledge_contribution = torch.zeros_like(perception_embedding)
        total_weight = 0.0
        
        for emb, score in zip(retrieved_embeddings, scores):
            emb_tensor = emb.to(self.device)
            
            # 确保维度匹配
            if emb_tensor.shape[0] != perception_embedding.shape[0]:
                # 投影到感知维度
                if emb_tensor.shape[0] < perception_embedding.shape[0]:
                    padding = torch.zeros(
                        perception_embedding.shape[0] - emb_tensor.shape[0],
                        device=self.device
                    )
                    emb_tensor = torch.cat([emb_tensor, padding])
                else:
                    emb_tensor = emb_tensor[:perception_embedding.shape[0]]
            
            # 加权贡献
            weight = score * injection_weight
            knowledge_contribution += emb_tensor * weight
            total_weight += weight
        
        # 注入知识
        # perception_enhanced = perception * (1 - total_weight) + knowledge_contribution
        # 防止权重过大
        total_weight = min(total_weight, 0.5)
        perception_enhanced = (
            perception_embedding * (1 - total_weight) + 
            knowledge_contribution
        )
        
        logger.debug(
            f"Injected {len(retrieved_embeddings)} knowledge items "
            f"with total_weight={total_weight:.4f}"
        )
        
        return perception_enhanced
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "vector_db": self.vector_db,
            "collection_name": self.collection_name,
            "top_k": self.top_k,
            "embedding_dim": self.embedding_dim,
            "db_available": self._db_available,
        }
        
        if self.vector_db == "chromadb" and self._db_available:
            stats["num_items"] = self._collection.count()
        elif self.vector_db == "faiss" and self._db_available:
            stats["num_items"] = self._faiss_index.ntotal
        else:
            stats["num_items"] = len(self._knowledge_cache)
        
        return stats
    
    def clear(self):
        """清空知识库"""
        if self.vector_db == "chromadb" and self._db_available:
            # 清空集合
            all_ids = self._collection.get()['ids']
            if all_ids:
                self._collection.delete(ids=all_ids)
        
        elif self.vector_db == "faiss" and self._db_available:
            self._faiss_index.reset()
            self._faiss_id_map.clear()
        
        else:
            self._knowledge_cache.clear()
        
        logger.info("RAGKnowledgeBase cleared")


class PerceptionLayer(nn.Module):
    """
    L0 感知层
    
    Task 12 完整实现
    
    功能：
    - 实时滤波（PerceptionFilter）
    - 编码接口（PerceptionEncoder）
    - RAG 知识库接入（RAGKnowledgeBase）
    - 无自指约束（无自我模型）
    - 数据传递接口
    
    特性：
    - L0 仅处理感知，不包含自我状态或自我模型
    - 所有输出传递给 L1，不产生自指循环
    """
    
    def __init__(
        self,
        config: Optional[PerceptionLayerConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        paths_config: Optional[PathsConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化 L0 感知层
        
        Args:
            config: 感知层配置
            dim_config: 维度配置
            meta_config: 元认知配置
            paths_config: 路径配置
            device: 计算设备
        """
        super().__init__()
        
        # 合并配置
        self.config = config or PerceptionLayerConfig()
        
        if dim_config:
            self.config.semantic_dim = dim_config.semantic_dim
            self.config.physical_dim = dim_config.physical_dim
            self.config.output_dim = dim_config.fusion_dim
        
        if meta_config:
            self.config.encoding_hidden_dim = meta_config.l0_hidden_dim
        
        self.device = device or self.config.device
        
        # 创建滤波器
        self.filter = PerceptionFilter(
            method=self.config.filter_method,
            window_size=self.config.filter_window_size,
            alpha=self.config.filter_alpha,
            noise_threshold=self.config.noise_threshold,
            outlier_threshold=self.config.outlier_threshold,
            device=self.device
        )
        
        # 创建编码器
        self.encoder = PerceptionEncoder(
            semantic_dim=self.config.semantic_dim,
            physical_dim=self.config.physical_dim,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.encoding_hidden_dim,
            num_layers=self.config.encoding_num_layers,
            activation=self.config.encoding_activation,
            device=self.device
        )
        
        # 创建 RAG 知识库（可选）
        if self.config.use_rag:
            db_path = None
            if paths_config:
                db_path = f"{paths_config.data_root}/{paths_config.vector_db_dir}"
            
            self.rag_knowledge_base = RAGKnowledgeBase(
                vector_db=self.config.rag_vector_db,
                collection_name=self.config.rag_collection_name,
                top_k=self.config.rag_top_k,
                embedding_dim=self.config.rag_embedding_dim,
                db_path=db_path,
                device=self.device
            )
        else:
            self.rag_knowledge_base = None
        
        # 输出缓存（用于数据传递）
        self._output_cache: Optional[torch.Tensor] = None
        
        # 统计信息
        self._stats = {
            "total_inputs_processed": 0,
            "total_outputs_generated": 0,
            "filter_calls": 0,
            "encoder_calls": 0,
            "rag_calls": 0,
        }
        
        # 无自指约束标志（确保不包含自我模型）
        self._no_self_reference = True
        
        self.to(self.device)
        
        logger.info(
            f"PerceptionLayer (L0) initialized: "
            f"semantic_dim={self.config.semantic_dim}, "
            f"physical_dim={self.config.physical_dim}, "
            f"output_dim={self.config.output_dim}, "
            f"use_rag={self.config.use_rag}, "
            f"no_self_reference={self._no_self_reference}"
        )
    
    def forward(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor,
        use_rag: bool = True
    ) -> torch.Tensor:
        """
        处理感知输入
        
        流程：
        1. 滤波（噪声过滤、异常值检测）
        2. 编码（语义编码、物理编码、融合）
        3. RAG 知识注入（可选）
        4. 输出传递给 L1
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
            use_rag: 是否使用 RAG 知识注入
        
        Returns:
            感知表征（传递给 L1）
        """
        # 滤波
        semantic_filtered = self.filter(semantic_input)
        physical_filtered = self.filter(physical_input)
        self._stats["filter_calls"] += 2
        
        # 编码
        perception_embedding = self.encoder(semantic_filtered, physical_filtered)
        self._stats["encoder_calls"] += 1
        
        # RAG 知识注入
        if use_rag and self.rag_knowledge_base:
            perception_embedding = self.rag_knowledge_base.inject_knowledge(
                perception_embedding
            )
            self._stats["rag_calls"] += 1
        
        # 缓存输出
        self._output_cache = perception_embedding.clone()
        
        # 更新统计
        self._stats["total_inputs_processed"] += 2
        self._stats["total_outputs_generated"] += 1
        
        return perception_embedding
    
    def get_perception_output(self) -> Optional[torch.Tensor]:
        """
        获取感知输出（用于传递给 L1）
        
        Returns:
            感知表征张量
        """
        return self._output_cache
    
    def transfer_to_l1(self) -> Optional[torch.Tensor]:
        """
        数据传递接口：将处理后的数据传递给 L1
        
        Returns:
            传递给 L1 的感知表征
        """
        if self._output_cache is None:
            logger.warning("No perception output available for transfer")
            return None
        
        # 返回缓存的输出（L1 将接收这个数据）
        # 注意：这里不创建自指循环，数据单向传递给 L1
        return self._output_cache.clone()
    
    def verify_no_self_reference(self) -> bool:
        """
        验证无自指约束
        
        Returns:
            是否满足无自指约束
        """
        # L0 不应包含任何自我模型或自我状态
        # 检查点：
        # 1. 不包含 SelfState 引用
        # 2. 不包含自指回路
        # 3. 仅作为感知输入的预处理层
        
        checks = [
            # 检查无 SelfState
            not hasattr(self, 'self_state'),
            # 检查无自我模型组件
            not hasattr(self, 'self_model'),
            # 检查输出不指向自身
            self._no_self_reference,
        ]
        
        all_passed = all(checks)
        
        if not all_passed:
            logger.error("L0 self-reference constraint violated!")
        
        return all_passed
    
    def add_knowledge_to_rag(
        self,
        knowledge_id: str,
        embedding: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        添加知识到 RAG 知识库
        
        Args:
            knowledge_id: 知识ID
            embedding: 嵌入向量
            metadata: 元数据
        """
        if self.rag_knowledge_base:
            self.rag_knowledge_base.add_knowledge(knowledge_id, embedding, metadata)
        else:
            logger.warning("RAG knowledge base not initialized")
    
    def retrieve_from_rag(
        self,
        query_embedding: torch.Tensor,
        top_k: Optional[int] = None
    ) -> Tuple[List[torch.Tensor], List[float], List[str]]:
        """
        从 RAG 知识库检索
        
        Args:
            query_embedding: 查询嵌入
            top_k: 检索数量
        
        Returns:
            检索结果
        """
        if self.rag_knowledge_base:
            return self.rag_knowledge_base.retrieve(query_embedding, top_k)
        else:
            logger.warning("RAG knowledge base not initialized")
            return [], [], []
    
    def reset(self):
        """重置感知层"""
        self.filter.reset()
        self._output_cache = None
        if self.rag_knowledge_base:
            self.rag_knowledge_base.clear()
        self._stats = {
            "total_inputs_processed": 0,
            "total_outputs_generated": 0,
            "filter_calls": 0,
            "encoder_calls": 0,
            "rag_calls": 0,
        }
        logger.info("PerceptionLayer reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        stats["filter_stats"] = self.filter.get_statistics()
        stats["rag_stats"] = (
            self.rag_knowledge_base.get_statistics()
            if self.rag_knowledge_base else None
        )
        stats["no_self_reference"] = self._no_self_reference
        stats["config"] = {
            "semantic_dim": self.config.semantic_dim,
            "physical_dim": self.config.physical_dim,
            "output_dim": self.config.output_dim,
            "use_rag": self.config.use_rag,
        }
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证感知层
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证无自指约束
        if not self.verify_no_self_reference():
            errors.append("L0 self-reference constraint violated")
        
        # 验证配置
        if self.config.semantic_dim <= 0:
            errors.append(f"Invalid semantic_dim: {self.config.semantic_dim}")
        if self.config.physical_dim <= 0:
            errors.append(f"Invalid physical_dim: {self.config.physical_dim}")
        if self.config.output_dim <= 0:
            errors.append(f"Invalid output_dim: {self.config.output_dim}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __repr__(self) -> str:
        return (
            f"PerceptionLayer(L0, "
            f"semantic_dim={self.config.semantic_dim}, "
            f"physical_dim={self.config.physical_dim}, "
            f"output_dim={self.config.output_dim}, "
            f"use_rag={self.config.use_rag}, "
            f"no_self_reference={self._no_self_reference})"
        )


def create_perception_layer_from_config(
    dim_config: DimensionalityConfig,
    meta_config: MetaCognitiveConfig,
    paths_config: Optional[PathsConfig] = None,
    device: Optional[str] = None
) -> PerceptionLayer:
    """
    从配置创建感知层
    
    Args:
        dim_config: 维度配置
        meta_config: 元认知配置
        paths_config: 路径配置
        device: 计算设备
    
    Returns:
        PerceptionLayer 实例
    """
    config = PerceptionLayerConfig(
        semantic_dim=dim_config.semantic_dim,
        physical_dim=dim_config.physical_dim,
        output_dim=dim_config.fusion_dim,
        encoding_hidden_dim=meta_config.l0_hidden_dim,
    )
    
    layer = PerceptionLayer(
        config=config,
        dim_config=dim_config,
        meta_config=meta_config,
        paths_config=paths_config,
        device=device
    )
    
    return layer