"""
睡眠重放系统（Sleep Replay）
=============================

实现 Chronos-Self 的睡眠重放机制，每交互满24模拟小时强制进入离线微调期，
使用关键帧向量数据库存储和重放重要时刻。

Task 19 实现：
- SubTask 19.1: 24小时触发机制（强制进入睡眠期）
- SubTask 19.2: 关键帧向量数据库存储
- SubTask 19.3: 重放一致性损失计算
- SubTask 19.4: 预测改善损失计算

核心功能：
1. 24小时触发机制：
   - 每交互满24模拟小时，强制进入5分钟离线微调期
   - 支持手动触发睡眠期
   - 记录睡眠历史

2. 关键帧向量数据库：
   - 使用向量数据库（ChromaDB）存储关键帧
   - 关键帧定义：高情感强度时刻、重要决策点
   - 存储内容：自我状态快照、外部输入、系统响应
   - 支持关键帧检索和查询

3. 重放一致性损失：
   - 损失公式：L_replay_consistency = ||ODE_Solve(E_keyframe, ΔT) - E_keyframe_recorded||²
   - 使用关键帧时实际记录的状态作为目标
   - 确保重放不扭曲历史记忆

4. 预测改善损失：
   - 损失公式：L_replay_improve = ||Predict(E_keyframe) - ActualOutcome||²
   - 在关键帧状态下重新预测后续事件
   - 与实际结果对比，改善预测能力
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# ChromaDB 导入
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning(
        "chromadb not available. Keyframe storage will use in-memory fallback. "
        "Install with: pip install chromadb"
    )

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    MemoryTemporalConfig,
    TrainingConfig,
    PathsConfig,
)
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.history import HistoryEntry, EventType


logger = logging.getLogger(__name__)


@dataclass
class SleepReplayConfig:
    """睡眠重放配置"""
    
    # 触发机制
    sleep_replay_interval_hours: float = 24.0  # 睡眠触发间隔（小时）
    sleep_duration_minutes: float = 5.0  # 睡眠持续时间（分钟）
    auto_trigger_enabled: bool = True  # 启用自动触发
    
    # 关键帧存储
    keyframe_emotional_threshold: float = 0.7  # 情感强度阈值
    keyframe_importance_threshold: float = 0.8  # 重要性阈值
    keyframe_interval_minutes: float = 30.0  # 自动关键帧间隔
    max_keyframes_per_sleep: int = 100  # 每次睡眠最多重放的关键帧数
    
    # 向量数据库配置
    vector_db_path: str = "data/vector_db/keyframes"  # ChromaDB 存储路径
    vector_db_collection_name: str = "chronos_keyframes"  # 集合名称
    embedding_dim: int = 512  # 嵌入维度
    
    # 重放损失参数
    consistency_loss_weight: float = 1.0  # 一致性损失权重
    improve_loss_weight: float = 0.5  # 预测改善损失权重
    replay_window_seconds: float = 60.0  # 重放窗口（秒）
    
    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


@dataclass
class KeyframeData:
    """
    关键帧数据结构
    
    包含完整的自我状态快照、外部输入和系统响应。
    """
    
    # 基本信息
    keyframe_id: str
    timestamp: float
    created_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 自我状态快照
    E_fast: Optional[np.ndarray] = None
    E_slow: Optional[np.ndarray] = None
    
    # 外部输入快照
    X_sem: Optional[np.ndarray] = None
    X_log: Optional[np.ndarray] = None
    X_proprio: Optional[np.ndarray] = None
    X_world: Optional[np.ndarray] = None
    
    # 系统响应
    response_data: Optional[Dict[str, Any]] = None
    
    # 关键帧标记信息
    emotional_intensity: float = 0.0
    importance: float = 0.0
    is_high_emotional: bool = False
    is_decision_point: bool = False
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 后续状态（用于预测改善损失）后续状态（用于预测改善损失）
    actual_outcome: Optional[np.ndarray] = None  # 实际后续状态
    outcome_timestamp: Optional[float] = None  # 后续状态时间戳
    
    def to_embedding(self) -> np.ndarray:
        """
        将关键帧转换为嵌入向量
        
        用于向量数据库存储和检索。
        
        Returns:
            嵌入向量
        """
        # 使用快变量状态作为主要嵌入
        if self.E_fast is not None:
            # 简化：使用部分维度作为嵌入
            embedding = self.E_fast[:512].copy()
        else:
            embedding = np.zeros(512)
        
        # 添加情感强度和重要性信息
        embedding[-2] = self.emotional_intensity
        embedding[-1] = self.importance
        
        return embedding
    
    def to_dict(self) -> Dict[str, Any]:
        """
        序列化为字典
        
        Returns:
            字典形式的关键帧数据
        """
        return {
            "keyframe_id": self.keyframe_id,
            "timestamp": self.timestamp,
            "created_time": self.created_time,
            "E_fast": self.E_fast.tolist() if self.E_fast is not None else None,
            "E_slow": self.E_slow.tolist() if self.E_slow is not None else None,
            "X_sem": self.X_sem.tolist() if self.X_sem is not None else None,
            "X_log": self.X_log.tolist() if self.X_log is not None else None,
            "emotional_intensity": self.emotional_intensity,
            "importance": self.importance,
            "is_high_emotional": self.is_high_emotional,
            "is_decision_point": self.is_decision_point,
            "metadata": self.metadata,
            "response_data": self.response_data,
            "actual_outcome": self.actual_outcome.tolist() if self.actual_outcome is not None else None,
            "outcome_timestamp": self.outcome_timestamp,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeyframeData":
        """
        从字典反序列化
        
        Args:
            data: 字典数据
            
        Returns:
            KeyframeData 实例
        """
        return cls(
            keyframe_id=data["keyframe_id"],
            timestamp=data["timestamp"],
            created_time=data.get("created_time", ""),
            E_fast=np.array(data["E_fast"]) if data.get("E_fast") is not None else None,
            E_slow=np.array(data["E_slow"]) if data.get("E_slow") is not None else None,
            X_sem=np.array(data["X_sem"]) if data.get("X_sem") is not None else None,
            X_log=np.array(data["X_log"]) if data.get("X_log") is not None else None,
            emotional_intensity=data.get("emotional_intensity", 0.0),
            importance=data.get("importance", 0.0),
            is_high_emotional=data.get("is_high_emotional", False),
            is_decision_point=data.get("is_decision_point", False),
            metadata=data.get("metadata", {}),
            response_data=data.get("response_data"),
            actual_outcome=np.array(data["actual_outcome"]) if data.get("actual_outcome") is not None else None,
            outcome_timestamp=data.get("outcome_timestamp"),
        )
    
    def to_state(self) -> SelfState:
        """
        转换为 SelfState 对象
        
        Returns:
            SelfState 实例
        """
        E_fast_tensor = torch.tensor(self.E_fast, dtype=torch.float32) if self.E_fast is not None else torch.zeros(2048)
        E_slow_tensor = torch.tensor(self.E_slow, dtype=torch.float32) if self.E_slow is not None else torch.zeros(512)
        
        return SelfState(
            E_fast=E_fast_tensor,
            E_slow=E_slow_tensor,
            timestamp=self.timestamp,
            metadata=self.metadata,
        )
    
    def __repr__(self) -> str:
        return (
            f"KeyframeData(id={self.keyframe_id}, "
            f"timestamp={self.timestamp:.2f}, "
            f"emotional={self.emotional_intensity:.2f}, "
            f"importance={self.importance:.2f})"
        )


class KeyframeDatabase:
    """
    关键帧向量数据库
    
    使用 ChromaDB 存储关键帧，支持向量检索和查询。
    
    Task 19.2 实现：关键帧向量数据库存储
    
    功能：
    - 关键帧存储和索引
    - 向量相似度检索
    - 基于时间范围查询
    - 基于情感强度查询
    - 持久化存储
    """
    
    def __init__(
        self,
        config: Optional[SleepReplayConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化关键帧数据库
        
        Args:
            config: 睡眠重放配置
            global_config: 全局配置
            device: 计算设备
        """
        self.config = config or SleepReplayConfig()
        self.global_config = global_config or ChronosConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从全局配置更新
        if global_config:
            self.config.vector_db_path = global_config.paths.vector_db_dir
            self.config.sleep_replay_interval_hours = global_config.memory_temporal.sleep_replay_interval_hours
            self.config.sleep_duration_minutes = global_config.memory_temporal.sleep_duration_minutes
        
        # ChromaDB 客户端和集合
        self.client: Optional[chromadb.Client] = None
        self.collection: Optional[chromadb.Collection] = None
        
        # 内存缓存（当 ChromaDB 不可用时使用）
        self._memory_cache: Dict[str, KeyframeData] = {}
        self._memory_embeddings: Dict[str, np.ndarray] = {}
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_keyframes": 0,
            "keyframes_added": 0,
            "keyframes_retrieved": 0,
            "queries_performed": 0,
            "chromadb_available": CHROMADB_AVAILABLE,
        }
        
        # 初始化数据库
        self._initialize_database()
        
        logger.info(
            f"KeyframeDatabase initialized: "
            f"chromadb={CHROMADB_AVAILABLE}, "
            f"path={self.config.vector_db_path}"
        )
    
    def _initialize_database(self) -> None:
        """
        初始化向量数据库
        """
        if CHROMADB_AVAILABLE:
            try:
                # 创建持久化客户端
                db_path = Path(self.config.vector_db_path)
                db_path.mkdir(parents=True, exist_ok=True)
                
                self.client = chromadb.PersistentClient(
                    path=str(db_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                    ),
                )
                
                # 创建或获取集合
                self.collection = self.client.get_or_create_collection(
                    name=self.config.vector_db_collection_name,
                    metadata={"hnsw_space": "l2"},  # 使用 L2 距离
                )
                
                logger.info(f"ChromaDB collection '{self.config.vector_db_collection_name}' initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB: {e}")
                self.client = None
                self.collection = None
        else:
            # 使用内存缓存
            logger.info("Using in-memory keyframe storage")
    
    def add_keyframe(
        self,
        keyframe: KeyframeData,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加关键帧到数据库
        
        Args:
            keyframe: 关键帧数据
            metadata: 额外元数据
            
        Returns:
            关键帧 ID
        """
        # 确保 keyframe_id 存在
        if not keyframe.keyframe_id:
            keyframe.keyframe_id = f"kf_{keyframe.timestamp:.2f}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 获取嵌入向量
        embedding = keyframe.to_embedding()
        
        # 准备元数据
        full_metadata = {
            "timestamp": keyframe.timestamp,
            "emotional_intensity": keyframe.emotional_intensity,
            "importance": keyframe.importance,
            "is_high_emotional": keyframe.is_high_emotional,
            "is_decision_point": keyframe.is_decision_point,
            "created_time": keyframe.created_time,
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # 存储
        if self.collection is not None:
            try:
                self.collection.add(
                    ids=[keyframe.keyframe_id],
                    embeddings=[embedding.tolist()],
                    metadatas=[full_metadata],
                    documents=[json.dumps(keyframe.to_dict())],
                )
                
                logger.debug(f"Keyframe added to ChromaDB: {keyframe.keyframe_id}")
                
            except Exception as e:
                logger.error(f"Failed to add keyframe to ChromaDB: {e}")
                # 回退到内存存储
                self._memory_cache[keyframe.keyframe_id] = keyframe
                self._memory_embeddings[keyframe.keyframe_id] = embedding
        
        else:
            # 内存存储
            self._memory_cache[keyframe.keyframe_id] = keyframe
            self._memory_embeddings[keyframe.keyframe_id] = embedding
        
        # 更新统计
        self._stats["total_keyframes"] += 1
        self._stats["keyframes_added"] += 1
        
        return keyframe.keyframe_id
    
    def retrieve_keyframe(self, keyframe_id: str) -> Optional[KeyframeData]:
        """
        检索指定 ID 的关键帧
        
        Args:
            keyframe_id: 关键帧 ID
            
        Returns:
            关键帧数据（如果存在）
        """
        # ChromaDB 查询
        if self.collection is not None:
            try:
                result = self.collection.get(
                    ids=[keyframe_id],
                    include=["documents", "metadatas"],
                )
                
                if result["documents"]:
                    data_dict = json.loads(result["documents"][0])
                    keyframe = KeyframeData.from_dict(data_dict)
                    
                    self._stats["keyframes_retrieved"] += 1
                    return keyframe
                    
            except Exception as e:
                logger.error(f"Failed to retrieve keyframe from ChromaDB: {e}")
        
        # 内存查询
        if keyframe_id in self._memory_cache:
            self._stats["keyframes_retrieved"] += 1
            return self._memory_cache[keyframe_id]
        
        return None
    
    def query_by_similarity(
        self,
        query_embedding: np.ndarray,
        n_results: int = 10
    ) -> List[KeyframeData]:
        """
        基于向量相似度查询关键帧
        
        Args:
            query_embedding: 查询嵌入向量
            n_results: 返回结果数量
            
        Returns:
            相似关键帧列表
        """
        keyframes = []
        
        if self.collection is not None:
            try:
                result = self.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"],
                )
                
                if result["documents"]:
                    for doc in result["documents"][0]:
                        data_dict = json.loads(doc)
                        keyframe = KeyframeData.from_dict(data_dict)
                        keyframes.append(keyframe)
                
                self._stats["queries_performed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to query ChromaDB: {e}")
        
        else:
            # 内存查询（简单的 L2 距离）
            distances = []
            for kf_id, embedding in self._memory_embeddings.items():
                dist = np.linalg.norm(query_embedding - embedding)
                distances.append((kf_id, dist))
            
            # 排序并取前 N 个
            distances.sort(key=lambda x: x[1])
            top_ids = [d[0] for d in distances[:n_results]]
            
            for kf_id in top_ids:
                if kf_id in self._memory_cache:
                    keyframes.append(self._memory_cache[kf_id])
            
            self._stats["queries_performed"] += 1
        
        return keyframes
    
    def query_by_time_range(
        self,
        start_time: float,
        end_time: float
    ) -> List[KeyframeData]:
        """
        基于时间范围查询关键帧
        
        Args:
            start_time: 开始时间戳
            end_time: 结束时间戳
            
        Returns:
            时间范围内的关键帧列表
        """
        keyframes = []
        
        if self.collection is not None:
            try:
                # ChromaDB 元数据过滤
                result = self.collection.get(
                    where={
                        "timestamp": {
                            "$gte": start_time,
                            "$lte": end_time,
                        }
                    },
                    include=["documents", "metadatas"],
                )
                
                if result["documents"]:
                    for doc in result["documents"]:
                        data_dict = json.loads(doc)
                        keyframe = KeyframeData.from_dict(data_dict)
                        keyframes.append(keyframe)
                
                self._stats["queries_performed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to query ChromaDB by time: {e}")
        
        else:
            # 内存查询
            for keyframe in self._memory_cache.values():
                if start_time <= keyframe.timestamp <= end_time:
                    keyframes.append(keyframe)
            
            self._stats["queries_performed"] += 1
        
        return keyframes
    
    def query_high_emotional(
        self,
        threshold: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[KeyframeData]:
        """
        查询高情感强度的关键帧
        
        Args:
            threshold: 情感强度阈值
            limit: 返回数量限制
            
        Returns:
            高情感强度关键帧列表
        """
        threshold = threshold or self.config.keyframe_emotional_threshold
        keyframes = []
        
        if self.collection is not None:
            try:
                result = self.collection.get(
                    where={
                        "emotional_intensity": {"$gte": threshold}
                    },
                    include=["documents", "metadatas"],
                    limit=limit or self.config.max_keyframes_per_sleep,
                )
                
                if result["documents"]:
                    for doc in result["documents"]:
                        data_dict = json.loads(doc)
                        keyframe = KeyframeData.from_dict(data_dict)
                        keyframes.append(keyframe)
                
                self._stats["queries_performed"] += 1
                
            except Exception as e:
                logger.error(f"Failed to query high emotional keyframes: {e}")
        
        else:
            # 内存查询
            for keyframe in self._memory_cache.values():
                if keyframe.emotional_intensity >= threshold:
                    keyframes.append(keyframe)
            
            if limit:
                keyframes = keyframes[:limit]
            
            self._stats["queries_performed"] += 1
        
        return keyframes
    
    def get_recent_keyframes(
        self,
        n: int = 100,
        time_window_seconds: Optional[float] = None
    ) -> List[KeyframeData]:
        """
        获取最近的关键帧
        
        Args:
            n: 数量限制
            time_window_seconds: 时间窗口（秒）
            
        Returns:
            最近关键帧列表
        """
        if time_window_seconds:
            # 获取当前时间
            current_time = time.time()
            start_time = current_time - time_window_seconds
            
            return self.query_by_time_range(start_time, current_time)[:n]
        
        # 否则按时间排序获取最近 N 个
        keyframes = []
        
        if self.collection is not None:
            try:
                # 获取所有关键帧（按时间降序）
                all_ids = self.collection.get()["ids"]
                
                # 这里的简化处理：直接获取前 N 个
                # 实际应该根据时间戳排序
                result = self.collection.get(
                    ids=all_ids[:n],
                    include=["documents", "metadatas"],
                )
                
                if result["documents"]:
                    for doc in result["documents"]:
                        data_dict = json.loads(doc)
                        keyframe = KeyframeData.from_dict(data_dict)
                        keyframes.append(keyframe)
                
                # 按时间排序
                keyframes.sort(key=lambda kf: kf.timestamp, reverse=True)
                
            except Exception as e:
                logger.error(f"Failed to get recent keyframes: {e}")
        
        else:
            # 内存查询
            all_keyframes = list(self._memory_cache.values())
            all_keyframes.sort(key=lambda kf: kf.timestamp, reverse=True)
            keyframes = all_keyframes[:n]
        
        return keyframes
    
    def update_outcome(
        self,
        keyframe_id: str,
        actual_outcome: np.ndarray,
        outcome_timestamp: float
    ) -> bool:
        """
        更新关键帧的实际后续状态
        
        Args:
            keyframe_id: 关键帧 ID
            actual_outcome: 实际后续状态
            outcome_timestamp: 后续状态时间戳
            
        Returns:
            更新是否成功
        """
        keyframe = self.retrieve_keyframe(keyframe_id)
        
        if keyframe is None:
            logger.warning(f"Keyframe {keyframe_id} not found")
            return False
        
        # 更新后续状态
        keyframe.actual_outcome = actual_outcome
        keyframe.outcome_timestamp = outcome_timestamp
        
        # 重新存储（ChromaDB 需要更新）
        if self.collection is not None:
            try:
                self.collection.update(
                    ids=[keyframe_id],
                    documents=[json.dumps(keyframe.to_dict())],
                )
                
                logger.debug(f"Keyframe outcome updated: {keyframe_id}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to update keyframe: {e}")
                return False
        
        else:
            # 内存更新
            self._memory_cache[keyframe_id] = keyframe
            return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据库统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        
        # ChromaDB 统计
        if self.collection is not None:
            try:
                stats["chromadb_count"] = self.collection.count()
            except:
                stats["chromadb_count"] = 0
        
        # 内存统计
        stats["memory_cache_size"] = len(self._memory_cache)
        
        return stats
    
    def clear(self) -> None:
        """
        清空数据库
        """
        if self.collection is not None:
            try:
                # 删除并重新创建集合
                self.client.delete_collection(self.config.vector_db_collection_name)
                self.collection = self.client.create_collection(
                    name=self.config.vector_db_collection_name,
                    metadata={"hnsw_space": "l2"},
                )
                
                logger.info("ChromaDB collection cleared")
                
            except Exception as e:
                logger.error(f"Failed to clear ChromaDB: {e}")
        
        # 清空内存缓存
        self._memory_cache.clear()
        self._memory_embeddings.clear()
        
        # 重置统计
        self._stats["total_keyframes"] = 0
        
        logger.info("KeyframeDatabase cleared")
    
    def __len__(self) -> int:
        """返回关键帧数量"""
        if self.collection is not None:
            return self.collection.count()
        
        return len(self._memory_cache)
    
    def __repr__(self) -> str:
        storage = "chromadb" if self.collection is not None else "memory"
        return (
            f"KeyframeDatabase(storage={storage}, "
            f"count={len(self)}, "
            f"path={self.config.vector_db_path})"
        )


class ReplayLossCalculator:
    """
    重放损失计算器
    
    计算重放一致性损失和预测改善损失。
    
    Task 19.3 和 19.4 实现：
    - SubTask 19.3: 重放一致性损失计算
    - SubTask 19.4: 预测改善损失计算
    
    损失公式：
    - 一致性损失：L_replay_consistency = ||ODE_Solve(E_keyframe, ΔT) - E_keyframe_recorded||²
    - 预测改善损失：L_replay_improve = ||Predict(E_keyframe) - ActualOutcome||²
    """
    
    def __init__(
        self,
        config: Optional[SleepReplayConfig] = None,
        integration_engine: Optional[IntegrationEngine] = None,
        device: Optional[str] = None
    ):
        """
        初始化重放损失计算器
        
        Args:
            config: 睡眠重放配置
            integration_engine: 积分引擎
            device: 计算设备
        """
        self.config = config or SleepReplayConfig()
        self.integration_engine = integration_engine
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 损失权重
        self.consistency_weight = self.config.consistency_loss_weight
        self.improve_weight = self.config.improve_loss_weight
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_loss_calculations": 0,
            "avg_consistency_loss": 0.0,
            "avg_improve_loss": 0.0,
            "avg_total_loss": 0.0,
        }
        
        logger.info(
            f"ReplayLossCalculator initialized: "
            f"consistency_weight={self.consistency_weight}, "
            f"improve_weight={self.improve_weight}"
        )
    
    def compute_losses(
        self,
        keyframes: List[KeyframeData],
        replay_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        计算重放损失
        
        Args:
            keyframes: 关键帧列表
            replay_window: 重放窗口（秒）
            
        Returns:
            损失结果字典
        """
        replay_window = replay_window or self.config.replay_window_seconds
        
        consistency_losses = []
        improve_losses = []
        
        for keyframe in keyframes:
            # 计算一致性损失
            consistency_loss = self._compute_consistency_loss(keyframe, replay_window)
            if consistency_loss is not None:
                consistency_losses.append(consistency_loss)
            
            # 计算预测改善损失
            improve_loss = self._compute_improve_loss(keyframe)
            if improve_loss is not None:
                improve_losses.append(improve_loss)
        
        # 计算平均损失
        avg_consistency = np.mean(consistency_losses) if consistency_losses else 0.0
        avg_improve = np.mean(improve_losses) if improve_losses else 0.0
        
        # 总损失
        total_loss = (
            self.consistency_weight * avg_consistency +
            self.improve_weight * avg_improve
        )
        
        # 更新统计
        self._stats["total_loss_calculations"] += 1
        self._stats["avg_consistency_loss"] = avg_consistency
        self._stats["avg_improve_loss"] = avg_improve
        self._stats["avg_total_loss"] = total_loss
        
        logger.debug(
            f"Replay losses computed: "
            f"consistency={avg_consistency:.4f}, "
            f"improve={avg_improve:.4f}, "
            f"total={total_loss:.4f}"
        )
        
        return {
            "consistency_loss": avg_consistency,
            "improve_loss": avg_improve,
            "total_loss": total_loss,
            "keyframes_count": len(keyframes),
            "consistency_losses_count": len(consistency_losses),
            "improve_losses_count": len(improve_losses),
        }
    
    def _compute_consistency_loss(
        self,
        keyframe: KeyframeData,
        replay_window: float
    ) -> Optional[float]:
        """
        计算重放一致性损失
        
        损失公式：L_replay_consistency = ||ODE_Solve(E_keyframe, ΔT) - E_keyframe_recorded||²
        
        Args:
            keyframe: 关键帧数据
            replay_window: 重放窗口
            
        Returns:
            一致性损失值
        """
        if keyframe.E_fast is None:
            return None
        
        if self.integration_engine is None:
            # 简化：使用状态范数差异作为损失
            return float(np.linalg.norm(keyframe.E_fast))
        
        # 获取关键帧状态
        initial_state = keyframe.to_state()
        
        # 从关键帧向前积分
        # 简化：仅进行单步积分
        try:
            # 转换为张量
            E_fast_tensor = torch.tensor(keyframe.E_fast, dtype=torch.float32, device=self.device)
            E_slow_tensor = torch.tensor(keyframe.E_slow, dtype=torch.float32, device=self.device) if keyframe.E_slow is not None else torch.zeros(self.config.slow_dim, device=self.device)
            
            # 执行单步积分
            dt = replay_window
            inputs = {}
            
            if keyframe.X_sem is not None:
                inputs['X_sem'] = torch.tensor(keyframe.X_sem, dtype=torch.float32, device=self.device)
            if keyframe.X_log is not None:
                inputs['X_log'] = torch.tensor(keyframe.X_log, dtype=torch.float32, device=self.device)
            
            E_fast_replayed = self.integration_engine.fast_dynamics.step(
                E_fast_tensor,
                E_slow_tensor,
                inputs,
                dt,
                keyframe.timestamp,
            )
            
            # 与记录的状态对比
            # 使用范数差异作为损失
            if keyframe.actual_outcome is not None:
                target = torch.tensor(keyframe.actual_outcome, dtype=torch.float32, device=self.device)
                loss = torch.norm(E_fast_replayed - target).item()
            else:
                # 使用初始状态的范数作为损失（确保不偏离过多）
                loss = torch.norm(E_fast_replayed - E_fast_tensor).item()
            
            return loss
            
        except Exception as e:
            logger.error(f"Failed to compute consistency loss: {e}")
            return None
    
    def _compute_improve_loss(
        self,
        keyframe: KeyframeData
    ) -> Optional[float]:
        """
        计算预测改善损失
        
        损失公式：L_replay_improve = ||Predict(E_keyframe) - ActualOutcome||²
        
        Args:
            keyframe: 关键帧数据
            
        Returns:
            预测改善损失值
        """
        # 需要有实际后续状态
        if keyframe.actual_outcome is None or keyframe.E_fast is None:
            return None
        
        if self.integration_engine is None:
            # 简化：使用状态差异
            return float(np.linalg.norm(keyframe.E_fast - keyframe.actual_outcome))
        
        # 在关键帧状态下重新预测后续状态
        try:
            # 计算预测窗口
            if keyframe.outcome_timestamp is not None:
                prediction_window = keyframe.outcome_timestamp - keyframe.timestamp
            else:
                prediction_window = self.config.replay_window_seconds
            
            # 转换为张量
            E_fast_tensor = torch.tensor(keyframe.E_fast, dtype=torch.float32, device=self.device)
            E_slow_tensor = torch.tensor(keyframe.E_slow, dtype=torch.float32, device=self.device) if keyframe.E_slow is not None else torch.zeros(self.config.slow_dim, device=self.device)
            
            # 执行预测积分
            inputs = {}
            
            if keyframe.X_sem is not None:
                inputs['X_sem'] = torch.tensor(keyframe.X_sem, dtype=torch.float32, device=self.device)
            if keyframe.X_log is not None:
                inputs['X_log'] = torch.tensor(keyframe.X_log, dtype=torch.float32, device=self.device)
            
            E_fast_predicted = self.integration_engine.fast_dynamics.step(
                E_fast_tensor,
                E_slow_tensor,
                inputs,
                prediction_window,
                keyframe.timestamp,
            )
            
            # 与实际结果对比
            actual_outcome_tensor = torch.tensor(keyframe.actual_outcome, dtype=torch.float32, device=self.device)
            
            loss = torch.norm(E_fast_predicted - actual_outcome_tensor).item()
            
            return loss
            
        except Exception as e:
            logger.error(f"Failed to compute improve loss: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return self._stats.copy()
    
    def reset(self) -> None:
        """
        重置统计信息
        """
        self._stats = {
            "total_loss_calculations": 0,
            "avg_consistency_loss": 0.0,
            "avg_improve_loss": 0.0,
            "avg_total_loss": 0.0,
        }
        
        logger.info("ReplayLossCalculator reset")
    
    def __repr__(self) -> str:
        return (
            f"ReplayLossCalculator(calculations={self._stats['total_loss_calculations']}, "
            f"avg_loss={self._stats['avg_total_loss']:.4f})"
        )


class SleepReplay:
    """
    睡眠重放系统
    
    整合24小时触发机制、关键帧数据库和重放损失计算，
    实现完整的睡眠重放流程。
    
    Task 19 完整实现：
    - SubTask 19.1: 24小时触发机制
    - SubTask 19.2: 关键帧向量数据库存储
    - SubTask 19.3: 重放一致性损失计算
    - SubTask 19.4: 预测改善损失计算
    
    使用示例：
        sleep_replay = SleepReplay(config=SleepReplayConfig())
        sleep_replay.initialize(integration_engine)
        
        # 检查是否需要睡眠
        if sleep_replay.should_sleep(current_time):
            # 执行睡眠重放
            result = sleep_replay.perform_sleep()
        
        # 手动触发睡眠
        sleep_replay.trigger_sleep()
    """
    
    def __init__(
        self,
        config: Optional[SleepReplayConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        integration_engine: Optional[IntegrationEngine] = None,
        device: Optional[str] = None
    ):
        """
        初始化睡眠重放系统
        
        Args:
            config: 睡眠重放配置
            global_config: 全局配置
            integration_engine: 积分引擎
            device: 计算设备
        """
        self.config = config or SleepReplayConfig()
        self.global_config = global_config or ChronosConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从全局配置更新
        if global_config:
            self.config.sleep_replay_interval_hours = global_config.memory_temporal.sleep_replay_interval_hours
            self.config.sleep_duration_minutes = global_config.memory_temporal.sleep_duration_minutes
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim
        
        # 积分引擎
        self.integration_engine = integration_engine
        
        # 核心组件
        self.keyframe_db: Optional[KeyframeDatabase] = None
        self.loss_calculator: Optional[ReplayLossCalculator] = None
        
        # 睡眠状态
        self._last_sleep_time: float = time.time()  # 初始化为当前时间，避免立即触发
        self._sleep_count: int = 0
        self._is_sleeping: bool = False
        
        # 时间跟踪（模拟小时）
        self._accumulated_time_hours: float = 0.0
        
        # 睡眠历史记录
        self._sleep_history: List[Dict[str, Any]] = []
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "total_sleeps": 0,
            "total_sleep_duration_minutes": 0.0,
            "total_keyframes_replayed": 0,
            "avg_sleep_duration_minutes": 0.0,
            "avg_keyframes_per_sleep": 0.0,
        }
        
        # 初始化标志
        self._initialized = False
        
        logger.info(
            f"SleepReplay created: "
            f"interval={self.config.sleep_replay_interval_hours}h, "
            f"duration={self.config.sleep_duration_minutes}min"
        )
    
    def initialize(
        self,
        integration_engine: Optional[IntegrationEngine] = None
    ) -> None:
        """
        初始化睡眠重放系统
        
        Args:
            integration_engine: 积分引擎
        """
        if integration_engine is not None:
            self.integration_engine = integration_engine
        
        # 创建关键帧数据库
        self.keyframe_db = KeyframeDatabase(
            config=self.config,
            global_config=self.global_config,
            device=self.device,
        )
        
        # 创建损失计算器
        self.loss_calculator = ReplayLossCalculator(
            config=self.config,
            integration_engine=self.integration_engine,
            device=self.device,
        )
        
        self._initialized = True
        
        logger.info("SleepReplay initialized")
    
    def should_sleep(self, current_time: float) -> bool:
        """
        判断是否应该进入睡眠期
        
        Task 19.1: 24小时触发机制
        
        Args:
            current_time: 当前模拟时间（秒）
            
        Returns:
            是否应该睡眠
        """
        if not self._initialized or not self.config.auto_trigger_enabled:
            return False
        
        # 检查是否已经在睡眠中
        if self._is_sleeping:
            return False
        
        # 计算距离上次睡眠的时间（小时）
        hours_since_last_sleep = (current_time - self._last_sleep_time) / 3600
        
        # 检查是否达到触发间隔
        should_sleep = hours_since_last_sleep >= self.config.sleep_replay_interval_hours
        
        if should_sleep:
            logger.info(
                f"Sleep trigger condition met: "
                f"hours_since_last={hours_since_last_sleep:.2f}, "
                f"interval={self.config.sleep_replay_interval_hours}"
            )
        
        return should_sleep
    
    def trigger_sleep(
        self,
        force: bool = False,
        duration_minutes: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        手动触发睡眠期
        
        Args:
            force: 是否强制触发（忽略间隔检查）
            duration_minutes: 睡眠持续时间（如果为 None，使用配置值）
            
        Returns:
            睡眠结果字典
        """
        if not self._initialized:
            raise ValueError("SleepReplay not initialized.")
        
        duration_minutes = duration_minutes or self.config.sleep_duration_minutes
        
        logger.info(f"Manual sleep triggered: duration={duration_minutes}min")
        
        return self.perform_sleep(duration_minutes)
    
    def perform_sleep(
        self,
        duration_minutes: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        执行睡眠重放
        
        Args:
            duration_minutes: 睡眠持续时间
            
        Returns:
            睡眠结果字典
        """
        if not self._initialized:
            raise ValueError("SleepReplay not initialized.")
        
        duration_minutes = duration_minutes or self.config.sleep_duration_minutes
        
        # 标记开始睡眠
        self._is_sleeping = True
        sleep_start_time = time.time()
        
        logger.info(
            f"Sleep started: duration={duration_minutes}min, "
            f"sleep_count={self._sleep_count + 1}"
        )
        
        # 获取关键帧
        keyframes = self.keyframe_db.get_recent_keyframes(
            n=self.config.max_keyframes_per_sleep
        )
        
        if not keyframes:
            logger.warning("No keyframes available for replay")
            keyframes = []
        
        # 计算重放损失
        losses_result = self.loss_calculator.compute_losses(keyframes)
        
        # 睡眠持续时间（模拟）
        # 实际系统中这里可能需要等待或执行其他操作
        # 这里简化为立即完成
        sleep_end_time = time.time()
        actual_duration_seconds = sleep_end_time - sleep_start_time
        
        # 标记结束睡眠
        self._is_sleeping = False
        self._last_sleep_time = time.time()
        self._sleep_count += 1
        
        # 记录睡眠历史
        sleep_record = {
            "sleep_count": self._sleep_count,
            "start_time": sleep_start_time,
            "end_time": sleep_end_time,
            "duration_minutes": actual_duration_seconds / 60,
            "keyframes_replayed": len(keyframes),
            "losses": losses_result,
            "timestamp": datetime.now().isoformat(),
        }
        
        self._sleep_history.append(sleep_record)
        
        # 更新统计
        self._stats["total_sleeps"] += 1
        self._stats["total_sleep_duration_minutes"] += actual_duration_seconds / 60
        self._stats["total_keyframes_replayed"] += len(keyframes)
        self._stats["avg_sleep_duration_minutes"] = (
            self._stats["total_sleep_duration_minutes"] / self._stats["total_sleeps"]
        )
        self._stats["avg_keyframes_per_sleep"] = (
            self._stats["total_keyframes_replayed"] / self._stats["total_sleeps"]
        )
        
        # 构建结果
        result = {
            "success": True,
            "sleep_count": self._sleep_count,
            "duration_minutes": actual_duration_seconds / 60,
            "keyframes_replayed": len(keyframes),
            "losses": losses_result,
            "keyframe_db_stats": self.keyframe_db.get_statistics(),
            "loss_calculator_stats": self.loss_calculator.get_statistics(),
        }
        
        logger.info(
            f"Sleep completed: sleep_count={self._sleep_count}, "
            f"duration={actual_duration_seconds/60:.2f}min, "
            f"keyframes={len(keyframes)}, "
            f"total_loss={losses_result['total_loss']:.4f}"
        )
        
        return result
    
    def add_keyframe(
        self,
        state: SelfState,
        inputs: Optional[ExternalInput] = None,
        response_data: Optional[Dict[str, Any]] = None,
        emotional_intensity: Optional[float] = None,
        importance: Optional[float] = None,
        is_decision_point: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        添加关键帧
        
        Args:
            state: 当前状态
            inputs: 外部输入
            response_data: 系统响应
            emotional_intensity: 情感强度
            importance: 重要性
            is_decision_point: 是否为决策点
            metadata: 元数据
            
        Returns:
            关键帧 ID
        """
        if not self._initialized:
            raise ValueError("SleepReplay not initialized.")
        
        # 从外部输入获取情感强度和重要性
        if inputs is not None:
            emotional_intensity = emotional_intensity or inputs.emotional_intensity
            importance = importance or inputs.importance
        
        # 创建关键帧
        keyframe = KeyframeData(
            keyframe_id="",  # 自动生成
            timestamp=state.timestamp,
            E_fast=state.E_fast.numpy(),
            E_slow=state.E_slow.numpy(),
            emotional_intensity=emotional_intensity or 0.0,
            importance=importance or 0.0,
            is_high_emotional=(emotional_intensity or 0.0) >= self.config.keyframe_emotional_threshold,
            is_decision_point=is_decision_point,
            metadata=metadata or {},
            response_data=response_data,
        )
        
        # 添加输入信息
        if inputs is not None:
            keyframe.X_sem = inputs.X_sem.numpy() if inputs.X_sem is not None else None
            keyframe.X_log = inputs.X_log.numpy() if inputs.X_log is not None else None
        
        # 存储到数据库
        keyframe_id = self.keyframe_db.add_keyframe(keyframe)
        
        logger.debug(
            f"Keyframe added: id={keyframe_id}, "
            f"emotional={emotional_intensity:.2f}, "
            f"importance={importance:.2f}"
        )
        
        return keyframe_id
    
    def update_accumulated_time(self, dt_seconds: float) -> None:
        """
        更新累积时间
        
        Args:
            dt_seconds: 时间增量（秒）
        """
        self._accumulated_time_hours += dt_seconds / 3600
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["sleep_count"] = self._sleep_count
        stats["is_sleeping"] = self._is_sleeping
        stats["accumulated_time_hours"] = self._accumulated_time_hours
        stats["initialized"] = self._initialized
        
        if self.keyframe_db:
            stats["keyframe_db_stats"] = self.keyframe_db.get_statistics()
        
        if self.loss_calculator:
            stats["loss_calculator_stats"] = self.loss_calculator.get_statistics()
        
        return stats
    
    def get_sleep_history(self) -> List[Dict[str, Any]]:
        """
        获取睡眠历史
        
        Returns:
            睡眠历史记录列表
        """
        return self._sleep_history.copy()
    
    def reset(self) -> None:
        """
        重置睡眠重放系统
        """
        self._last_sleep_time = 0.0
        self._sleep_count = 0
        self._is_sleeping = False
        self._accumulated_time_hours = 0.0
        self._sleep_history.clear()
        
        if self.keyframe_db:
            self.keyframe_db.clear()
        
        if self.loss_calculator:
            self.loss_calculator.reset()
        
        self._stats = {
            "total_sleeps": 0,
            "total_sleep_duration_minutes": 0.0,
            "total_keyframes_replayed": 0,
            "avg_sleep_duration_minutes": 0.0,
            "avg_keyframes_per_sleep": 0.0,
        }
        
        logger.info("SleepReplay reset")
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        sleeping = "sleeping" if self._is_sleeping else "awake"
        return (
            f"SleepReplay(status={status}, sleeping={sleeping}, "
            f"sleeps={self._sleep_count}, "
            f"time={self._accumulated_time_hours:.2f}h)"
        )


def create_sleep_replay_from_config(
    config: ChronosConfig,
    integration_engine: Optional[IntegrationEngine] = None,
    device: Optional[str] = None
) -> SleepReplay:
    """
    从全局配置创建睡眠重放系统
    
    Args:
        config: 全局配置
        integration_engine: 积分引擎
        device: 计算设备
        
    Returns:
        SleepReplay 实例
    """
    sleep_config = SleepReplayConfig(
        sleep_replay_interval_hours=config.memory_temporal.sleep_replay_interval_hours,
        sleep_duration_minutes=config.memory_temporal.sleep_duration_minutes,
        vector_db_path=config.paths.vector_db_dir,
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
    )
    
    sleep_replay = SleepReplay(
        config=sleep_config,
        global_config=config,
        integration_engine=integration_engine,
        device=device,
    )
    
    sleep_replay.initialize()
    return sleep_replay