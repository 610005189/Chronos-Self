"""
探索驱动引擎 - Exploration Engine
==================================

实现基于预测误差的探索驱动输入选择机制，集成到递归状态监控系统中。

核心功能：
- 新奇度检测（与历史模式的差异）
- 复杂度评分（信息熵、预测误差）
- 不确定性度量
- 综合探索分数计算
- 输入优先级排序
- 探索-利用平衡（epsilon-greedy 策略）

探索评分公式：
exploration_score = novelty * w_novelty + complexity * w_complexity + uncertainty * w_uncertainty
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class CuriosityConfig:
    """探索驱动引擎配置"""
    
    # 启用/禁用
    enabled: bool = False
    
    # 好奇心指标权重
    novelty_weight: float = 0.4
    complexity_weight: float = 0.3
    uncertainty_weight: float = 0.3
    
    # 探索率参数（epsilon-greedy）
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01
    
    # 好奇心衰减率
    curiosity_decay_rate: float = 0.9
    
    # 历史窗口大小
    history_window_size: int = 100
    
    # 新奇度检测参数
    novelty_threshold: float = 0.1
    
    # 复杂度计算参数
    complexity_bins: int = 32
    
    # 输入优先级队列大小
    priority_queue_size: int = 10
    
    # 设备
    device: str = "cpu"


@dataclass
class CuriosityMetrics:
    """探索指标数据类"""
    
    novelty: float = 0.0
    complexity: float = 0.0
    uncertainty: float = 0.0
    curiosity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "novelty": self.novelty,
            "complexity": self.complexity,
            "uncertainty": self.uncertainty,
            "curiosity_score": self.curiosity_score,
        }


@dataclass
class PrioritizedInput:
    """带优先级的输入项"""
    
    input_id: str
    input_data: torch.Tensor
    priority: float
    curiosity_metrics: CuriosityMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


class NoveltyDetector:
    """
    新奇度检测器
    
    通过比较当前输入与历史输入模式的差异来计算新奇度。
    使用滑动窗口历史记录和余弦相似度度量。
    """
    
    def __init__(
        self,
        history_window_size: int = 100,
        device: str = "cpu"
    ):
        """
        初始化新奇度检测器
        
        Args:
            history_window_size: 历史窗口大小
            device: 计算设备
        """
        self.history_window_size = history_window_size
        self.device = device
        
        self._history: deque = deque(maxlen=history_window_size)
        self._history_embeddings: List[torch.Tensor] = []
    
    def compute_novelty(self, input_vector: torch.Tensor) -> float:
        """
        计算输入的新奇度
        
        Args:
            input_vector: 输入向量
            
        Returns:
            新奇度分数 (0.0 - 1.0)
        """
        input_vector = input_vector.to(self.device)
        
        if len(self._history_embeddings) == 0:
            self._add_to_history(input_vector)
            return 1.0
        
        input_norm = torch.norm(input_vector)
        if input_norm < 1e-8:
            return 0.0
        
        similarities = []
        for hist_vec in self._history_embeddings:
            hist_norm = torch.norm(hist_vec)
            if hist_norm < 1e-8:
                continue
            
            cosine_sim = torch.dot(input_vector, hist_vec) / (input_norm * hist_norm)
            similarities.append(cosine_sim.item())
        
        if len(similarities) == 0:
            self._add_to_history(input_vector)
            return 1.0
        
        avg_similarity = np.mean(similarities)
        novelty = 1.0 - max(0.0, min(1.0, avg_similarity))
        
        self._add_to_history(input_vector)
        
        return float(novelty)
    
    def _add_to_history(self, input_vector: torch.Tensor) -> None:
        """添加输入到历史记录"""
        self._history.append(input_vector.clone())
        self._history_embeddings.append(input_vector.clone())
        
        if len(self._history_embeddings) > self.history_window_size:
            self._history_embeddings.pop(0)
    
    def reset(self) -> None:
        """重置检测器"""
        self._history.clear()
        self._history_embeddings.clear()
    
    def get_history_size(self) -> int:
        """获取历史记录大小"""
        return len(self._history_embeddings)


class ComplexityScorer:
    """
    复杂度评分器
    
    使用信息熵和统计特性来评估输入的复杂度。
    """
    
    def __init__(
        self,
        num_bins: int = 32,
        device: str = "cpu"
    ):
        """
        初始化复杂度评分器
        
        Args:
            num_bins: 直方图分箱数
            device: 计算设备
        """
        self.num_bins = num_bins
        self.device = device
    
    def compute_complexity(self, input_vector: torch.Tensor) -> float:
        """
        计算输入的复杂度
        
        Args:
            input_vector: 输入向量
            
        Returns:
            复杂度分数 (0.0 - 1.0)
        """
        input_vector = input_vector.to(self.device)
        
        if input_vector.numel() == 0:
            return 0.0
        
        flat_input = input_vector.flatten()
        
        histogram = torch.histc(flat_input, bins=self.num_bins)
        
        total = histogram.sum()
        if total < 1e-8:
            return 0.0
        
        probabilities = histogram / total
        
        probabilities = probabilities + 1e-10
        entropy = -torch.sum(probabilities * torch.log2(probabilities))
        
        max_entropy = torch.log2(torch.tensor(self.num_bins, dtype=torch.float32, device=self.device))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        std_val = torch.std(flat_input)
        mean_val = torch.mean(torch.abs(flat_input))
        cv = std_val / (mean_val + 1e-8)
        normalized_cv = torch.sigmoid(cv)
        
        complexity = 0.6 * normalized_entropy.item() + 0.4 * normalized_cv.item()
        
        return float(min(1.0, max(0.0, complexity)))
    
    def reset(self) -> None:
        """重置评分器"""
        pass


class UncertaintyEstimator:
    """
    不确定性估计器
    
    基于预测误差和状态波动来估计不确定性。
    """
    
    def __init__(
        self,
        window_size: int = 50,
        device: str = "cpu"
    ):
        """
        初始化不确定性估计器
        
        Args:
            window_size: 滑动窗口大小
            device: 计算设备
        """
        self.window_size = window_size
        self.device = device
        
        self._prediction_errors: deque = deque(maxlen=window_size)
        self._state_history: deque = deque(maxlen=window_size)
    
    def compute_uncertainty(
        self,
        input_vector: torch.Tensor,
        prediction_error: Optional[float] = None
    ) -> float:
        """
        计算不确定性
        
        Args:
            input_vector: 输入向量
            prediction_error: 预测误差（可选）
            
        Returns:
            不确定性分数 (0.0 - 1.0)
        """
        input_vector = input_vector.to(self.device)
        
        uncertainty_components = []
        
        if prediction_error is not None:
            normalized_error = min(1.0, max(0.0, prediction_error))
            uncertainty_components.append(normalized_error)
            self._prediction_errors.append(prediction_error)
        
        self._state_history.append(input_vector.clone())
        
        if len(self._state_history) >= 5:
            recent_states = torch.stack(list(self._state_history)[-5:])
            state_variance = torch.var(recent_states, dim=0).mean()
            normalized_variance = torch.sigmoid(state_variance).item()
            uncertainty_components.append(normalized_variance)
        
        if len(self._prediction_errors) >= 3:
            recent_errors = list(self._prediction_errors)[-10:]
            error_trend = np.std(recent_errors) if len(recent_errors) > 1 else 0.0
            normalized_trend = min(1.0, error_trend)
            uncertainty_components.append(normalized_trend)
        
        if len(uncertainty_components) == 0:
            return 0.5
        
        uncertainty = float(np.mean(uncertainty_components))
        
        return min(1.0, max(0.0, uncertainty))
    
    def reset(self) -> None:
        """重置估计器"""
        self._prediction_errors.clear()
        self._state_history.clear()


class InputPriorityQueue:
    """
    输入优先级队列
    
    根据探索分数对输入进行优先级排序。
    """
    
    def __init__(self, max_size: int = 10):
        """
        初始化优先级队列
        
        Args:
            max_size: 队列最大大小
        """
        self.max_size = max_size
        self._queue: List[PrioritizedInput] = []
    
    def add_input(
        self,
        input_id: str,
        input_data: torch.Tensor,
        curiosity_metrics: CuriosityMetrics,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加输入到队列
        
        Args:
            input_id: 输入标识
            input_data: 输入数据
            curiosity_metrics: 好奇心指标
            metadata: 元数据
        """
        item = PrioritizedInput(
            input_id=input_id,
            input_data=input_data,
            priority=curiosity_metrics.curiosity_score,
            curiosity_metrics=curiosity_metrics,
            metadata=metadata or {}
        )
        
        self._queue.append(item)
        self._queue.sort(key=lambda x: x.priority, reverse=True)
        
        if len(self._queue) > self.max_size:
            self._queue = self._queue[:self.max_size]
    
    def get_highest_priority(self) -> Optional[PrioritizedInput]:
        """获取最高优先级的输入"""
        return self._queue[0] if self._queue else None
    
    def get_all_sorted(self) -> List[PrioritizedInput]:
        """获取所有按优先级排序的输入"""
        return list(self._queue)
    
    def pop_highest(self) -> Optional[PrioritizedInput]:
        """弹出最高优先级的输入"""
        return self._queue.pop(0) if self._queue else None
    
    def clear(self) -> None:
        """清空队列"""
        self._queue.clear()
    
    def size(self) -> int:
        """获取队列大小"""
        return len(self._queue)


class CuriosityEngine(nn.Module):
    """
    探索驱动引擎
    
    整合新奇度检测、复杂度评分、不确定性估计，
    实现探索驱动的输入选择和注意力分配。
    """
    
    def __init__(
        self,
        config: Optional[CuriosityConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化好奇心引擎
        
        Args:
            config: 好奇心配置
            device: 计算设备
        """
        super().__init__()
        
        self.config = config or CuriosityConfig()
        self.device = device or self.config.device
        
        self.novelty_detector = NoveltyDetector(
            history_window_size=self.config.history_window_size,
            device=self.device
        )
        
        self.complexity_scorer = ComplexityScorer(
            num_bins=self.config.complexity_bins,
            device=self.device
        )
        
        self.uncertainty_estimator = UncertaintyEstimator(
            window_size=self.config.history_window_size // 2,
            device=self.device
        )
        
        self.priority_queue = InputPriorityQueue(
            max_size=self.config.priority_queue_size
        )
        
        self._current_exploration_rate = self.config.exploration_rate
        self._step_count: int = 0
        
        self._curiosity_history: List[float] = []
        self._input_counter: int = 0
        
        logger.info(
            f"CuriosityEngine initialized: "
            f"enabled={self.config.enabled}, "
            f"exploration_rate={self._current_exploration_rate}"
        )
    
    def compute_curiosity(
        self,
        input_vector: torch.Tensor,
        prediction_error: Optional[float] = None
    ) -> CuriosityMetrics:
        """
        计算输入的综合好奇心分数
        
        Args:
            input_vector: 输入向量
            prediction_error: 预测误差（可选）
            
        Returns:
            好奇心指标
        """
        input_vector = input_vector.to(self.device)
        
        novelty = self.novelty_detector.compute_novelty(input_vector)
        complexity = self.complexity_scorer.compute_complexity(input_vector)
        uncertainty = self.uncertainty_estimator.compute_uncertainty(
            input_vector, prediction_error
        )
        
        curiosity_score = (
            novelty * self.config.novelty_weight +
            complexity * self.config.complexity_weight +
            uncertainty * self.config.uncertainty_weight
        )
        
        metrics = CuriosityMetrics(
            novelty=novelty,
            complexity=complexity,
            uncertainty=uncertainty,
            curiosity_score=curiosity_score
        )
        
        self._curiosity_history.append(curiosity_score)
        if len(self._curiosity_history) > self.config.history_window_size:
            self._curiosity_history.pop(0)
        
        return metrics
    
    def select_input(
        self,
        input_candidates: List[Tuple[str, torch.Tensor]],
        prediction_errors: Optional[Dict[str, float]] = None,
        epsilon_greedy: bool = True
    ) -> Tuple[str, torch.Tensor, CuriosityMetrics]:
        """
        使用好奇心驱动的策略选择输入
        
        Args:
            input_candidates: 候选输入列表 [(input_id, input_data), ...]
            prediction_errors: 各输入的预测误差字典
            epsilon_greedy: 是否使用 epsilon-greedy 策略
            
        Returns:
            (selected_id, selected_data, curiosity_metrics)
        """
        if not input_candidates:
            raise ValueError("No input candidates provided")
        
        if len(input_candidates) == 1:
            input_id, input_data = input_candidates[0]
            metrics = self.compute_curiosity(
                input_data,
                prediction_errors.get(input_id) if prediction_errors else None
            )
            return input_id, input_data, metrics
        
        candidate_metrics = []
        for input_id, input_data in input_candidates:
            pred_error = prediction_errors.get(input_id) if prediction_errors else None
            metrics = self.compute_curiosity(input_data, pred_error)
            candidate_metrics.append((input_id, input_data, metrics))
        
        if epsilon_greedy and torch.rand(1).item() < self._current_exploration_rate:
            idx = torch.randint(0, len(candidate_metrics), (1,)).item()
            selected = candidate_metrics[idx]
            logger.debug(f"Exploration: randomly selected input {selected[0]}")
        else:
            candidate_metrics.sort(key=lambda x: x[2].curiosity_score, reverse=True)
            selected = candidate_metrics[0]
            logger.debug(f"Exploitation: selected input {selected[0]} with curiosity={selected[2].curiosity_score:.4f}")
        
        self._step_count += 1
        self._decay_exploration_rate()
        
        return selected[0], selected[1], selected[2]
    
    def prioritize_inputs(
        self,
        input_candidates: List[Tuple[str, torch.Tensor, Dict[str, Any]]],
        prediction_errors: Optional[Dict[str, float]] = None
    ) -> List[PrioritizedInput]:
        """
        对输入进行优先级排序
        
        Args:
            input_candidates: 候选输入列表 [(input_id, input_data, metadata), ...]
            prediction_errors: 各输入的预测误差字典
            
        Returns:
            按优先级排序的输入列表
        """
        prioritized = []
        
        for input_id, input_data, metadata in input_candidates:
            pred_error = prediction_errors.get(input_id) if prediction_errors else None
            metrics = self.compute_curiosity(input_data, pred_error)
            
            self.priority_queue.add_input(
                input_id=input_id,
                input_data=input_data,
                curiosity_metrics=metrics,
                metadata=metadata
            )
            
            prioritized.append(PrioritizedInput(
                input_id=input_id,
                input_data=input_data,
                priority=metrics.curiosity_score,
                curiosity_metrics=metrics,
                metadata=metadata
            ))
        
        prioritized.sort(key=lambda x: x.priority, reverse=True)
        
        return prioritized
    
    def compute_attention_weights(
        self,
        input_vectors: List[torch.Tensor],
        prediction_errors: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        计算基于好奇心的注意力权重
        
        Args:
            input_vectors: 输入向量列表
            prediction_errors: 对应的预测误差列表
            
        Returns:
            注意力权重张量
        """
        if not input_vectors:
            return torch.tensor([], device=self.device)
        
        curiosity_scores = []
        for i, input_vec in enumerate(input_vectors):
            pred_error = prediction_errors[i] if prediction_errors else None
            metrics = self.compute_curiosity(input_vec, pred_error)
            curiosity_scores.append(metrics.curiosity_score)
        
        scores_tensor = torch.tensor(curiosity_scores, device=self.device)
        
        if scores_tensor.sum() < 1e-8:
            weights = torch.ones_like(scores_tensor) / len(scores_tensor)
        else:
            weights = scores_tensor / scores_tensor.sum()
        
        return weights
    
    def _decay_exploration_rate(self) -> None:
        """衰减探索率"""
        if self._current_exploration_rate > self.config.min_exploration_rate:
            self._current_exploration_rate *= self.config.exploration_decay
            self._current_exploration_rate = max(
                self._current_exploration_rate,
                self.config.min_exploration_rate
            )
    
    def get_curiosity_trend(self) -> Dict[str, float]:
        """
        获取好奇心趋势统计
        
        Returns:
            好奇心趋势字典
        """
        if not self._curiosity_history:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "recent_trend": 0.0
            }
        
        history = self._curiosity_history
        mean_val = float(np.mean(history))
        std_val = float(np.std(history))
        min_val = float(np.min(history))
        max_val = float(np.max(history))
        
        if len(history) >= 10:
            first_half = np.mean(history[:len(history)//2])
            second_half = np.mean(history[len(history)//2:])
            trend = second_half - first_half
        else:
            trend = 0.0
        
        return {
            "mean": mean_val,
            "std": std_val,
            "min": min_val,
            "max": max_val,
            "recent_trend": trend
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "enabled": self.config.enabled,
            "step_count": self._step_count,
            "exploration_rate": self._current_exploration_rate,
            "novelty_weight": self.config.novelty_weight,
            "complexity_weight": self.config.complexity_weight,
            "uncertainty_weight": self.config.uncertainty_weight,
            "curiosity_trend": self.get_curiosity_trend(),
            "priority_queue_size": self.priority_queue.size(),
            "history_size": self.novelty_detector.get_history_size(),
        }
    
    def reset(self) -> None:
        """重置好奇心引擎"""
        self.novelty_detector.reset()
        self.complexity_scorer.reset()
        self.uncertainty_estimator.reset()
        self.priority_queue.clear()
        
        self._current_exploration_rate = self.config.exploration_rate
        self._step_count = 0
        self._curiosity_history.clear()
        self._input_counter = 0
        
        logger.info("CuriosityEngine reset")
    
    def __repr__(self) -> str:
        return (
            f"CuriosityEngine("
            f"enabled={self.config.enabled}, "
            f"exploration_rate={self._current_exploration_rate:.4f}, "
            f"steps={self._step_count})"
        )


def create_curiosity_engine_from_config(
    config: CuriosityConfig,
    device: Optional[str] = None
) -> CuriosityEngine:
    """
    从配置创建好奇心引擎
    
    Args:
        config: 好奇心配置
        device: 计算设备
        
    Returns:
        CuriosityEngine 实例
    """
    engine = CuriosityEngine(
        config=config,
        device=device
    )
    return engine
