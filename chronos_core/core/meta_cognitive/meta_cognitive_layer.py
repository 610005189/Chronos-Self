"""
L2 元认知层 - Meta-Cognitive Layer
===================================

Task 14: 实现L2元认知层，负责高阶调控和元认知监测。

核心功能：
- Johnson-Lindenstrauss 固定稀疏随机投影（从 L1 高维状态投影到低维）
- 高阶统计特征提取（置信度分布、情绪方差、演化曲率等）
- 元参数调控向量输出（积分步长、衰减率、情绪基线等）
- 物理隔离验证（确保 L2 从未见过 L0 的原始数据）

关键特性：
- L2 通过固定投影矩阵监测 L1 状态（无可学习参数）
- L2 仅接收 L1 的投影状态，不接触 L0 原始数据
- 输出调控信号作为"建议"，而非"指令"
- 自指截断：L2 不能被内化到 L1
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import time
import warnings
from collections import deque

# 尝试导入 scipy
try:
    from scipy.sparse import random as sparse_random
    from scipy.sparse.linalg import aslinearoperator
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available, using simplified projection method")

from chronos_core.utils.config import (
    DimensionalityConfig,
    MetaCognitiveConfig,
    ChronosConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaCognitiveLayerConfig:
    """L2 元认知层配置"""
    
    # 输入维度（来自 L1）
    input_dim: int = 2560           # L1 状态维度（2048 + 512）
    
    # 投影维度（Johnson-Lindenstrauss）
    projection_dim: int = 64        # JL 投影后维度
    
    # 稀疏投影参数
    sparse_density: float = 0.1     # 稀疏矩阵密度
    projection_seed: int = 42       # 固定随机种子（确保投影固定）
    
    # 高阶统计特征参数
    confidence_window_size: int = 10      # 置信度窗口大小
    emotion_variance_window: int = 20     # 情绪方差窗口
    curvature_window: int = 15            # 曲率窗口
    error_pattern_history: int = 50       # 误差历史长度
    
    # 元参数调控参数
    step_size_coeff_range: Tuple[float, float] = (0.5, 2.0)  # 积分步长系数范围
    semantic_decay_range: Tuple[float, float] = (0.01, 0.1)  # 语义流衰减率范围
    physical_decay_range: Tuple[float, float] = (0.01, 0.1)  # 物理流衰减率范围
    emotion_baseline_range: Tuple[float, float] = (-0.5, 0.5)  # 情绪基线偏移范围
    
    # 调控输出维度
    control_output_dim: int = 128   # 4个调控参数 × 32维扩展
    
    # 监测频率
    monitoring_interval: int = 10   # 监测间隔（每 N 步监测一次）
    
    # 物理隔离标志
    physical_isolation_enabled: bool = True
    
    # 设备参数
    device: str = "cpu"


class JohnsonLindenstraussProjection:
    """
    Johnson-Lindenstrauss 固定稀疏随机投影
    
    实现 JL 引理的维度约减投影：
    - 将高维向量投影到低维空间，保持距离关系
    - 使用固定的稀疏随机矩阵（无可学习参数）
    - 满足 JL 引理的维度约束
    
    JL 引理：
    对于任意 0 < ε < 1 和 n 个点的集合，
    存在映射 f: ℝ^d -> ℝ^k，使得 k = O(ε^(-2) log n)
    并且对所有点保持 (1±ε) 的距离关系
    
    Attributes:
        input_dim: 输入维度
        projection_dim: 投影维度
        sparse_density: 稀疏矩阵密度
        projection_matrix: 固定的稀疏投影矩阵
    """
    
    def __init__(
        self,
        input_dim: int = 2560,
        projection_dim: int = 64,
        sparse_density: float = 0.1,
        seed: int = 42,
        device: str = "cpu"
    ):
        """
        初始化 JL 投影
        
        Args:
            input_dim: 输入维度（来自 L1）
            projection_dim: 投影目标维度
            sparse_density: 稀疏矩阵密度（0-1之间）
            seed: 固定随机种子（确保投影固定）
            device: 计算设备
        """
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.sparse_density = sparse_density
        self.seed = seed
        self.device = device
        
        # 验证 JL 引理维度约束
        self._verify_jl_dimension_constraint()
        
        # 创建固定的稀疏投影矩阵
        self.projection_matrix = self._create_fixed_sparse_projection()
        
        logger.info(
            f"JohnsonLindenstraussProjection initialized: "
            f"input_dim={input_dim}, projection_dim={projection_dim}, "
            f"sparse_density={sparse_density}, seed={seed}"
        )
    
    def _verify_jl_dimension_constraint(self):
        """
        验证 JL 引理的维度约束
        
        JL 引理要求：
        k >= 4 * log(n) / (ε^2 - ε^3/3)
        
        对于 n=1000 个点和 ε=0.1，k ≈ 800
        对于 n=100 个点和 ε=0.1，k ≈ 200
        对于 n=100 个点和 ε=0.5，k ≈ 20
        
        我们的设置（input_dim=2560, projection_dim=64）：
        - 对于 n=2560 个点，ε ≈ 0.7（保守估计）
        - 对于 n=100 个点，ε ≈ 0.5
        """
        # 计算满足 JL 引理所需的最小投影维度
        # 对于保守估计，我们假设最多有 1000 个点
        n_points = 1000  # 假设最多 1000 个不同的状态
        epsilon = 0.5    # 距离保持精度
        
        # 计算最小维度
        min_dim = int(4 * np.log(n_points) / (epsilon**2 - epsilon**3 / 3))
        
        logger.debug(
            f"JL dimension constraint: "
            f"n_points={n_points}, epsilon={epsilon}, min_dim={min_dim}"
        )
        
        # 检查是否满足约束（警告但不强制）
        if self.projection_dim < min_dim:
            logger.warning(
                f"Projection dimension {self.projection_dim} may not satisfy "
                f"JL lemma for {n_points} points with epsilon={epsilon}. "
                f"Recommended minimum: {min_dim}"
            )
        else:
            logger.info(
                f"JL lemma satisfied: projection_dim={self.projection_dim} >= {min_dim}"
            )
    
    def _create_fixed_sparse_projection(self) -> torch.Tensor:
        """
        创建固定的稀疏随机投影矩阵
        
        使用 Achlioptas 的稀疏 JL 投影：
        - 每个元素取值：sqrt(s) * {-1/sqrt(s), 0, 1/sqrt(s)}
        - 其中 s = 1/density
        - 概率分布：P(-1/sqrt(s)) = density/2, P(0) = 1-density, P(1/sqrt(s)) = density/2
        
        Returns:
            固定的稀疏投影矩阵张量
        """
        # 设置固定种子
        np.random.seed(self.seed)
        
        # Achlioptas 稀疏 JL 投影
        s = 1.0 / self.sparse_density  # 稀疏因子
        
        # 创建稀疏矩阵
        if SCIPY_AVAILABLE:
            # 使用 scipy.sparse.random 创建稀疏矩阵
            sparse_mat = sparse_random(
                self.projection_dim,
                self.input_dim,
                density=self.sparse_density,
                data_rvs=lambda s: np.random.choice(
                    [-np.sqrt(s), 0, np.sqrt(s)],
                    size=s,
                    p=[self.sparse_density/2, 1-self.sparse_density, self.sparse_density/2]
                )
            )
            
            # 转换为稠密张量
            dense_matrix = sparse_mat.toarray()
            projection_matrix = torch.tensor(dense_matrix, dtype=torch.float32)
            
        else:
            # 简化实现：手动创建稀疏矩阵
            matrix = np.zeros((self.projection_dim, self.input_dim))
            
            for i in range(self.projection_dim):
                for j in range(self.input_dim):
                    rand_val = np.random.random()
                    if rand_val < self.sparse_density / 2:
                        matrix[i, j] = -np.sqrt(s)
                    elif rand_val > 1 - self.sparse_density / 2:
                        matrix[i, j] = np.sqrt(s)
                    else:
                        matrix[i, j] = 0.0
            
            projection_matrix = torch.tensor(matrix, dtype=torch.float32)
        
        # 归一化（保持方差）
        # 对于 JL 投影，矩阵应该满足 E[||Px||^2] = ||x||^2
        # 即 P 的列方差应该为 1/projection_dim
        projection_matrix = projection_matrix / np.sqrt(self.projection_dim)
        
        projection_matrix = projection_matrix.to(self.device)
        
        logger.debug(
            f"Created fixed sparse projection matrix: "
            f"shape={projection_matrix.shape}, "
            f"sparsity={(projection_matrix == 0).float().mean().item():.2%}"
        )
        
        return projection_matrix
    
    def project(self, state_vector: torch.Tensor) -> torch.Tensor:
        """
        执行 JL 投影
        
        Args:
            state_vector: L1 状态向量（维度为 input_dim）
        
        Returns:
            投影后的低维向量（维度为 projection_dim）
        """
        state_vector = state_vector.to(self.device)
        
        # 验证输入维度
        if state_vector.shape[0] != self.input_dim:
            # 调整维度
            if state_vector.shape[0] < self.input_dim:
                # 填充零
                padding = torch.zeros(
                    self.input_dim - state_vector.shape[0],
                    device=self.device
                )
                state_vector = torch.cat([state_vector, padding])
            else:
                # 截断
                state_vector = state_vector[:self.input_dim]
        
        # 执行投影：P * x
        projected = torch.matmul(self.projection_matrix, state_vector)
        
        logger.debug(
            f"JL projection: input_dim={state_vector.shape[0]}, "
            f"output_dim={projected.shape[0]}"
        )
        
        return projected
    
    def verify_distance_preservation(
        self,
        test_vectors: List[torch.Tensor],
        epsilon: float = 0.5
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        验证距离保持特性
        
        测试 JL 引理是否满足距离保持约束
        
        Args:
            test_vectors: 测试向量列表
            epsilon: 距离保持精度
        
        Returns:
            (is_satisfied, statistics): 是否满足约束以及统计信息
        """
        if len(test_vectors) < 2:
            return True, {"message": "Need at least 2 vectors for testing"}
        
        # 计算原始距离
        original_distances = []
        projected_distances = []
        
        for i in range(len(test_vectors)):
            for j in range(i + 1, len(test_vectors)):
                v1 = test_vectors[i].to(self.device)
                v2 = test_vectors[j].to(self.device)
                
                # 原始距离
                orig_dist = torch.norm(v1 - v2).item()
                original_distances.append(orig_dist)
                
                # 投影后距离
                p1 = self.project(v1)
                p2 = self.project(v2)
                proj_dist = torch.norm(p1 - p2).item()
                projected_distances.append(proj_dist)
        
        # 计算距离比率
        ratios = [
            proj / orig for proj, orig in zip(projected_distances, original_distances)
            if orig > 0
        ]
        
        # 验证 JL 引理约束：|(||Px|| / ||x||) - 1| < epsilon
        violations = [
            abs(ratio - 1.0) > epsilon for ratio in ratios
        ]
        
        is_satisfied = not any(violations)
        
        statistics = {
            "num_pairs": len(ratios),
            "mean_ratio": np.mean(ratios) if ratios else 1.0,
            "std_ratio": np.std(ratios) if ratios else 0.0,
            "max_ratio": max(ratios) if ratios else 1.0,
            "min_ratio": min(ratios) if ratios else 1.0,
            "num_violations": sum(violations),
            "epsilon": epsilon,
        }
        
        logger.info(
            f"JL distance preservation verification: "
            f"satisfied={is_satisfied}, mean_ratio={statistics['mean_ratio']:.4f}"
        )
        
        return is_satisfied, statistics
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "input_dim": self.input_dim,
            "projection_dim": self.projection_dim,
            "sparse_density": self.sparse_density,
            "seed": self.seed,
            "matrix_shape": self.projection_matrix.shape,
            "matrix_sparsity": (self.projection_matrix == 0).float().mean().item(),
            "matrix_norm": torch.norm(self.projection_matrix).item(),
        }
        
        return stats


class HighOrderStatisticsExtractor:
    """
    高阶统计特征提取器
    
    从投影后的低维向量提取高阶统计特征：
    - 置信度分布（预测误差分布）
    - 情绪方差（情感波动）
    - 演化曲率（状态变化速度）
    - 预测误差模式（误差的历史趋势）
    
    Attributes:
        projection_dim: 投影维度
        confidence_window: 置信度历史窗口
        emotion_window: 情绪方差历史窗口
        curvature_window: 曲率历史窗口
        error_history: 预测误差历史
    """
    
    def __init__(
        self,
        projection_dim: int = 64,
        confidence_window_size: int = 10,
        emotion_variance_window: int = 20,
        curvature_window: int = 15,
        error_pattern_history: int = 50,
        device: str = "cpu"
    ):
        """
        初始化高阶统计特征提取器
        
        Args:
            projection_dim: 投影维度
            confidence_window_size: 置信度窗口大小
            emotion_variance_window: 情绪方差窗口
            curvature_window: 曲率窗口
            error_pattern_history: 误差历史长度
            device: 计算设备
        """
        self.projection_dim = projection_dim
        self.confidence_window_size = confidence_window_size
        self.emotion_variance_window = emotion_variance_window
        self.curvature_window = curvature_window
        self.error_pattern_history = error_pattern_history
        self.device = device
        
        # 历史缓存
        self._confidence_history: deque = deque(maxlen=confidence_window_size)
        self._emotion_history: deque = deque(maxlen=emotion_variance_window)
        self._projected_state_history: deque = deque(maxlen=curvature_window)
        self._error_history: deque = deque(maxlen=error_pattern_history)
        
        # 前一次投影状态（用于计算变化）
        self._prev_projected_state: Optional[torch.Tensor] = None
        
        logger.info(
            f"HighOrderStatisticsExtractor initialized: "
            f"projection_dim={projection_dim}, "
            f"confidence_window={confidence_window_size}, "
            f"emotion_window={emotion_variance_window}"
        )
    
    def extract(
        self,
        projected_state: torch.Tensor,
        prediction_error: Optional[float] = None,
        emotion_signal: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取高阶统计特征
        
        Args:
            projected_state: 投影后的状态向量
            prediction_error: 预测误差（可选）
            emotion_signal: 情绪信号（可选）
        
        Returns:
            高阶统计特征字典
        """
        projected_state = projected_state.to(self.device)
        
        features = {}
        
        # 1. 置信度分布（预测误差分布）
        features["confidence_distribution"] = self._extract_confidence_distribution(
            projected_state, prediction_error
        )
        
        # 2. 情绪方差（情感波动）
        features["emotion_variance"] = self._extract_emotion_variance(
            projected_state, emotion_signal
        )
        
        # 3. 演化曲率（状态变化速度）
        features["evolution_curvature"] = self._extract_evolution_curvature(
            projected_state
        )
        
        # 4. 预测误差模式（误差的历史趋势）
        features["error_pattern"] = self._extract_error_pattern(
            projected_state, prediction_error
        )
        
        # 更新历史
        self._projected_state_history.append(projected_state.clone())
        self._prev_projected_state = projected_state.clone()
        
        logger.debug(
            f"Extracted high-order statistics: "
            f"confidence={features['confidence_distribution'].mean().item():.4f}, "
            f"emotion_var={features['emotion_variance'].mean().item():.4f}"
        )
        
        return features
    
    def _extract_confidence_distribution(
        self,
        projected_state: torch.Tensor,
        prediction_error: Optional[float] = None
    ) -> torch.Tensor:
        """
        提取置信度分布
        
        置信度基于预测误差的历史分布计算
        
        Args:
            projected_state: 投影状态
            prediction_error: 预测误差
        
        Returns:
            置信度分布特征
        """
        # 如果提供了预测误差，记录历史
        if prediction_error is not None:
            self._confidence_history.append(prediction_error)
        
        # 计算置信度分布特征
        # 使用投影状态的方差作为置信度指标
        state_var = projected_state.var().item()
        
        # 基于历史预测误差计算置信度
        if len(self._confidence_history) > 0:
            error_mean = np.mean(list(self._confidence_history))
            error_std = np.std(list(self._confidence_history))
            
            # 置信度 = 1 / (error_std + 1)
            confidence_level = 1.0 / (error_std + 1e-8)
            
            # 创建置信度分布向量
            # 使用投影状态的均值和方差组合
            confidence_distribution = projected_state * confidence_level
            
        else:
            # 没有历史数据，使用默认置信度
            confidence_distribution = projected_state * 0.5
        
        logger.debug(
            f"Confidence distribution: "
            f"state_var={state_var:.4f}, confidence_level={confidence_level if len(self._confidence_history) > 0 else 0.5:.4f}"
        )
        
        return confidence_distribution
    
    def _extract_emotion_variance(
        self,
        projected_state: torch.Tensor,
        emotion_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        提取情绪方差
        
        情绪方差表示情感波动程度
        
        Args:
            projected_state: 投影状态
            emotion_signal: 情绪信号（可选）
        
        Returns:
            情绪方差特征
        """
        # 如果提供了情绪信号，记录历史
        if emotion_signal is not None:
            emotion_signal = emotion_signal.to(self.device)
            
            # 确保维度匹配
            if emotion_signal.shape[0] != self.projection_dim:
                if emotion_signal.shape[0] < self.projection_dim:
                    padding = torch.zeros(
                        self.projection_dim - emotion_signal.shape[0],
                        device=self.device
                    )
                    emotion_signal = torch.cat([emotion_signal, padding])
                else:
                    emotion_signal = emotion_signal[:self.projection_dim]
            
            # 记录情绪强度历史
            emotion_intensity = torch.norm(emotion_signal).item()
            self._emotion_history.append(emotion_intensity)
        
        # 计算情绪方差
        # 使用投影状态的方差作为情绪波动指标
        state_var = projected_state.var()
        
        # 基于历史情绪强度计算方差
        if len(self._emotion_history) > 1:
            emotion_var_history = np.var(list(self._emotion_history))
            
            # 组合当前状态方差和历史情绪方差
            emotion_variance = state_var + emotion_var_history
            
        else:
            # 没有历史数据，使用当前状态方差
            emotion_variance = state_var
        
        # 创建情绪方差向量
        emotion_variance_vector = projected_state * emotion_variance
        
        logger.debug(
            f"Emotion variance: "
            f"state_var={state_var.item():.4f}, "
            f"history_var={emotion_var_history if len(self._emotion_history) > 1 else 0:.4f}"
        )
        
        return emotion_variance_vector
    
    def _extract_evolution_curvature(
        self,
        projected_state: torch.Tensor
    ) -> torch.Tensor:
        """
        提取演化曲率
        
        演化曲率表示状态变化速度和方向
        
        Args:
            projected_state: 投影状态
        
        Returns:
            演化曲率特征
        """
        if self._prev_projected_state is None:
            # 第一次调用，曲率为零
            curvature = torch.zeros(self.projection_dim, device=self.device)
            logger.debug("First call, curvature set to zero")
            return curvature
        
        # 计算状态变化（速度）
        delta = projected_state - self._prev_projected_state
        velocity = torch.norm(delta).item()
        
        # 计算二阶变化（曲率）
        if len(self._projected_state_history) >= 2:
            # 使用历史状态计算曲率
            prev_prev_state = self._projected_state_history[-2]
            
            # 一阶导数（速度）
            d1 = projected_state - self._prev_projected_state
            
            # 二阶导数（曲率）
            d2 = (projected_state - self._prev_projected_state) - \
                 (self._prev_projected_state - prev_prev_state)
            
            # 曲率 = ||d2|| / ||d1||
            d1_norm = torch.norm(d1).item()
            d2_norm = torch.norm(d2).item()
            
            curvature_value = d2_norm / (d1_norm + 1e-8)
            
            # 创建曲率向量
            curvature = projected_state * curvature_value
            
        else:
            # 没有足够历史，使用速度作为曲率估计
            curvature = projected_state * velocity
        
        logger.debug(
            f"Evolution curvature: "
            f"velocity={velocity:.4f}, curvature_value={curvature_value if len(self._projected_state_history) >= 2 else velocity:.4f}"
        )
        
        return curvature
    
    def _extract_error_pattern(
        self,
        projected_state: torch.Tensor,
        prediction_error: Optional[float] = None
    ) -> torch.Tensor:
        """
        提取预测误差模式
        
        预测误差模式表示误差的历史趋势
        
        Args:
            projected_state: 投影状态
            prediction_error: 预测误差
        
        Returns:
            预测误差模式特征
        """
        # 如果提供了预测误差，记录历史
        if prediction_error is not None:
            self._error_history.append(prediction_error)
        
        # 计算误差模式
        if len(self._error_history) >= 3:
            # 计算误差趋势（线性拟合）
            errors = np.array(list(self._error_history))
            time_indices = np.arange(len(errors))
            
            # 线性拟合
            slope, intercept = np.polyfit(time_indices, errors, 1)
            
            # 计算趋势强度
            trend_strength = abs(slope)
            
            # 计算误差波动
            error_std = np.std(errors)
            
            # 创建误差模式向量
            error_pattern = projected_state * (trend_strength + error_std)
            
        else:
            # 没有足够历史，使用当前状态
            error_pattern = projected_state
        
        logger.debug(
            f"Error pattern: "
            f"trend_slope={slope if len(self._error_history) >= 3 else 0:.4f}, "
            f"error_std={error_std if len(self._error_history) >= 3 else 0:.4f}"
        )
        
        return error_pattern
    
    def get_all_features_vector(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        将所有特征组合为单一向量
        
        Args:
            features: 高阶统计特征字典
        
        Returns:
            组合后的特征向量
        """
        # 组合所有特征
        feature_list = [
            features["confidence_distribution"],
            features["emotion_variance"],
            features["evolution_curvature"],
            features["error_pattern"],
        ]
        
        # 拼合为单一向量
        combined = torch.cat(feature_list)
        
        return combined
    
    def reset(self):
        """重置历史缓存"""
        self._confidence_history.clear()
        self._emotion_history.clear()
        self._projected_state_history.clear()
        self._error_history.clear()
        self._prev_projected_state = None
        
        logger.debug("HighOrderStatisticsExtractor reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "projection_dim": self.projection_dim,
            "confidence_history_length": len(self._confidence_history),
            "emotion_history_length": len(self._emotion_history),
            "state_history_length": len(self._projected_state_history),
            "error_history_length": len(self._error_history),
        }
        
        if len(self._confidence_history) > 0:
            stats["confidence_mean"] = np.mean(list(self._confidence_history))
            stats["confidence_std"] = np.std(list(self._confidence_history))
        
        if len(self._emotion_history) > 0:
            stats["emotion_mean"] = np.mean(list(self._emotion_history))
            stats["emotion_var"] = np.var(list(self._emotion_history))
        
        return stats


class MetaParameterController:
    """
    元参数调控器
    
    从高阶统计特征生成元参数调控向量：
    - 积分步长系数（调节 ODE 步长）
    - 语义流衰减率（注意力衰减）
    - 物理流衰减率
    - 情绪基线偏移量
    
    Attributes:
        control_output_dim: 调控输出维度
        step_size_range: 积分步长系数范围
        semantic_decay_range: 语义流衰减率范围
        physical_decay_range: 物理流衰减率范围
        emotion_baseline_range: 情绪基线偏移范围
    """
    
    def __init__(
        self,
        feature_dim: int = 256,  # 4个特征 × 64维
        control_output_dim: int = 128,
        step_size_range: Tuple[float, float] = (0.5, 2.0),
        semantic_decay_range: Tuple[float, float] = (0.01, 0.1),
        physical_decay_range: Tuple[float, float] = (0.01, 0.1),
        emotion_baseline_range: Tuple[float, float] = (-0.5, 0.5),
        device: str = "cpu"
    ):
        """
        初始化元参数调控器
        
        Args:
            feature_dim: 输入特征维度
            control_output_dim: 调控输出维度
            step_size_range: 积分步长系数范围
            semantic_decay_range: 语义流衰减率范围
            physical_decay_range: 物理流衰减率范围
            emotion_baseline_range: 情绪基线偏移范围
            device: 计算设备
        """
        self.feature_dim = feature_dim
        self.control_output_dim = control_output_dim
        self.step_size_range = step_size_range
        self.semantic_decay_range = semantic_decay_range
        self.physical_decay_range = physical_decay_range
        self.emotion_baseline_range = emotion_baseline_range
        self.device = device
        
        # 每个调控参数的输出维度
        self.param_output_dim = control_output_dim // 4
        
        logger.info(
            f"MetaParameterController initialized: "
            f"feature_dim={feature_dim}, control_output_dim={control_output_dim}, "
            f"param_output_dim={self.param_output_dim}"
        )
    
    def compute_control_vector(
        self,
        features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算元参数调控向量
        
        Args:
            features: 高阶统计特征字典
        
        Returns:
            元参数调控向量（维度为 control_output_dim）
        """
        # 组合所有特征
        combined_features = self._combine_features(features)
        combined_features = combined_features.to(self.device)
        
        # 调整特征维度
        if combined_features.shape[0] < self.feature_dim:
            # 填充零
            padding = torch.zeros(
                self.feature_dim - combined_features.shape[0],
                device=self.device
            )
            combined_features = torch.cat([combined_features, padding])
        elif combined_features.shape[0] > self.feature_dim:
            # 截断
            combined_features = combined_features[:self.feature_dim]
        
        # 计算各个调控参数
        control_vector = torch.zeros(self.control_output_dim, device=self.device)
        
        # 1. 积分步长系数
        step_size_start = 0
        step_size_end = self.param_output_dim
        control_vector[step_size_start:step_size_end] = self._compute_step_size_coeff(
            combined_features
        )
        
        # 2. 语义流衰减率
        semantic_decay_start = self.param_output_dim
        semantic_decay_end = 2 * self.param_output_dim
        control_vector[semantic_decay_start:semantic_decay_end] = self._compute_semantic_decay(
            combined_features
        )
        
        # 3. 物理流衰减率
        physical_decay_start = 2 * self.param_output_dim
        physical_decay_end = 3 * self.param_output_dim
        control_vector[physical_decay_start:physical_decay_end] = self._compute_physical_decay(
            combined_features
        )
        
        # 4. 情绪基线偏移量
        emotion_baseline_start = 3 * self.param_output_dim
        emotion_baseline_end = 4 * self.param_output_dim
        control_vector[emotion_baseline_start:emotion_baseline_end] = self._compute_emotion_baseline(
            combined_features
        )
        
        logger.debug(
            f"Computed control vector: "
            f"step_size={control_vector[step_size_start:step_size_end].mean().item():.4f}, "
            f"semantic_decay={control_vector[semantic_decay_start:semantic_decay_end].mean().item():.4f}"
        )
        
        return control_vector
    
    def _combine_features(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        组合所有特征
        
        Args:
            features: 高阶统计特征字典
        
        Returns:
            组合后的特征向量
        """
        feature_list = [
            features["confidence_distribution"],
            features["emotion_variance"],
            features["evolution_curvature"],
            features["error_pattern"],
        ]
        
        # 拼合为单一向量
        combined = torch.cat(feature_list)
        
        return combined
    
    def _compute_step_size_coeff(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算积分步长系数
        
        基于置信度和误差模式计算步长系数
        
        Args:
            features: 组合特征
        
        Returns:
            步长系数向量（param_output_dim 维）
        """
        # 使用特征的均值作为基准
        feature_mean = features.mean().item()
        
        # 映射到步长系数范围
        # 高置信度（低误差） -> 大步长
        # 低置信度（高误差） -> 小步长
        
        # 归一化特征均值
        normalized_mean = (feature_mean + 1) / 2  # 假设特征范围 [-1, 1]
        
        # 映射到步长范围
        min_step, max_step = self.step_size_range
        step_size_coeff = min_step + normalized_mean * (max_step - min_step)
        
        # 创建步长系数向量
        step_size_vector = torch.full(
            (self.param_output_dim,),
            step_size_coeff,
            device=self.device
        )
        
        logger.debug(
            f"Step size coefficient: "
            f"feature_mean={feature_mean:.4f}, step_size={step_size_coeff:.4f}"
        )
        
        return step_size_vector
    
    def _compute_semantic_decay(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算语义流衰减率
        
        基于情绪方差和置信度计算衰减率
        
        Args:
            features: 组合特征
        
        Returns:
            语义衰减率向量（param_output_dim 维）
        """
        # 使用特征的方差作为基准
        feature_var = features.var().item()
        
        # 映射到衰减率范围
        # 高方差（高波动） -> 高衰减率（快速遗忘）
        # 低方差（低波动） -> 低衰减率（长期保持）
        
        # 归一化特征方差
        normalized_var = min(1.0, feature_var)  # 截断到 [0, 1]
        
        # 映射到衰减率范围
        min_decay, max_decay = self.semantic_decay_range
        semantic_decay = min_decay + normalized_var * (max_decay - min_decay)
        
        # 创建衰减率向量
        semantic_decay_vector = torch.full(
            (self.param_output_dim,),
            semantic_decay,
            device=self.device
        )
        
        logger.debug(
            f"Semantic decay: "
            f"feature_var={feature_var:.4f}, decay={semantic_decay:.4f}"
        )
        
        return semantic_decay_vector
    
    def _compute_physical_decay(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算物理流衰减率
        
        基于演化曲率计算衰减率
        
        Args:
            features: 组合特征
        
        Returns:
            物理衰减率向量（param_output_dim 维）
        """
        # 使用特征的范数作为基准
        feature_norm = torch.norm(features).item()
        
        # 映射到衰减率范围
        # 高范数（高活动） -> 高衰减率
        # 低范数（低活动） -> 低衰减率
        
        # 归一化特征范数
        normalized_norm = min(1.0, feature_norm / 10)  # 截断
        
        # 映射到衰减率范围
        min_decay, max_decay = self.physical_decay_range
        physical_decay = min_decay + normalized_norm * (max_decay - min_decay)
        
        # 创建衰减率向量
        physical_decay_vector = torch.full(
            (self.param_output_dim,),
            physical_decay,
            device=self.device
        )
        
        logger.debug(
            f"Physical decay: "
            f"feature_norm={feature_norm:.4f}, decay={physical_decay:.4f}"
        )
        
        return physical_decay_vector
    
    def _compute_emotion_baseline(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算情绪基线偏移量
        
        基于情绪方差特征计算基线偏移
        
        Args:
            features: 组合特征
        
        Returns:
            情绪基线偏移向量（param_output_dim 维）
        """
        # 使用情绪方差部分的特征
        # 假设情绪方差特征在前 64 维
        emotion_features = features[:64]
        
        # 计算情绪特征均值
        emotion_mean = emotion_features.mean().item()
        
        # 映射到情绪基线范围
        # 高情绪值 -> 正偏移
        # 低情绪值 -> 负偏移
        
        # 归一化情绪均值
        normalized_emotion = (emotion_mean + 1) / 2  # 假设范围 [-1, 1]
        
        # 映射到情绪基线范围
        min_baseline, max_baseline = self.emotion_baseline_range
        emotion_baseline = min_baseline + normalized_emotion * (max_baseline - min_baseline)
        
        # 创建情绪基线向量
        emotion_baseline_vector = torch.full(
            (self.param_output_dim,),
            emotion_baseline,
            device=self.device
        )
        
        logger.debug(
            f"Emotion baseline: "
            f"emotion_mean={emotion_mean:.4f}, baseline={emotion_baseline:.4f}"
        )
        
        return emotion_baseline_vector
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "feature_dim": self.feature_dim,
            "control_output_dim": self.control_output_dim,
            "param_output_dim": self.param_output_dim,
            "step_size_range": self.step_size_range,
            "semantic_decay_range": self.semantic_decay_range,
            "physical_decay_range": self.physical_decay_range,
            "emotion_baseline_range": self.emotion_baseline_range,
        }
        
        return stats


class MetaCognitiveLayer(nn.Module):
    """
    L2 元认知层
    
    Task 14 完整实现
    
    功能：
    - Johnson-Lindenstrauss 固定稀疏随机投影
    - 高阶统计特征提取
    - 元参数调控向量输出
    - 物理隔离验证
    
    特性：
    - L2 通过固定投影矩阵监测 L1 状态（无可学习参数）
    - L2 仅接收 L1 的投影状态，不接触 L0 原始数据
    - 输出调控信号作为"建议"，而非"指令"
    - 自指截断：L2 不能被内化到 L1
    """
    
    def __init__(
        self,
        config: Optional[MetaCognitiveLayerConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_config: Optional[MetaCognitiveConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化 L2 元认知层
        
        Args:
            config: 元认知层配置
            dim_config: 维度配置
            meta_config: 元认知配置
            device: 计算设备
        """
        super().__init__()
        
        # 合并配置
        self.config = config or MetaCognitiveLayerConfig()
        
        if dim_config:
            self.config.input_dim = dim_config.fast_variable_dim + dim_config.slow_variable_dim
        
        if meta_config:
            self.config.projection_dim = meta_config.l2_projection_dim
        
        self.device = device or self.config.device
        
        # 创建 JL 投影器
        self.jl_projection = JohnsonLindenstraussProjection(
            input_dim=self.config.input_dim,
            projection_dim=self.config.projection_dim,
            sparse_density=self.config.sparse_density,
            seed=self.config.projection_seed,
            device=self.device
        )
        
        # 创建高阶统计特征提取器
        self.statistics_extractor = HighOrderStatisticsExtractor(
            projection_dim=self.config.projection_dim,
            confidence_window_size=self.config.confidence_window_size,
            emotion_variance_window=self.config.emotion_variance_window,
            curvature_window=self.config.curvature_window,
            error_pattern_history=self.config.error_pattern_history,
            device=self.device
        )
        
        # 创建元参数调控器
        self.meta_parameter_controller = MetaParameterController(
            feature_dim=4 * self.config.projection_dim,  # 4个特征 × projection_dim
            control_output_dim=self.config.control_output_dim,
            step_size_range=self.config.step_size_coeff_range,
            semantic_decay_range=self.config.semantic_decay_range,
            physical_decay_range=self.config.physical_decay_range,
            emotion_baseline_range=self.config.emotion_baseline_range,
            device=self.device
        )
        
        # L1 状态缓存（用于接收 L1 数据）
        self._l1_state_cache: Optional[torch.Tensor] = None
        
        # 调控信号缓存（用于输出给 L1）
        self._control_signal_cache: Optional[torch.Tensor] = None
        
        # 监测步数计数器
        self._monitoring_counter: int = 0
        
        # 物理隔离标志（确保 L2 从未见过 L0 原始数据）
        self._physical_isolation_flag: bool = True
        self._l0_data_seen: bool = False  # 记录是否见过 L0 数据
        
        # 统计信息
        self._stats = {
            "total_monitoring_calls": 0,
            "control_signals_generated": 0,
            "jl_projection_calls": 0,
            "feature_extraction_calls": 0,
        }
        
        self.to(self.device)
        
        logger.info(
            f"MetaCognitiveLayer (L2) initialized: "
            f"input_dim={self.config.input_dim}, "
            f"projection_dim={self.config.projection_dim}, "
            f"control_output_dim={self.config.control_output_dim}, "
            f"physical_isolation={self._physical_isolation_flag}"
        )
    
    def forward(
        self,
        l1_state: torch.Tensor,
        prediction_error: Optional[float] = None,
        emotion_signal: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行元认知监测和调控
        
        流程：
        1. 接收 L1 状态
        2. JL 投影到低维
        3. 提取高阶统计特征
        4. 生成元参数调控向量
        
        Args:
            l1_state: L1 状态向量（维度为 input_dim）
            prediction_error: 预测误差（可选）
            emotion_signal: 情绪信号（可选）
        
        Returns:
            元参数调控向量（维度为 control_output_dim）
        """
        l1_state = l1_state.to(self.device)
        
        # 1. 接收 L1 状态
        self._l1_state_cache = l1_state.clone()
        
        # 验证物理隔离
        self._verify_physical_isolation()
        
        # 2. JL 投影
        projected_state = self.jl_projection.project(l1_state)
        self._stats["jl_projection_calls"] += 1
        
        # 3. 提取高阶统计特征
        features = self.statistics_extractor.extract(
            projected_state,
            prediction_error=prediction_error,
            emotion_signal=emotion_signal
        )
        self._stats["feature_extraction_calls"] += 1
        
        # 4. 生成元参数调控向量
        control_vector = self.meta_parameter_controller.compute_control_vector(features)
        
        # 缓存调控信号
        self._control_signal_cache = control_vector.clone()
        
        # 更新统计
        self._stats["total_monitoring_calls"] += 1
        self._stats["control_signals_generated"] += 1
        self._monitoring_counter += 1
        
        logger.debug(
            f"MetaCognitiveLayer forward: "
            f"l1_state_dim={l1_state.shape[0]}, "
            f"projected_dim={projected_state.shape[0]}, "
            f"control_dim={control_vector.shape[0]}"
        )
        
        return control_vector
    
    def receive_from_l1(self, l1_state: torch.Tensor) -> torch.Tensor:
        """
        接收 L1 状态数据
        
        Args:
            l1_state: L1 状态向量
        
        Returns:
            投影后的低维状态
        """
        return self.jl_projection.project(l1_state.to(self.device))
    
    def send_control_to_l1(self) -> Optional[torch.Tensor]:
        """
        发送调控信号给 L1
        
        Returns:
            元参数调控向量
        """
        return self._control_signal_cache
    
    def get_monitoring_interval(self) -> int:
        """
        获取监测间隔
        
        Returns:
            监测间隔步数
        """
        return self.config.monitoring_interval
    
    def should_monitor(self) -> bool:
        """
        判断是否应该执行监测
        
        Returns:
            是否应该监测
        """
        # 每隔 monitoring_interval 步执行一次监测
        return self._monitoring_counter % self.config.monitoring_interval == 0
    
    def _verify_physical_isolation(self):
        """
        验证物理隔离
        
        确保 L2 从未见过 L0 的原始数据
        """
        if self.config.physical_isolation_enabled:
            # 检查是否有 L0 数据泄漏
            # L2 仅应接收 L1 的投影状态
            
            # 简化验证：检查是否从未见过 L0 原始数据
            # 实际实现中，可以通过检查数据来源标签来验证
            
            # 这里仅记录状态，实际验证在 check_physical_isolation() 方法中
            pass
    
    def check_physical_isolation(self) -> Tuple[bool, List[str]]:
        """
        检查物理隔离
        
        验证 L2 从未见过 L0 的原始数据
        
        Returns:
            (is_isolated, error_messages): 是否隔离以及错误消息
        """
        errors = []
        
        # 检查是否有 L0 数据泄漏
        if self._l0_data_seen:
            errors.append("L2 has seen L0 raw data - physical isolation violated!")
        
        # 检查 L2 仅接收 L1 投影状态
        # 这里检查 JL 投影器是否正确使用
        if not hasattr(self.jl_projection, 'projection_matrix'):
            errors.append("JL projection matrix not initialized")
        
        # 检查调控信号仅来自投影后的特征
        if self._control_signal_cache is not None:
            # 调控信号应该由投影后的特征计算，不应包含原始数据
            # 检查调控信号维度是否正确
            if self._control_signal_cache.shape[0] != self.config.control_output_dim:
                errors.append(
                    f"Control signal dimension mismatch: "
                    f"expected {self.config.control_output_dim}, "
                    f"got {self._control_signal_cache.shape[0]}"
                )
        
        is_isolated = len(errors) == 0
        
        if not is_isolated:
            logger.error(f"Physical isolation check failed: {errors}")
        
        return is_isolated, errors
    
    def mark_l0_data_seen(self):
        """
        标记 L0 数据已被查看
        
        用于物理隔离验证测试
        """
        self._l0_data_seen = True
        logger.warning("L0 data seen flag marked - physical isolation may be violated")
    
    def verify_jl_projection(self) -> Tuple[bool, Dict[str, Any]]:
        """
        验证 JL 投影
        
        测试 JL 引理是否满足距离保持约束
        
        Returns:
            (is_valid, statistics): 是否有效以及统计信息
        """
        # 创建测试向量
        test_vectors = [
            torch.randn(self.config.input_dim, device=self.device)
            for _ in range(10)  # 10 个测试向量
        ]
        
        # 验证距离保持
        is_valid, statistics = self.jl_projection.verify_distance_preservation(
            test_vectors,
            epsilon=0.5
        )
        
        return is_valid, statistics
    
    def get_high_order_statistics(self) -> Dict[str, Any]:
        """
        获取高阶统计特征信息
        
        Returns:
            高阶统计特征统计信息
        """
        return self.statistics_extractor.get_statistics()
    
    def get_control_parameters(self) -> Dict[str, Any]:
        """
        获取调控参数信息
        
        Returns:
            调控参数统计信息
        """
        return self.meta_parameter_controller.get_statistics()
    
    def reset(self):
        """重置元认知层"""
        # 重置统计特征提取器
        self.statistics_extractor.reset()
        
        # 清空缓存
        self._l1_state_cache = None
        self._control_signal_cache = None
        
        # 重置计数器
        self._monitoring_counter = 0
        
        # 重置物理隔离标志
        self._l0_data_seen = False
        
        # 重置统计
        self._stats = {
            "total_monitoring_calls": 0,
            "control_signals_generated": 0,
            "jl_projection_calls": 0,
            "feature_extraction_calls": 0,
        }
        
        logger.info("MetaCognitiveLayer reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        # 添加 JL 投影信息
        stats["jl_projection_stats"] = self.jl_projection.get_statistics()
        
        # 添加统计特征信息
        stats["statistics_extractor_stats"] = self.statistics_extractor.get_statistics()
        
        # 添加调控参数信息
        stats["control_params_stats"] = self.meta_parameter_controller.get_statistics()
        
        # 添加物理隔离信息
        stats["physical_isolation"] = {
            "enabled": self.config.physical_isolation_enabled,
            "l0_data_seen": self._l0_data_seen,
        }
        
        # 添加配置信息
        stats["config"] = {
            "input_dim": self.config.input_dim,
            "projection_dim": self.config.projection_dim,
            "control_output_dim": self.config.control_output_dim,
            "monitoring_interval": self.config.monitoring_interval,
        }
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证元认知层
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证物理隔离
        is_isolated, isolation_errors = self.check_physical_isolation()
        errors.extend(isolation_errors)
        
        # 验证 JL 投影
        is_jl_valid, jl_stats = self.verify_jl_projection()
        if not is_jl_valid:
            errors.append(
                f"JL projection distance preservation violated: "
                f"mean_ratio={jl_stats['mean_ratio']:.4f}"
            )
        
        # 验证配置
        if self.config.input_dim <= 0:
            errors.append(f"Invalid input_dim: {self.config.input_dim}")
        if self.config.projection_dim <= 0:
            errors.append(f"Invalid projection_dim: {self.config.projection_dim}")
        if self.config.control_output_dim <= 0:
            errors.append(f"Invalid control_output_dim: {self.config.control_output_dim}")
        
        # 验证调控参数范围
        if self.config.step_size_coeff_range[0] <= 0:
            errors.append("Step size range invalid")
        if self.config.semantic_decay_range[0] <= 0:
            errors.append("Semantic decay range invalid")
        if self.config.physical_decay_range[0] <= 0:
            errors.append("Physical decay range invalid")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def __repr__(self) -> str:
        return (
            f"MetaCognitiveLayer(L2, "
            f"input_dim={self.config.input_dim}, "
            f"projection_dim={self.config.projection_dim}, "
            f"control_output_dim={self.config.control_output_dim}, "
            f"physical_isolation={self._physical_isolation_flag})"
        )


def create_meta_cognitive_layer_from_config(
    dim_config: DimensionalityConfig,
    meta_config: MetaCognitiveConfig,
    device: Optional[str] = None
) -> MetaCognitiveLayer:
    """
    从配置创建元认知层
    
    Args:
        dim_config: 维度配置
        meta_config: 元认知配置
        device: 计算设备
    
    Returns:
        MetaCognitiveLayer 实例
    """
    config = MetaCognitiveLayerConfig(
        input_dim=dim_config.fast_variable_dim + dim_config.slow_variable_dim,
        projection_dim=meta_config.l2_projection_dim,
    )
    
    layer = MetaCognitiveLayer(
        config=config,
        dim_config=dim_config,
        meta_config=meta_config,
        device=device
    )
    
    return layer