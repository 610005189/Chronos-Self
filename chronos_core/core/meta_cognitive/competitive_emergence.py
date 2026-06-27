"""
竞争性稳态模块 - Competitive Stability Module
================================================

实现基于多指标竞争的稳态判定机制，通过动力学与行为学指标的
退火竞争收敛过程，检测系统稳态的产生、维持与瓦解。

核心功能：
- 多维度稳态指标计算（3个动力学 + 3个行为学）
- 退火竞争收敛算法
- 迟滞状态机（防止频繁切换）
- 稳态状态监测与统计

动力学指标：
- 状态熵（State Entropy）：衡量状态分布的不确定性
- 李雅普诺夫指数（Lyapunov Exponent）：衡量轨迹发散速率
- 吸引子维数（Attractor Dimension）：衡量相空间复杂度

行为学指标：
- 信息整合度（Information Integration）：衡量多通道信息融合程度
- 响应多样性（Response Diversity）：衡量输出模式的丰富度
- 适应性（Adaptability）：衡量对新输入的适应能力

稳态状态：
- LATENT：潜在状态（未达到稳态条件）
- EMERGING：形成中（指标上升期）
- STABLE：稳定状态（指标维持在高位）
- DISRUPTING：瓦解中（指标下降期）
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class EmergenceState(Enum):
    """稳态状态枚举"""
    
    LATENT = "latent"          # 潜在状态（未达到稳态）
    EMERGING = "emerging"      # 形成中（上升期）
    STABLE = "stable"          # 稳定状态
    DISRUPTING = "disrupting"  # 瓦解中（下降期）


@dataclass
class EmergenceConfig:
    """竞争性稳态配置"""
    
    # 启用开关
    enable_emergence: bool = False             # 是否启用竞争性涌现（默认关闭）
    
    # 计算窗口参数
    calculation_window: int = 200              # 指标计算窗口大小
    calculation_interval: int = 10             # 计算间隔步数
    history_max_length: int = 5000             # 历史数据最大长度
    
    # 动力学指标权重
    state_entropy_weight: float = 0.30         # 状态熵权重
    lyapunov_weight: float = 0.25              # 李雅普诺夫指数权重
    attractor_dim_weight: float = 0.20         # 吸引子维数权重
    
    # 行为学指标权重
    info_integration_weight: float = 0.15      # 信息整合度权重
    response_diversity_weight: float = 0.10    # 响应多样性权重
    adaptability_weight: float = 0.10          # 适应性权重
    
    # 涌现阈值
    emergence_threshold: float = 0.65          # 涌现判定阈值（综合得分）
    stable_threshold: float = 0.75             # 稳定涌现阈值
    disruption_threshold: float = 0.40         # 瓦解判定阈值
    
    # 迟滞参数
    hysteresis_margin: float = 0.10            # 迟滞边距
    min_state_duration: int = 50               # 最小状态持续步数
    
    # 退火竞争参数
    annealing_initial_temp: float = 2.0        # 初始温度
    annealing_cooling_rate: float = 0.995      # 冷却速率
    annealing_min_temp: float = 0.1            # 最低温度
    competition_iterations: int = 10           # 每轮竞争迭代次数
    
    # 状态熵参数
    entropy_bins: int = 50                     # 熵计算分箱数
    entropy_normalization: float = 1.0         # 熵归一化因子
    
    # 李雅普诺夫参数
    lyapunov_perturbation_scale: float = 1e-5  # 扰动尺度
    lyapunov_max_value: float = 0.5            # 李雅普诺夫指数最大值（用于归一化）
    
    # 吸引子维数参数
    attractor_dim_embedding_dim: int = 3       # 嵌入维数
    attractor_dim_max_value: float = 10.0      # 吸引子维数最大值（归一化）
    
    # 设备
    device: str = "cpu"


@dataclass
class EmergenceIndicators:
    """涌现指标数据"""
    
    # 动力学指标
    state_entropy: float = 0.0                 # 状态熵
    state_entropy_normalized: float = 0.0      # 归一化状态熵
    lyapunov_exponent: float = 0.0             # 李雅普诺夫指数
    lyapunov_normalized: float = 0.0           # 归一化李雅普诺夫指数
    attractor_dimension: float = 0.0           # 吸引子维数
    attractor_dim_normalized: float = 0.0      # 归一化吸引子维数
    
    # 行为学指标
    info_integration: float = 0.0              # 信息整合度
    response_diversity: float = 0.0            # 响应多样性
    adaptability: float = 0.0                  # 适应性
    
    # 综合得分
    dynamics_score: float = 0.0                # 动力学综合得分
    behavioral_score: float = 0.0              # 行为学综合得分
    composite_score: float = 0.0               # 综合涌现得分
    
    # 退火竞争信息
    annealing_temperature: float = 0.0         # 当前退火温度
    competition_converged: bool = False        # 竞争是否收敛
    
    # 时间戳
    timestamp: float = 0.0
    step_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "dynamics": {
                "state_entropy": self.state_entropy,
                "state_entropy_normalized": self.state_entropy_normalized,
                "lyapunov_exponent": self.lyapunov_exponent,
                "lyapunov_normalized": self.lyapunov_normalized,
                "attractor_dimension": self.attractor_dimension,
                "attractor_dim_normalized": self.attractor_dim_normalized,
                "dynamics_score": self.dynamics_score,
            },
            "behavioral": {
                "info_integration": self.info_integration,
                "response_diversity": self.response_diversity,
                "adaptability": self.adaptability,
                "behavioral_score": self.behavioral_score,
            },
            "composite": {
                "composite_score": self.composite_score,
                "annealing_temperature": self.annealing_temperature,
                "competition_converged": self.competition_converged,
            },
            "timestamp": self.timestamp,
            "step_count": self.step_count,
        }


@dataclass
class EmergenceStatistics:
    """涌现统计信息"""
    
    # 当前状态
    current_state: EmergenceState = EmergenceState.LATENT
    state_duration: int = 0                    # 当前状态持续步数
    
    # 得分统计
    avg_composite_score: float = 0.0           # 平均综合得分
    max_composite_score: float = 0.0           # 最大综合得分
    min_composite_score: float = 0.0           # 最小综合得分
    score_std: float = 0.0                     # 得分标准差
    
    # 状态转换统计
    total_transitions: int = 0                 # 总转换次数
    emergence_count: int = 0                   # 涌现发生次数
    stable_count: int = 0                      # 稳定涌现次数
    disruption_count: int = 0                  # 瓦解次数
    
    # 时间统计
    total_emergence_time: float = 0.0          # 总涌现时间（步数）
    avg_emergence_duration: float = 0.0        # 平均涌现持续时间
    max_emergence_duration: float = 0.0        # 最大涌现持续时间
    
    # 指标贡献度
    indicator_contributions: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "current_state": self.current_state.value,
            "state_duration": self.state_duration,
            "scores": {
                "avg": self.avg_composite_score,
                "max": self.max_composite_score,
                "min": self.min_composite_score,
                "std": self.score_std,
            },
            "transitions": {
                "total": self.total_transitions,
                "emergence_count": self.emergence_count,
                "stable_count": self.stable_count,
                "disruption_count": self.disruption_count,
            },
            "timing": {
                "total_emergence_time": self.total_emergence_time,
                "avg_emergence_duration": self.avg_emergence_duration,
                "max_emergence_duration": self.max_emergence_duration,
            },
            "indicator_contributions": self.indicator_contributions,
        }


class CompetitiveEmergence:
    """
    竞争性涌现判定器
    
    通过多指标竞争的退火收敛过程，判定系统涌现状态。
    
    核心算法：
    1. 计算6个涌现指标（3动力学 + 3行为学）
    2. 退火竞争：指标间通过模拟退火进行权重竞争
    3. 综合判定：基于加权得分和迟滞机制判定状态
    4. 状态机：LATENT → EMERGING → STABLE → DISRUPTING → LATENT
    
    使用示例：
        emergence = CompetitiveEmergence(config=EmergenceConfig())
        emergence.initialize()
        
        for step in range(num_steps):
            indicators = emergence.update(state_vector, response_vector)
            state = emergence.get_current_state()
    """
    
    def __init__(
        self,
        config: Optional[EmergenceConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化竞争性涌现判定器
        
        Args:
            config: 涌现配置
            device: 计算设备
        """
        self.config = config or EmergenceConfig()
        self.device = device or self.config.device
        
        # 状态历史缓存
        self._state_history: deque = deque(maxlen=self.config.history_max_length)
        self._response_history: deque = deque(maxlen=self.config.history_max_length)
        
        # 指标历史
        self._indicator_history: deque = deque(maxlen=self.config.history_max_length)
        self._score_history: deque = deque(maxlen=self.config.history_max_length)
        
        # 状态机
        self._current_state: EmergenceState = EmergenceState.LATENT
        self._state_entry_step: int = 0
        self._step_count: int = 0
        
        # 退火温度
        self._current_temp: float = self.config.annealing_initial_temp
        
        # 统计信息
        self._statistics = EmergenceStatistics()
        self._score_sum: float = 0.0
        self._score_sq_sum: float = 0.0
        
        # 涌现周期跟踪
        self._emergence_start_step: Optional[int] = None
        self._emergence_durations: List[int] = []
        
        # 当前指标
        self._current_indicators: Optional[EmergenceIndicators] = None
        
        logger.info(
            f"CompetitiveEmergence initialized: "
            f"enabled={self.config.enable_emergence}, "
            f"threshold={self.config.emergence_threshold}, "
            f"device={self.device}"
        )
    
    def initialize(self) -> None:
        """初始化涌现判定器"""
        self._state_history.clear()
        self._response_history.clear()
        self._indicator_history.clear()
        self._score_history.clear()
        self._current_state = EmergenceState.LATENT
        self._state_entry_step = 0
        self._step_count = 0
        self._current_temp = self.config.annealing_initial_temp
        self._statistics = EmergenceStatistics()
        self._score_sum = 0.0
        self._score_sq_sum = 0.0
        self._emergence_start_step = None
        self._emergence_durations.clear()
        self._current_indicators = None
        
        logger.info("CompetitiveEmergence initialized")
    
    def update(
        self,
        state_vector: torch.Tensor,
        response_vector: Optional[torch.Tensor] = None,
        verbose: bool = False
    ) -> EmergenceIndicators:
        """
        更新涌现判定
        
        Args:
            state_vector: 当前状态向量
            response_vector: 响应输出向量（可选）
            verbose: 是否输出详细日志
            
        Returns:
            EmergenceIndicators: 当前涌现指标
        """
        if not self.config.enable_emergence:
            return EmergenceIndicators()
        
        self._step_count += 1
        
        # 保存历史数据
        self._state_history.append(state_vector.detach().clone())
        if response_vector is not None:
            self._response_history.append(response_vector.detach().clone())
        
        # 按间隔计算指标
        if self._step_count % self.config.calculation_interval == 0:
            indicators = self._calculate_indicators(state_vector, response_vector)
            self._current_indicators = indicators
            
            # 更新历史
            self._indicator_history.append(indicators)
            self._score_history.append(indicators.composite_score)
            
            # 更新统计
            self._update_statistics(indicators)
            
            # 退火竞争
            self._annealing_competition(indicators)
            
            # 状态机更新
            self._update_state_machine(indicators)
            
            if verbose:
                logger.info(
                    f"[Step {self._step_count}] Emergence: "
                    f"state={self._current_state.value}, "
                    f"score={indicators.composite_score:.4f}, "
                    f"temp={self._current_temp:.4f}"
                )
            
            return indicators
        
        return self._current_indicators or EmergenceIndicators()
    
    def _calculate_indicators(
        self,
        state_vector: torch.Tensor,
        response_vector: Optional[torch.Tensor]
    ) -> EmergenceIndicators:
        """
        计算所有涌现指标
        
        Args:
            state_vector: 当前状态向量
            response_vector: 响应向量（可选）
            
        Returns:
            EmergenceIndicators: 涌现指标
        """
        indicators = EmergenceIndicators(
            timestamp=time.time(),
            step_count=self._step_count
        )
        
        # 确保有足够的历史数据
        if len(self._state_history) < self.config.calculation_window:
            indicators.annealing_temperature = self._current_temp
            return indicators
        
        # 获取历史状态窗口
        recent_states = list(self._state_history)[-self.config.calculation_window:]
        
        # ===== 动力学指标 =====
        
        # 1. 状态熵
        indicators.state_entropy = self._calculate_state_entropy(recent_states)
        indicators.state_entropy_normalized = min(
            1.0,
            indicators.state_entropy / (self.config.entropy_normalization * np.log2(self.config.entropy_bins))
        )
        
        # 2. 李雅普诺夫指数（近似）
        indicators.lyapunov_exponent = self._calculate_lyapunov_approx(recent_states)
        indicators.lyapunov_normalized = min(
            1.0,
            max(0.0, indicators.lyapunov_exponent / self.config.lyapunov_max_value)
        )
        
        # 3. 吸引子维数（相关维数近似）
        indicators.attractor_dimension = self._calculate_attractor_dimension(recent_states)
        indicators.attractor_dim_normalized = min(
            1.0,
            indicators.attractor_dimension / self.config.attractor_dim_max_value
        )
        
        # 动力学综合得分
        indicators.dynamics_score = (
            self.config.state_entropy_weight * indicators.state_entropy_normalized +
            self.config.lyapunov_weight * indicators.lyapunov_normalized +
            self.config.attractor_dim_weight * indicators.attractor_dim_normalized
        )
        
        # ===== 行为学指标 =====
        
        # 4. 信息整合度
        indicators.info_integration = self._calculate_info_integration(recent_states)
        
        # 5. 响应多样性
        if response_vector is not None and len(self._response_history) >= self.config.calculation_window:
            recent_responses = list(self._response_history)[-self.config.calculation_window:]
            indicators.response_diversity = self._calculate_response_diversity(recent_responses)
        else:
            indicators.response_diversity = indicators.state_entropy_normalized * 0.8
        
        # 6. 适应性
        indicators.adaptability = self._calculate_adaptability(recent_states)
        
        # 行为学综合得分
        indicators.behavioral_score = (
            self.config.info_integration_weight * indicators.info_integration +
            self.config.response_diversity_weight * indicators.response_diversity +
            self.config.adaptability_weight * indicators.adaptability
        )
        
        # 综合涌现得分（初始计算，后续通过退火竞争调整）
        total_dynamic_weight = (
            self.config.state_entropy_weight +
            self.config.lyapunov_weight +
            self.config.attractor_dim_weight
        )
        total_behavior_weight = (
            self.config.info_integration_weight +
            self.config.response_diversity_weight +
            self.config.adaptability_weight
        )
        
        indicators.composite_score = (
            indicators.dynamics_score + indicators.behavioral_score
        ) / (total_dynamic_weight + total_behavior_weight) * (total_dynamic_weight + total_behavior_weight)
        
        # 归一化到 [0, 1]
        max_possible = total_dynamic_weight + total_behavior_weight
        if max_possible > 0:
            indicators.composite_score = (
                indicators.dynamics_score + indicators.behavioral_score
            ) / max_possible
        
        indicators.annealing_temperature = self._current_temp
        
        return indicators
    
    def _calculate_state_entropy(self, state_series: List[torch.Tensor]) -> float:
        """
        计算状态熵
        
        使用直方图估计状态分布的香农熵。
        
        Args:
            state_series: 状态向量序列
            
        Returns:
            状态熵值
        """
        if len(state_series) < 2:
            return 0.0
        
        # 计算状态范数序列
        norms = torch.stack([torch.norm(s) for s in state_series]).numpy()
        
        # 直方图估计
        hist, bin_edges = np.histogram(norms, bins=self.config.entropy_bins, density=True)
        
        # 避免零概率
        hist = hist + 1e-10
        hist = hist / hist.sum()
        
        # 香农熵
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return float(entropy)
    
    def _calculate_lyapunov_approx(self, state_series: List[torch.Tensor]) -> float:
        """
        近似计算最大李雅普诺夫指数
        
        使用状态轨迹的发散速率近似估计。
        
        Args:
            state_series: 状态向量序列
            
        Returns:
            李雅普诺夫指数近似值
        """
        if len(state_series) < 10:
            return 0.0
        
        # 计算相邻状态间的距离变化
        distances = []
        for i in range(1, len(state_series)):
            dist = torch.norm(state_series[i] - state_series[i-1]).item()
            distances.append(dist)
        
        distances = np.array(distances)
        
        # 移除零距离
        distances = distances[distances > 1e-10]
        
        if len(distances) < 5:
            return 0.0
        
        # 使用对数距离的平均增长率近似
        log_distances = np.log(distances + 1e-10)
        
        # 线性拟合斜率
        x = np.arange(len(log_distances))
        if len(x) > 1:
            slope = np.polyfit(x, log_distances, 1)[0]
            return float(max(0.0, slope))
        
        return 0.0
    
    def _calculate_attractor_dimension(self, state_series: List[torch.Tensor]) -> float:
        """
        计算吸引子维数（相关维数近似）
        
        使用 Grassberger-Procaccia 算法的简化版本。
        
        Args:
            state_series: 状态向量序列
            
        Returns:
            吸引子维数近似值
        """
        if len(state_series) < 20:
            return 0.0
        
        # 采样部分点对以节省计算
        n_samples = min(500, len(state_series))
        indices = np.random.choice(len(state_series), n_samples, replace=False)
        sampled = [state_series[i] for i in indices]
        
        # 计算距离矩阵
        distances = []
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                dist = torch.norm(sampled[i] - sampled[j]).item()
                distances.append(dist)
        
        distances = np.array(distances)
        
        if len(distances) < 10:
            return 0.0
        
        # 相关维数估计：log(C(r)) vs log(r) 的斜率
        sorted_dists = np.sort(distances)
        n_points = len(sorted_dists)
        
        # 在多个半径下计算相关和
        radii = np.logspace(
            np.log10(max(sorted_dists[0], 1e-8)),
            np.log10(sorted_dists[-1]),
            20
        )
        
        correlations = []
        for r in radii:
            C_r = np.sum(sorted_dists < r) / n_points
            correlations.append(max(C_r, 1e-10))
        
        # 线性回归估计维数
        log_r = np.log(radii)
        log_C = np.log(correlations)
        
        # 取中间区域进行拟合
        mid_start = len(log_r) // 4
        mid_end = 3 * len(log_r) // 4
        
        if mid_end - mid_start > 2:
            slope = np.polyfit(
                log_r[mid_start:mid_end],
                log_C[mid_start:mid_end],
                1
            )[0]
            return float(max(0.0, slope))
        
        return 0.0
    
    def _calculate_info_integration(self, state_series: List[torch.Tensor]) -> float:
        """
        计算信息整合度
        
        衡量状态不同部分之间的互信息程度。
        
        Args:
            state_series: 状态向量序列
            
        Returns:
            信息整合度 [0, 1]
        """
        if len(state_series) < 10:
            return 0.0
        
        # 将状态向量分成两半
        state_matrix = torch.stack(state_series)
        dim = state_matrix.shape[1]
        half_dim = dim // 2
        
        part1 = state_matrix[:, :half_dim]
        part2 = state_matrix[:, half_dim:half_dim * 2]
        
        # 计算协方差矩阵
        cov1 = torch.cov(part1.T)
        cov2 = torch.cov(part2.T)
        cov_joint = torch.cov(state_matrix[:, :half_dim * 2].T)
        
        # 使用行列式比近似互信息
        det1 = torch.det(cov1 + torch.eye(half_dim) * 1e-6).item()
        det2 = torch.det(cov2 + torch.eye(half_dim) * 1e-6).item()
        det_joint = torch.det(cov_joint + torch.eye(half_dim * 2) * 1e-6).item()
        
        if det1 > 0 and det2 > 0 and det_joint > 0:
            # 互信息近似：0.5 * log(det1 * det2 / det_joint)
            mi = 0.5 * np.log(det1 * det2 / max(det_joint, 1e-10))
            # 归一化
            normalized = 1.0 - np.exp(-abs(mi) / (half_dim * 0.5))
            return float(min(1.0, max(0.0, normalized)))
        
        return 0.0
    
    def _calculate_response_diversity(self, response_series: List[torch.Tensor]) -> float:
        """
        计算响应多样性
        
        衡量响应模式的丰富度和差异性。
        
        Args:
            response_series: 响应向量序列
            
        Returns:
            响应多样性 [0, 1]
        """
        if len(response_series) < 5:
            return 0.0
        
        # 计算响应向量间的平均余弦距离
        responses = torch.stack(response_series)
        
        # 归一化
        norms = torch.norm(responses, dim=1, keepdim=True)
        normalized = responses / (norms + 1e-10)
        
        # 计算余弦相似度矩阵
        similarity_matrix = torch.mm(normalized, normalized.T)
        
        # 平均相似度（取上三角）
        n = len(responses)
        total_sim = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_sim += similarity_matrix[i, j].item()
                count += 1
        
        if count > 0:
            avg_sim = total_sim / count
            # 多样性 = 1 - 平均相似度
            diversity = 1.0 - avg_sim
            return float(min(1.0, max(0.0, diversity)))
        
        return 0.0
    
    def _calculate_adaptability(self, state_series: List[torch.Tensor]) -> float:
        """
        计算适应性
        
        衡量系统对变化的响应速度和调整能力。
        
        Args:
            state_series: 状态向量序列
            
        Returns:
            适应性 [0, 1]
        """
        if len(state_series) < 20:
            return 0.0
        
        # 计算状态变化率
        state_matrix = torch.stack(state_series)
        diffs = torch.diff(state_matrix, dim=0)
        change_rates = torch.norm(diffs, dim=1).numpy()
        
        if len(change_rates) < 10:
            return 0.0
        
        # 适应性：变化率的变异系数（标准差/均值）
        mean_rate = np.mean(change_rates)
        std_rate = np.std(change_rates)
        
        if mean_rate > 1e-10:
            cv = std_rate / mean_rate
            # 归一化：合适的变异系数表示良好的适应性
            # 太小（僵化）或太大（不稳定）都不好
            # 使用高斯函数映射
            optimal_cv = 0.5
            adaptability = np.exp(-((cv - optimal_cv) ** 2) / (2 * 0.5 ** 2))
            return float(min(1.0, max(0.0, adaptability)))
        
        return 0.0
    
    def _annealing_competition(self, indicators: EmergenceIndicators) -> None:
        """
        退火竞争收敛
        
        各指标通过模拟退火过程竞争权重，最终收敛到稳定的综合得分。
        
        Args:
            indicators: 当前涌现指标
        """
        if self._current_temp <= self.config.annealing_min_temp:
            indicators.competition_converged = True
            return
        
        # 指标值列表（归一化后）
        indicator_values = np.array([
            indicators.state_entropy_normalized,
            indicators.lyapunov_normalized,
            indicators.attractor_dim_normalized,
            indicators.info_integration,
            indicators.response_diversity,
            indicators.adaptability,
        ])
        
        # 基础权重
        base_weights = np.array([
            self.config.state_entropy_weight,
            self.config.lyapunov_weight,
            self.config.attractor_dim_weight,
            self.config.info_integration_weight,
            self.config.response_diversity_weight,
            self.config.adaptability_weight,
        ])
        
        # 竞争过程：高指标值获得更多权重
        current_weights = base_weights.copy()
        
        for _ in range(self.config.competition_iterations):
            # 计算每个指标的竞争力
            competitiveness = indicator_values * current_weights
            
            # 玻尔兹曼分布选择
            if self._current_temp > 1e-6:
                boltzmann = np.exp(competitiveness / self._current_temp)
                boltzmann = boltzmann / boltzmann.sum()
                
                # 权重向高竞争力指标倾斜
                learning_rate = 0.1
                current_weights = (
                    (1 - learning_rate) * current_weights +
                    learning_rate * boltzmann * current_weights.sum()
                )
                
                # 归一化
                current_weights = current_weights / current_weights.sum() * base_weights.sum()
        
        # 使用竞争后的权重重新计算综合得分
        dynamics_indices = [0, 1, 2]
        behavioral_indices = [3, 4, 5]
        
        dynamics_score = sum(indicator_values[i] * current_weights[i] for i in dynamics_indices)
        behavioral_score = sum(indicator_values[i] * current_weights[i] for i in behavioral_indices)
        
        total_weight = current_weights.sum()
        if total_weight > 0:
            indicators.composite_score = (dynamics_score + behavioral_score) / total_weight
        
        # 冷却
        self._current_temp = max(
            self.config.annealing_min_temp,
            self._current_temp * self.config.annealing_cooling_rate
        )
        
        # 判断是否收敛
        if self._current_temp <= self.config.annealing_min_temp:
            indicators.competition_converged = True
        
        indicators.annealing_temperature = self._current_temp
    
    def _update_state_machine(self, indicators: EmergenceIndicators) -> None:
        """
        更新状态机（带迟滞机制）
        
        状态转换图：
        LATENT → EMERGING：得分超过涌现阈值 + 最小持续时间
        EMERGING → STABLE：得分超过稳定阈值 + 最小持续时间
        STABLE → DISRUPTING：得分低于瓦解阈值 + 最小持续时间
        DISRUPTING → LATENT：得分低于瓦解阈值 + 持续下降 + 最小持续时间
        
        Args:
            indicators: 当前涌现指标
        """
        score = indicators.composite_score
        prev_state = self._current_state
        state_changed = False
        
        # 检查最小状态持续时间（迟滞机制）
        in_current_state_steps = self._step_count - self._state_entry_step
        if in_current_state_steps < self.config.min_state_duration:
            # 维持当前状态
            self._statistics.state_duration = in_current_state_steps
            return
        
        # 计算迟滞阈值
        hysteresis = self.config.hysteresis_margin
        
        # 状态转换逻辑
        if self._current_state == EmergenceState.LATENT:
            # 潜在 → 涌现中
            if score >= self.config.emergence_threshold - hysteresis * 0.5:
                self._transition_to(EmergenceState.EMERGING)
                state_changed = True
        
        elif self._current_state == EmergenceState.EMERGING:
            if score >= self.config.stable_threshold:
                # 涌现中 → 稳定
                self._transition_to(EmergenceState.STABLE)
                state_changed = True
            elif score < self.config.emergence_threshold - hysteresis:
                # 涌现中 → 潜在（回落）
                self._transition_to(EmergenceState.LATENT)
                state_changed = True
        
        elif self._current_state == EmergenceState.STABLE:
            if score < self.config.disruption_threshold:
                # 稳定 → 瓦解中
                self._transition_to(EmergenceState.DISRUPTING)
                state_changed = True
        
        elif self._current_state == EmergenceState.DISRUPTING:
            if score >= self.config.emergence_threshold:
                # 瓦解中 → 涌现中（恢复）
                self._transition_to(EmergenceState.EMERGING)
                state_changed = True
            elif score < self.config.disruption_threshold - hysteresis:
                # 瓦解中 → 潜在（完全瓦解）
                self._transition_to(EmergenceState.LATENT)
                state_changed = True
        
        self._statistics.state_duration = self._step_count - self._state_entry_step
    
    def _transition_to(self, new_state: EmergenceState) -> None:
        """
        执行状态转换
        
        Args:
            new_state: 新状态
        """
        old_state = self._current_state
        self._current_state = new_state
        self._state_entry_step = self._step_count
        
        self._statistics.total_transitions += 1
        
        # 更新状态统计
        if new_state == EmergenceState.EMERGING:
            self._statistics.emergence_count += 1
            self._emergence_start_step = self._step_count
        elif new_state == EmergenceState.STABLE:
            self._statistics.stable_count += 1
        elif new_state == EmergenceState.DISRUPTING:
            self._statistics.disruption_count += 1
        elif new_state == EmergenceState.LATENT:
            # 结束一个涌现周期
            if self._emergence_start_step is not None:
                duration = self._step_count - self._emergence_start_step
                self._emergence_durations.append(duration)
                self._statistics.total_emergence_time += duration
                self._statistics.max_emergence_duration = max(
                    self._statistics.max_emergence_duration,
                    duration
                )
                if self._statistics.emergence_count > 0:
                    self._statistics.avg_emergence_duration = (
                        self._statistics.total_emergence_time /
                        self._statistics.emergence_count
                    )
                self._emergence_start_step = None
        
        logger.debug(
            f"State transition: {old_state.value} → {new_state.value} "
            f"(step={self._step_count})"
        )
    
    def _update_statistics(self, indicators: EmergenceIndicators) -> None:
        """
        更新统计信息
        
        Args:
            indicators: 当前涌现指标
        """
        score = indicators.composite_score
        
        # 得分统计
        self._score_sum += score
        self._score_sq_sum += score * score
        
        n = len(self._score_history)
        if n > 0:
            self._statistics.avg_composite_score = self._score_sum / n
            self._statistics.max_composite_score = max(
                self._statistics.max_composite_score,
                score
            )
            self._statistics.min_composite_score = min(
                self._statistics.min_composite_score if self._statistics.min_composite_score > 0 else score,
                score
            )
            if n > 1:
                variance = (self._score_sq_sum / n) - (self._score_sum / n) ** 2
                self._statistics.score_std = np.sqrt(max(0.0, variance))
        
        # 指标贡献度
        self._statistics.indicator_contributions = {
            "state_entropy": indicators.state_entropy_normalized * self.config.state_entropy_weight,
            "lyapunov": indicators.lyapunov_normalized * self.config.lyapunov_weight,
            "attractor_dim": indicators.attractor_dim_normalized * self.config.attractor_dim_weight,
            "info_integration": indicators.info_integration * self.config.info_integration_weight,
            "response_diversity": indicators.response_diversity * self.config.response_diversity_weight,
            "adaptability": indicators.adaptability * self.config.adaptability_weight,
        }
    
    def get_current_state(self) -> EmergenceState:
        """获取当前涌现状态"""
        return self._current_state
    
    def get_current_indicators(self) -> Optional[EmergenceIndicators]:
        """获取当前涌现指标"""
        return self._current_indicators
    
    def get_statistics(self) -> EmergenceStatistics:
        """获取统计信息"""
        self._statistics.current_state = self._current_state
        self._statistics.state_duration = self._step_count - self._state_entry_step
        return self._statistics
    
    def get_emergence_report(self) -> Dict[str, Any]:
        """获取涌现报告"""
        stats = self.get_statistics()
        indicators = self.get_current_indicators()
        
        report = {
            "enabled": self.config.enable_emergence,
            "current_state": self._current_state.value,
            "state_duration_steps": self._step_count - self._state_entry_step,
            "step_count": self._step_count,
            "annealing_temperature": self._current_temp,
            "indicators": indicators.to_dict() if indicators else {},
            "statistics": stats.to_dict(),
        }
        
        return report
    
    def is_emerging(self) -> bool:
        """判断是否处于涌现状态（涌现中或稳定）"""
        return self._current_state in (EmergenceState.EMERGING, EmergenceState.STABLE)
    
    def is_stable(self) -> bool:
        """判断是否处于稳定涌现状态"""
        return self._current_state == EmergenceState.STABLE
    
    def reset(self) -> None:
        """重置涌现判定器"""
        self.initialize()
        logger.info("CompetitiveEmergence reset")
    
    def __repr__(self) -> str:
        return (
            f"CompetitiveEmergence("
            f"state={self._current_state.value}, "
            f"step={self._step_count}, "
            f"temp={self._current_temp:.4f})"
        )
