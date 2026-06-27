"""
递归状态监控模块 - Recursive State Monitoring Module
====================================================

实现完整的 L0-L1-L2 三层递归状态监控模块。

核心功能：
- 整合 L0、L1、L2 三层结构
- 实现信息流：外部输入 → L0 → L1 → L2 → 调控信号 → L1
- 实现调控循环：L2 监测 L1 状态并输出调控
- 提供消融测试接口
- 提供状态监测接口

层级结构：
- L0: 感知层（Perception Layer）- 仅处理感知输入
- L1: 状态层（State Layer）- 完整状态积分，包含系统状态
- L2: 监控层（Monitoring Layer）- 高阶调控，物理隔离于 L0

信息流：
1. 外部输入 → L0（感知处理）
2. L0 → L1（状态更新）
3. L1 → L2（状态监测）
4. L2 → 调控信号 → L1（参数调整）

调控循环：
- L2 监测 L1 状态（定期，如每10步）
- L2 输出调控信号
- L1 根据调控调整参数
- L1 状态演化（使用 IntegrationEngine）
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field
import logging
import time
import warnings

from chronos_core.core.meta_cognitive.perception_layer import (
    PerceptionLayer,
    PerceptionLayerConfig,
)
from chronos_core.core.meta_cognitive.self_state_layer import (
    SelfStateLayer,
    SelfStateLayerConfig,
)
from chronos_core.core.meta_cognitive.meta_cognitive_layer import (
    MetaCognitiveLayer,
    MetaCognitiveLayerConfig,
)
from chronos_core.core.meta_cognitive.meta_cognitive_manager import (
    MetaCognitiveManager,
    MetaCognitiveManagerConfig,
)
from chronos_core.core.meta_cognitive.curiosity_engine import (
    CuriosityEngine,
    CuriosityConfig,
    CuriosityMetrics,
)
from chronos_core.core.meta_cognitive.competitive_emergence import (
    CompetitiveEmergence,
    EmergenceConfig,
    EmergenceState,
    EmergenceIndicators,
    EmergenceStatistics,
)
from chronos_core.utils.config import (
    DimensionalityConfig,
    MetaCognitiveConfig,
    MemoryTemporalConfig,
    ChronosConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class MetaCognitiveConfig:
    """递归状态监控模块配置"""
    
    # 系统层级配置
    use_l0: bool = True                     # 是否使用 L0 感知层
    use_l1: bool = True                     # 是否使用 L1 自我状态层
    use_l2: bool = True                     # 是否使用 L2 元认知层
    
    # 信息流配置
    perception_to_state_enabled: bool = True     # L0 → L1 数据传递
    state_to_meta_enabled: bool = True           # L1 → L2 状态发送
    meta_to_state_enabled: bool = True           # L2 → L1 调控信号
    
    # 调控循环配置
    regulation_cycle_interval: int = 10          # 调控循环间隔步数
    regulation_enabled: bool = True              # 是否启用调控
    
    # 消融测试配置
    ablation_test_enabled: bool = True           # 是否启用消融测试
    ablation_test_interval: int = 100            # 消融测试间隔步数
    ablation_duration: int = 50                  # 消融测试持续时间
    
    # 状态监测配置
    state_monitoring_enabled: bool = True        # 是否启用状态监测
    monitoring_log_interval: int = 100           # 监测日志间隔步数
    
    # 竞争性涌现配置
    enable_emergence: bool = False               # 是否启用竞争性涌现（默认关闭）
    emergence_calculation_interval: int = 10     # 涌现计算间隔步数
    
    # 设备参数
    device: str = "cpu"


class MetaCognitive(nn.Module):
    """
    递归状态监控模块
    
    整合 L0、L1、L2 三层结构，实现完整的状态监控调控功能。
    
    功能：
    - 整合 L0、L1、L2 三层结构
    - 实现信息流
    - 实现调控循环
    - 提供消融测试接口
    - 提供状态监测接口
    
    特性：
    - L0 仅处理感知输入
    - L1 包含完整系统状态
    - L2 高阶调控，物理隔离于 L0
    - 状态监控截断机制
    """
    
    def __init__(
        self,
        config: Optional["MetaCognitiveConfig"] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        meta_cognitive_config: Optional[MetaCognitiveConfig] = None,
        memory_config: Optional[MemoryTemporalConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化元认知模块
        
        Args:
            config: 模块配置
            dim_config: 维度配置
            meta_cognitive_config: 元认知配置（全局配置中的）
            memory_config: 内存配置
            global_config: 全局配置
            device: 计算设备
        """
        super().__init__()
        
        # 合并配置
        self.config = config or MetaCognitiveConfig()
        self.global_config = global_config
        
        # 从 global_config 提取配置（如果提供了）
        if global_config:
            self.dim_config = dim_config or global_config.dim
            self.meta_config = meta_cognitive_config or global_config.meta_cognitive
            self.memory_config = memory_config or global_config.memory_temporal
        else:
            self.dim_config = dim_config or DimensionalityConfig()
            self.meta_config = meta_cognitive_config or MetaCognitiveConfig()
            self.memory_config = memory_config or MemoryTemporalConfig()
        
        self.device = device or self.config.device
        
        # 好奇心引擎
        self.curiosity_engine: Optional[CuriosityEngine] = None
        self._curiosity_config: Optional[CuriosityConfig] = None
        
        # 竞争性涌现
        self.emergence_detector: Optional[CompetitiveEmergence] = None
        self._emergence_config: Optional[EmergenceConfig] = None
        self._last_emergence_indicators: Optional[EmergenceIndicators] = None
        
        # 系统状态缓存
        self._current_step: int = 0
        self._l0_output_cache: Optional[torch.Tensor] = None
        self._l1_state_cache: Optional[Any] = None
        self._l2_control_cache: Optional[torch.Tensor] = None
        self._last_curiosity_metrics: Optional[CuriosityMetrics] = None
        
        # 初始化各层
        self._initialize_layers()
        
        # 初始化元认知管理器
        self._initialize_manager()
        
        # 初始化好奇心引擎
        self._initialize_curiosity_engine()
        
        # 初始化竞争性涌现
        self._initialize_emergence()
        
        # 消融测试状态
        self._ablation_active: bool = False
        self._ablation_step_count: int = 0
        
        # 统计信息
        self._stats = {
            "total_steps": 0,
            "l0_updates": 0,
            "l1_updates": 0,
            "l2_updates": 0,
            "regulation_cycles": 0,
            "ablation_tests": 0,
        }
        
        self.to(self.device)
        
        logger.info(
            f"MetaCognitiveSystem initialized: "
            f"use_l0={self.config.use_l0}, "
            f"use_l1={self.config.use_l1}, "
            f"use_l2={self.config.use_l2}, "
            f"device={self.device}"
        )
    
    def _initialize_layers(self):
        """初始化各层"""
        # L0 感知层
        if self.config.use_l0:
            self.l0_layer = PerceptionLayer(
                dim_config=self.dim_config,
                meta_config=self.meta_config,
                device=self.device
            )
            logger.info("L0 PerceptionLayer initialized")
        else:
            self.l0_layer = None
            logger.info("L0 PerceptionLayer disabled")
        
        # L1 自我状态层
        if self.config.use_l1:
            self.l1_layer = SelfStateLayer(
                dim_config=self.dim_config,
                meta_config=self.meta_config,
                memory_config=self.memory_config,
                global_config=self.global_config,
                device=self.device
            )
            logger.info("L1 SelfStateLayer initialized")
        else:
            self.l1_layer = None
            logger.info("L1 SelfStateLayer disabled")
        
        # L2 元认知层
        if self.config.use_l2:
            self.l2_layer = MetaCognitiveLayer(
                dim_config=self.dim_config,
                meta_config=self.meta_config,
                device=self.device
            )
            logger.info("L2 MetaCognitiveLayer initialized")
        else:
            self.l2_layer = None
            logger.info("L2 MetaCognitiveLayer disabled")
    
    def _initialize_manager(self):
        """初始化元认知管理器"""
        if self.config.use_l2:
            self.manager = MetaCognitiveManager(
                meta_config=self.meta_config,
                control_signal_dim=self.l2_layer.config.control_output_dim,
                device=self.device
            )
            logger.info("MetaCognitiveManager initialized")
        else:
            self.manager = None
            logger.info("MetaCognitiveManager disabled (L2 not available)")
    
    def _initialize_curiosity_engine(self):
        """初始化好奇心引擎"""
        if hasattr(self.meta_config, 'enable_curiosity') and self.meta_config.enable_curiosity:
            curiosity_config = CuriosityConfig(
                enabled=True,
                novelty_weight=getattr(self.meta_config, 'curiosity_novelty_weight', 0.4),
                complexity_weight=getattr(self.meta_config, 'curiosity_complexity_weight', 0.3),
                uncertainty_weight=getattr(self.meta_config, 'curiosity_uncertainty_weight', 0.3),
                exploration_rate=getattr(self.meta_config, 'curiosity_exploration_rate', 0.1),
                exploration_decay=getattr(self.meta_config, 'curiosity_exploration_decay', 0.995),
                min_exploration_rate=getattr(self.meta_config, 'curiosity_min_exploration_rate', 0.01),
                curiosity_decay_rate=getattr(self.meta_config, 'curiosity_decay_rate', 0.9),
                history_window_size=getattr(self.meta_config, 'curiosity_history_window', 100),
                device=self.device
            )
            self._curiosity_config = curiosity_config
            self.curiosity_engine = CuriosityEngine(
                config=curiosity_config,
                device=self.device
            )
            logger.info("CuriosityEngine initialized and enabled")
        else:
            self.curiosity_engine = None
            logger.info("CuriosityEngine disabled")
    
    def _initialize_emergence(self):
        """初始化竞争性涌现判定器"""
        if self.meta_config.enable_emergence:
            emergence_config = EmergenceConfig(
                enable_emergence=True,
                calculation_window=getattr(self.meta_config, 'emergence_calculation_window', 200),
                calculation_interval=self.config.emergence_calculation_interval,
                state_entropy_weight=getattr(self.meta_config, 'emergence_state_entropy_weight', 0.30),
                lyapunov_weight=getattr(self.meta_config, 'emergence_lyapunov_weight', 0.25),
                attractor_dim_weight=getattr(self.meta_config, 'emergence_attractor_dim_weight', 0.20),
                info_integration_weight=getattr(self.meta_config, 'emergence_info_integration_weight', 0.15),
                response_diversity_weight=getattr(self.meta_config, 'emergence_response_diversity_weight', 0.10),
                adaptability_weight=getattr(self.meta_config, 'emergence_adaptability_weight', 0.10),
                emergence_threshold=getattr(self.meta_config, 'emergence_threshold', 0.65),
                stable_threshold=getattr(self.meta_config, 'emergence_stable_threshold', 0.75),
                disruption_threshold=getattr(self.meta_config, 'emergence_disruption_threshold', 0.40),
                hysteresis_margin=getattr(self.meta_config, 'emergence_hysteresis_margin', 0.10),
                min_state_duration=getattr(self.meta_config, 'emergence_min_state_duration', 50),
                annealing_initial_temp=getattr(self.meta_config, 'emergence_annealing_initial_temp', 2.0),
                annealing_cooling_rate=getattr(self.meta_config, 'emergence_annealing_cooling_rate', 0.995),
                annealing_min_temp=getattr(self.meta_config, 'emergence_annealing_min_temp', 0.1),
                device=self.device
            )
            self._emergence_config = emergence_config
            self.emergence_detector = CompetitiveEmergence(
                config=emergence_config,
                device=self.device
            )
            self.emergence_detector.initialize()
            logger.info("CompetitiveEmergence initialized and enabled")
        else:
            self.emergence_detector = None
            self._emergence_config = None
            logger.info("CompetitiveEmergence disabled")
    
    def compute_curiosity(self, input_vector: torch.Tensor, prediction_error: Optional[float] = None) -> Optional[CuriosityMetrics]:
        """
        计算输入的好奇心指标
        
        Args:
            input_vector: 输入向量
            prediction_error: 预测误差（可选）
            
        Returns:
            好奇心指标（如果好奇心引擎启用）
        """
        if self.curiosity_engine is None:
            return None
        
        metrics = self.curiosity_engine.compute_curiosity(input_vector, prediction_error)
        self._last_curiosity_metrics = metrics
        return metrics
    
    def select_input_with_curiosity(
        self,
        input_candidates: List[Tuple[str, torch.Tensor]],
        prediction_errors: Optional[Dict[str, float]] = None,
        epsilon_greedy: bool = True
    ) -> Optional[Tuple[str, torch.Tensor, CuriosityMetrics]]:
        """
        使用好奇心驱动的策略选择输入
        
        Args:
            input_candidates: 候选输入列表 [(input_id, input_data), ...]
            prediction_errors: 各输入的预测误差字典
            epsilon_greedy: 是否使用 epsilon-greedy 策略
            
        Returns:
            (selected_id, selected_data, curiosity_metrics) 或 None
        """
        if self.curiosity_engine is None:
            return None
        
        return self.curiosity_engine.select_input(
            input_candidates=input_candidates,
            prediction_errors=prediction_errors,
            epsilon_greedy=epsilon_greedy
        )
    
    def get_curiosity_statistics(self) -> Optional[Dict[str, Any]]:
        """
        获取好奇心引擎统计信息
        
        Returns:
            统计信息字典或 None
        """
        if self.curiosity_engine is None:
            return None
        
        return self.curiosity_engine.get_statistics()
    
    def forward(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor,
        dt: Optional[float] = None,
        prediction_error: Optional[float] = None,
        emotion_signal: Optional[torch.Tensor] = None,
        apply_regulation: bool = True
    ) -> Dict[str, Any]:
        """
        执行完整元认知循环
        
        流程：
        1. L0 感知处理（外部输入 → L0）
        2. L0 → L1 数据传递
        3. L1 状态演化
        4. L1 → L2 状态发送
        5. L2 元认知监测
        6. L2 → L1 调控信号
        7. L1 根据调控调整参数
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
            dt: 时间步长
            prediction_error: 预测误差（可选）
            emotion_signal: 情绪信号（可选）
            apply_regulation: 是否应用调控
        
        Returns:
            系统输出字典
        """
        # 默认时间步长
        dt = dt or 0.01
        
        outputs = {}
        
        # 0. 好奇心计算（如果启用）
        if self.curiosity_engine is not None:
            combined_input = torch.cat([semantic_input, physical_input], dim=0)
            curiosity_metrics = self.compute_curiosity(combined_input, prediction_error)
            if curiosity_metrics:
                outputs["curiosity_metrics"] = curiosity_metrics
                self._last_curiosity_metrics = curiosity_metrics
                logger.debug(
                    f"Curiosity: score={curiosity_metrics.curiosity_score:.4f}, "
                    f"novelty={curiosity_metrics.novelty:.4f}, "
                    f"complexity={curiosity_metrics.complexity:.4f}"
                )
        
        # 1. L0 感知处理
        if self.l0_layer:
            l0_output = self.l0_layer(semantic_input, physical_input)
            self._l0_output_cache = l0_output.clone()
            self._stats["l0_updates"] += 1
            outputs["l0_output"] = l0_output
            
            logger.debug(f"L0 processed: output_dim={l0_output.shape[0]}")
        
        # 2. L0 → L1 数据传递
        l1_input = self._l0_output_cache if self.config.perception_to_state_enabled else None
        
        # 3. L1 状态演化
        if self.l1_layer:
            # 获取 L2 调控信号（如果存在且启用）
            l2_control = self._get_l2_control_signal(apply_regulation)
            
            # L1 状态演化
            l1_state = self.l1_layer(
                l0_perception_data=l1_input,
                l2_control_signal=l2_control,
                dt=dt
            )
            self._l1_state_cache = l1_state
            self._stats["l1_updates"] += 1
            outputs["l1_state"] = l1_state
            
            logger.debug(
                f"L1 updated: timestamp={l1_state.timestamp:.2f}, "
                f"fast_norm={l1_state.get_fast_norm():.4f}"
            )
        
        # 4. L1 → L2 状态发送
        # 5. L2 元认知监测
        if self.l2_layer and self.config.state_to_meta_enabled:
            # 检查是否应该执行调控循环
            if self._should_execute_regulation_cycle():
                # 发送 L1 状态给 L2
                l1_state_vector = self._build_l1_state_vector()
                
                # L2 元认知监测
                l2_control = self.l2_layer(
                    l1_state=l1_state_vector,
                    prediction_error=prediction_error,
                    emotion_signal=emotion_signal
                )
                self._l2_control_cache = l2_control.clone()
                self._stats["l2_updates"] += 1
                outputs["l2_control"] = l2_control
                
                logger.debug(f"L2 generated control signal: dim={l2_control.shape[0]}")
                
                # 记录调控循环
                self._stats["regulation_cycles"] += 1
        
        # 6. L2 → L1 调控信号（已经在 L1 演化中处理）
        # 7. L1 根据调控调整参数（已经在 L1 演化中处理）
        
        # 更新步数
        self._current_step += 1
        self._stats["total_steps"] += 1
        
        # 竞争性涌现检测
        if self.meta_config.enable_emergence and self.emergence_detector:
            self._update_emergence_detection(outputs)
        
        # 消融测试检查
        if self.config.ablation_test_enabled:
            self._check_ablation_test()
        
        # 状态监测
        if self.config.state_monitoring_enabled:
            self._monitor_state()
        
        return outputs
    
    def _get_l2_control_signal(self, apply_regulation: bool) -> Optional[torch.Tensor]:
        """
        获取 L2 调控信号
        
        Args:
            apply_regulation: 是否应用调控
        
        Returns:
            L2 调控信号（如果消融测试激活则返回 None）
        """
        # 如果消融测试激活，不应用调控信号
        if self._ablation_active:
            logger.debug("Ablation active - no L2 control signal")
            return None
        
        # 如果不应用调控，返回 None
        if not apply_regulation:
            return None
        
        # 如果 L2 不存在或调控未启用，返回 None
        if not self.l2_layer or not self.config.meta_to_state_enabled:
            return None
        
        # 获取 L2 调控信号
        control_signal = self.l2_layer.send_control_to_l1()
        
        # 如果存在管理器，处理调控信号
        if control_signal is not None and self.manager:
            # 处理调控信号（扰动 + 依赖权重）
            processed_signal, dependency_weight = self.manager.process_control_signal(
                control_signal,
                apply_perturbation=True
            )
            
            logger.debug(
                f"Processed L2 control signal: "
                f"dependency_weight={dependency_weight:.4f}"
            )
            
            return processed_signal
        
        return control_signal
    
    def _build_l1_state_vector(self) -> torch.Tensor:
        """
        构建 L1 状态向量
        
        合合快变量和慢变量为完整状态向量
        
        Returns:
            L1 状态向量
        """
        if not self.l1_layer or not self._l1_state_cache:
            # 创建默认状态向量
            state_vector = torch.zeros(
                self.dim_config.fast_variable_dim + self.dim_config.slow_variable_dim,
                device=self.device
            )
            return state_vector
        
        # 从 L1 获取状态
        l1_state = self.l1_layer.send_to_l2()
        
        return l1_state
    
    def _should_execute_regulation_cycle(self) -> bool:
        """
        判断是否应该执行调控循环
        
        Returns:
            是否应该执行调控循环
        """
        # 每隔 regulation_cycle_interval 步执行一次调控循环
        return self._current_step % self.config.regulation_cycle_interval == 0
    
    def _check_ablation_test(self):
        """
        检查消融测试
        
        自动触发消融测试
        """
        # 检查是否应该开始消融测试
        if not self._ablation_active and \
           self._current_step % self.config.ablation_test_interval == 0:
            # 开始消融测试
            self.start_ablation_test()
        
        # 检查是否应该结束消融测试
        if self._ablation_active and \
           self._ablation_step_count >= self.config.ablation_duration:
            # 结束消融测试
            self.end_ablation_test()
    
    def start_ablation_test(self):
        """
        开始消融测试
        
        移除 L2 调控信号
        """
        self._ablation_active = True
        self._ablation_step_count = 0
        
        # 如果管理器存在，开始消融测试
        if self.manager:
            self.manager.start_ablation_test()
        
        logger.info(
            f"Ablation test started at step {self._current_step}: "
            f"L2 control signal removed"
        )
        
        self._stats["ablation_tests"] += 1
    
    def end_ablation_test(self):
        """
        结束消融测试
        
        恢复 L2 调控信号
        """
        self._ablation_active = False
        self._ablation_step_count = 0
        
        # 如果管理器存在，结束消融测试
        if self.manager:
            self.manager.end_ablation_test()
        
        logger.info(
            f"Ablation test ended at step {self._current_step}: "
            f"L2 control signal restored"
        )
    
    def is_ablation_active(self) -> bool:
        """
        检查消融测试是否激活
        
        Returns:
            是否处于消融状态
        """
        return self._ablation_active
    
    def _update_emergence_detection(self, outputs: Dict[str, Any]):
        """
        更新竞争性涌现检测
        
        Args:
            outputs: 前向传播输出字典
        """
        if self.emergence_detector is None:
            return
        
        # 获取状态向量（优先使用 L1 状态）
        state_vector = None
        if self._l1_state_cache is not None and hasattr(self._l1_state_cache, 'E_fast'):
            state_vector = self._l1_state_cache.E_fast
        elif self._l0_output_cache is not None:
            state_vector = self._l0_output_cache
        else:
            return
        
        # 获取响应向量（L2 控制信号作为响应）
        response_vector = self._l2_control_cache
        
        # 更新涌现检测
        indicators = self.emergence_detector.update(
            state_vector=state_vector,
            response_vector=response_vector
        )
        
        self._last_emergence_indicators = indicators
        
        # 记录到输出中
        outputs["emergence_indicators"] = indicators
        outputs["emergence_state"] = self.emergence_detector.get_current_state()
        
        # 定期记录涌现状态
        if self._current_step % self.config.monitoring_log_interval == 0:
            logger.debug(
                f"Emergence at step {self._current_step}: "
                f"state={self.emergence_detector.get_current_state().value}, "
                f"score={indicators.composite_score:.4f}"
            )
    
    def get_emergence_state(self) -> EmergenceState:
        """
        获取当前涌现状态
        
        Returns:
            当前涌现状态
        """
        if self.emergence_detector:
            return self.emergence_detector.get_current_state()
        return EmergenceState.LATENT
    
    def get_emergence_indicators(self) -> Optional[EmergenceIndicators]:
        """
        获取当前涌现指标
        
        Returns:
            当前涌现指标（如果启用）
        """
        return self._last_emergence_indicators
    
    def get_emergence_statistics(self) -> Optional[EmergenceStatistics]:
        """
        获取涌现统计信息
        
        Returns:
            涌现统计信息（如果启用）
        """
        if self.emergence_detector:
            return self.emergence_detector.get_statistics()
        return None
    
    def is_emerging(self) -> bool:
        """
        判断系统是否处于涌现状态
        
        Returns:
            是否处于涌现状态
        """
        if self.emergence_detector:
            return self.emergence_detector.is_emerging()
        return False
    
    def _monitor_state(self):
        """
        监测系统状态
        
        定期记录系统状态信息
        """
        # 每隔 monitoring_log_interval 步记录状态
        if self._current_step % self.config.monitoring_log_interval == 0:
            # 记录状态信息
            state_info = self._build_state_info()
            
            logger.debug(
                f"System state at step {self._current_step}: "
                f"l0_output_norm={state_info['l0_output_norm']:.4f}, "
                f"l1_fast_norm={state_info['l1_fast_norm']:.4f}, "
                f"l2_control_norm={state_info['l2_control_norm']:.4f}"
            )
    
    def _build_state_info(self) -> Dict[str, Any]:
        """
        构建状态信息
        
        Returns:
            状态信息字典
        """
        state_info = {}
        
        # L0 状态
        if self._l0_output_cache is not None:
            state_info["l0_output_norm"] = torch.norm(self._l0_output_cache).item()
        else:
            state_info["l0_output_norm"] = 0.0
        
        # L1 状态
        if self._l1_state_cache is not None:
            state_info["l1_fast_norm"] = self._l1_state_cache.get_fast_norm()
            state_info["l1_slow_norm"] = self._l1_state_cache.get_slow_norm()
            state_info["l1_timestamp"] = self._l1_state_cache.timestamp
        else:
            state_info["l1_fast_norm"] = 0.0
            state_info["l1_slow_norm"] = 0.0
            state_info["l1_timestamp"] = 0.0
        
        # L2 状态
        if self._l2_control_cache is not None:
            state_info["l2_control_norm"] = torch.norm(self._l2_control_cache).item()
        else:
            state_info["l2_control_norm"] = 0.0
        
        # 系统状态
        state_info["total_steps"] = self._stats["total_steps"]
        state_info["ablation_active"] = self._ablation_active
        
        return state_info
    
    def test_with_l2(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        测试带 L2 调控的系统
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
            num_steps: 测试步数
        
        Returns:
            测试结果
        """
        logger.info(f"Testing system with L2 regulation for {num_steps} steps")
        
        # 确保消融测试未激活
        self._ablation_active = False
        
        # 运行测试
        results = []
        performance_metrics = []
        
        for step in range(num_steps):
            # 执行一步
            outputs = self.forward(
                semantic_input=semantic_input,
                physical_input=physical_input,
                apply_regulation=True
            )
            
            # 记录结果
            results.append(outputs)
            
            # 计算性能指标（使用状态范数作为示例）
            if outputs.get("l1_state"):
                performance_metric = outputs["l1_state"].get_fast_norm()
                performance_metrics.append(performance_metric)
        
        # 统计性能
        test_result = {
            "mode": "with_l2",
            "num_steps": num_steps,
            "results": results,
            "performance_metrics": performance_metrics,
            "mean_performance": np.mean(performance_metrics) if performance_metrics else 0.0,
            "std_performance": np.std(performance_metrics) if performance_metrics else 0.0,
        }
        
        logger.info(
            f"Test with L2 completed: "
            f"mean_performance={test_result['mean_performance']:.4f}"
        )
        
        return test_result
    
    def test_without_l2(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        测试不带 L2 调控的系统（消融测试）
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
            num_steps: 测试步数
        
        Returns:
            测试结果
        """
        logger.info(f"Testing system without L2 regulation for {num_steps} steps")
        
        # 手动激活消融
        self._ablation_active = True
        
        # 运行测试
        results = []
        performance_metrics = []
        
        for step in range(num_steps):
            # 执行一步
            outputs = self.forward(
                semantic_input=semantic_input,
                physical_input=physical_input,
                apply_regulation=False  # 不应用调控
            )
            
            # 记录结果
            results.append(outputs)
            
            # 计算性能指标
            if outputs.get("l1_state"):
                performance_metric = outputs["l1_state"].get_fast_norm()
                performance_metrics.append(performance_metric)
        
        # 结束消融
        self._ablation_active = False
        
        # 统计性能
        test_result = {
            "mode": "without_l2",
            "num_steps": num_steps,
            "results": results,
            "performance_metrics": performance_metrics,
            "mean_performance": np.mean(performance_metrics) if performance_metrics else 0.0,
            "std_performance": np.std(performance_metrics) if performance_metrics else 0.0,
        }
        
        logger.info(
            f"Test without L2 completed: "
            f"mean_performance={test_result['mean_performance']:.4f}"
        )
        
        return test_result
    
    def compare_performance(
        self,
        semantic_input: torch.Tensor,
        physical_input: torch.Tensor,
        num_steps: int = 100
    ) -> Dict[str, Any]:
        """
        比较 L2 调控的性能
        
        Args:
            semantic_input: 语义输入
            physical_input: 物理输入
            num_steps: 测试步数
        
        Returns:
            性能比较结果
        """
        logger.info(f"Comparing performance for {num_steps} steps")
        
        # 测试带 L2
        test_with_l2_result = self.test_with_l2(
            semantic_input, physical_input, num_steps
        )
        
        # 重置系统
        self.reset()
        
        # 测试不带 L2
        test_without_l2_result = self.test_without_l2(
            semantic_input, physical_input, num_steps
        )
        
        # 计算功能维持率
        with_l2_performance = test_with_l2_result["mean_performance"]
        without_l2_performance = test_without_l2_result["mean_performance"]
        
        if with_l2_performance > 0:
            retention_rate = without_l2_performance / with_l2_performance
        else:
            retention_rate = 0.0
        
        # 验证独立性
        is_valid = retention_rate > self.meta_config.l2_ablation_threshold
        
        # 构建比较结果
        comparison_result = {
            "with_l2_performance": with_l2_performance,
            "without_l2_performance": without_l2_performance,
            "retention_rate": retention_rate,
            "threshold": self.meta_config.l2_ablation_threshold,
            "is_valid": is_valid,
            "test_with_l2": test_with_l2_result,
            "test_without_l2": test_without_l2_result,
        }
        
        logger.info(
            f"Performance comparison: "
            f"with_l2={with_l2_performance:.4f}, "
            f"without_l2={without_l2_performance:.4f}, "
            f"retention_rate={retention_rate:.4f}, "
            f"is_valid={is_valid}"
        )
        
        return comparison_result
    
    def get_layer_states(self) -> Dict[str, Any]:
        """
        获取各层状态
        
        Returns:
            各层状态字典
        """
        states = {}
        
        # L0 状态
        if self.l0_layer:
            states["l0"] = {
                "output": self._l0_output_cache,
                "statistics": self.l0_layer.get_statistics(),
            }
        
        # L1 状态
        if self.l1_layer:
            states["l1"] = {
                "state": self._l1_state_cache,
                "statistics": self.l1_layer.get_statistics(),
            }
        
        # L2 状态
        if self.l2_layer:
            states["l2"] = {
                "control_signal": self._l2_control_cache,
                "statistics": self.l2_layer.get_statistics(),
            }
        
        return states
    
    def get_regulation_signals(self) -> Dict[str, Any]:
        """
        获取调控信号
        
        Returns:
            调控信号字典
        """
        signals = {}
        
        # L2 调控信号
        if self._l2_control_cache is not None:
            signals["l2_control_raw"] = self._l2_control_cache
            
            # 如果有管理器，获取处理后的信号
            if self.manager:
                processed_signal, dependency_weight = self.manager.process_control_signal(
                    self._l2_control_cache,
                    apply_perturbation=False  # 不应用扰动，仅获取依赖权重
                )
                signals["l2_control_processed"] = processed_signal
                signals["dependency_weight"] = dependency_weight
        
        return signals
    
    def monitor_cycle(self) -> Dict[str, Any]:
        """
        监测调控循环
        
        Returns:
            调控循环监测信息
        """
        cycle_info = {
            "current_step": self._current_step,
            "regulation_cycle_interval": self.config.regulation_cycle_interval,
            "next_regulation_step": (
                (self._current_step // self.config.regulation_cycle_interval + 1) *
                self.config.regulation_cycle_interval
            ),
            "ablation_active": self._ablation_active,
            "total_regulation_cycles": self._stats["regulation_cycles"],
        }
        
        # 如果 L2 存在，添加监测信息
        if self.l2_layer:
            cycle_info["l2_monitoring_interval"] = self.l2_layer.get_monitoring_interval()
            cycle_info["l2_should_monitor"] = self.l2_layer.should_monitor()
        
        # 如果管理器存在，添加管理器信息
        if self.manager:
            cycle_info["manager_stats"] = self.manager.get_statistics()
        
        return cycle_info
    
    def reset(self):
        """重置系统"""
        # 重置各层
        if self.l0_layer:
            self.l0_layer.reset()
        
        if self.l1_layer:
            self.l1_layer.reset()
        
        if self.l2_layer:
            self.l2_layer.reset()
        
        # 重置管理器
        if self.manager:
            self.manager.reset()
        
        # 重置好奇心引擎
        if self.curiosity_engine:
            self.curiosity_engine.reset()
        
        # 重置竞争性涌现
        if self.emergence_detector:
            self.emergence_detector.reset()
        
        self._last_emergence_indicators = None
        
        # 清空缓存
        self._current_step = 0
        self._l0_output_cache = None
        self._l1_state_cache = None
        self._l2_control_cache = None
        
        # 重置消融测试状态
        self._ablation_active = False
        self._ablation_step_count = 0
        
        # 重置统计
        self._stats = {
            "total_steps": 0,
            "l0_updates": 0,
            "l1_updates": 0,
            "l2_updates": 0,
            "regulation_cycles": 0,
            "ablation_tests": 0,
        }
        
        logger.info("MetaCognitiveSystem reset")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._stats.copy()
        
        stats["current_step"] = self._current_step
        stats["ablation_active"] = self._ablation_active
        
        return stats
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        验证系统
        
        Returns:
            (is_valid, error_messages): 是否有效以及错误消息列表
        """
        errors = []
        
        # 验证各层
        if self.l0_layer:
            is_valid, l0_errors = self.l0_layer.validate()
            if not is_valid:
                errors.extend([f"L0: {e}" for e in l0_errors])
        
        if self.l1_layer:
            is_valid, l1_errors = self.l1_layer.validate()
            if not is_valid:
                errors.extend([f"L1: {e}" for e in l1_errors])
        
        if self.l2_layer:
            is_valid, l2_errors = self.l2_layer.validate()
            if not is_valid:
                errors.extend([f"L2: {e}" for e in l2_errors])
        
        # 验证管理器
        if self.manager:
            is_valid, manager_errors = self.manager.validate()
            if not is_valid:
                errors.extend([f"Manager: {e}" for e in manager_errors])
        
        # 验证 L2 物理隔离
        if self.l2_layer:
            is_isolated, isolation_errors = self.l2_layer.check_physical_isolation()
            if not is_isolated:
                errors.extend([f"Physical isolation: {e}" for e in isolation_errors])
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.error(f"System validation failed: {errors}")
        
        return is_valid, errors
    
    def __repr__(self) -> str:
        return (
            f"MetaCognitive("
            f"step={self._current_step}, "
            f"use_l0={self.config.use_l0}, "
            f"use_l1={self.config.use_l1}, "
            f"use_l2={self.config.use_l2}, "
            f"ablation={self._ablation_active})"
        )


def create_meta_cognitive_from_config(
    global_config: ChronosConfig,
    device: Optional[str] = None
) -> MetaCognitive:
    """
    从全局配置创建元认知模块
    
    Args:
        global_config: 全局配置
        device: 计算设备
    
    Returns:
        MetaCognitive 实例
    """
    config = MetaCognitiveConfig(
        device=device or global_config.device,
    )
    
    module = MetaCognitive(
        config=config,
        dim_config=global_config.dim,
        meta_cognitive_config=global_config.meta_cognitive,
        memory_config=global_config.memory_temporal,
        global_config=global_config,
        device=device
    )
    
    return module


MetaCognitiveSystem = MetaCognitive
MetaCognitiveSystemConfig = MetaCognitiveConfig
create_meta_cognitive_system_from_config = create_meta_cognitive_from_config