"""
完整系统集成（System Integration）
=====================================

整合所有核心组件，实现完整的系统流程：
- 输入：外部输入（文本、环境状态等）
- 处理：双通道编码 → 融合 → 积分引擎演化 → 状态更新
- 调控：元认知监测 → 调控信号 → 参数调整
- 反思：实时反思 → 睡眠重放 → 参数更新
- 输出：系统响应（行为输出、状态报告）

核心功能：
1. 整合所有核心组件（表征系统、积分引擎、元认知、反思、训练、验证）
2. 实现完整系统流程（输入→积分→输出）
3. 实现系统控制器（启动、运行、停止、生命周期管理）
4. 提供统一接口（处理输入、获取状态、生成输出）

系统架构：
- ChronosSystem: 完整系统实例
- ChronosSystemController: 系统控制器
- SystemState: 系统状态管理

使用示例：
    # 创建系统
    system = ChronosSystem(config=ChronosSystemConfig())
    system.initialize()
    
    # 创建控制器
    controller = ChronosSystemController(system)
    
    # 启动系统
    controller.start()
    
    # 处理输入
    response = controller.process_input(text_input="你好")
    
    # 获取状态
    state = controller.get_system_state()
    
    # 停止系统
    controller.stop()
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from pathlib import Path
import json

from chronos_core.utils.config import ChronosConfig, EncoderConfig
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.dmn_system import DefaultModeNetwork
from chronos_core.core.meta_cognitive.meta_cognitive_system import (
    MetaCognitiveSystem,
    MetaCognitiveSystemConfig,
)
from chronos_core.core.reflection.reflection_system import (
    ReflectionSystem,
    ReflectionSystemConfig,
)
from chronos_core.memory.work_memory import WorkingMemory
from chronos_core.training.training_system import TrainingSystem, TrainingSystemConfig
from chronos_core.validation.validation_system import (
    ValidationSystem,
    ValidationSystemConfig,
    ValidationMode,
)
from chronos_core.representation.semantic_encoder import SemanticEncoder, create_semantic_encoder
from chronos_core.representation.logical_encoder import LogicalEncoder, create_logical_encoder
from chronos_core.representation.fusion import FusionModule, create_fusion_module


logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """系统状态枚举"""
    CREATED = "created"         # 已创建未初始化
    INITIALIZING = "initializing"  # 正在初始化
    READY = "ready"             # 已就绪
    RUNNING = "running"         # 正在运行
    PAUSED = "paused"           # 已暂停
    SLEEPING = "sleeping"       # 睡眠模式
    ERROR = "error"             # 错误状态
    STOPPED = "stopped"         # 已停止


class OperationMode(Enum):
    """操作模式枚举"""
    INTERACTIVE = "interactive"  # 交互模式（实时响应）
    CONTINUOUS = "continuous"    # 连续模式（长时间运行）
    TRAINING = "training"        # 训练模式
    VALIDATION = "validation"    # 验证模式
    DEMO = "demo"                # 演示模式


@dataclass
class ChronosSystemConfig:
    """完整系统配置"""
    
    # 基本配置
    device: str = "cuda"
    random_seed: int = 42
    
    # 操作模式
    default_operation_mode: OperationMode = OperationMode.INTERACTIVE
    
    # 系统参数
    default_dt: float = 0.01  # 默认时间步长
    max_continuous_hours: float = 72.0  # 最大连续运行时长
    
    # 表征系统
    enable_semantic_encoder: bool = True
    enable_logical_encoder: bool = True
    enable_fusion_module: bool = True
    
    # 元认知系统
    enable_meta_cognitive: bool = True
    
    # 反思系统
    enable_reflection: bool = True
    enable_realtime_reflection: bool = True
    enable_sleep_replay: bool = True
    
    # 工作记忆
    enable_working_memory: bool = True
    working_memory_capacity: int = 7
    
    # 训练系统
    enable_training_system: bool = False  # 默认不启用训练
    
    # 验证系统
    enable_validation_system: bool = False  # 默认不启用验证
    
    # 性能配置
    enable_performance_monitoring: bool = True
    performance_log_interval: int = 1000
    
    # 状态保存
    auto_save_state: bool = True
    state_save_interval_hours: float = 1.0
    state_save_path: str = "data/system_states"
    
    # 数值稳定性
    auto_stability_correction: bool = True
    stability_check_interval: int = 100


@dataclass
class SystemState:
    """系统状态数据"""
    
    # 系统状态
    status: SystemStatus = SystemStatus.CREATED
    operation_mode: OperationMode = OperationMode.INTERACTIVE
    
    # 时间信息
    start_time: float = 0.0
    elapsed_time: float = 0.0
    simulated_time: float = 0.0
    
    # 步数统计
    total_steps: int = 0
    integration_steps: int = 0
    reflection_count: int = 0
    sleep_count: int = 0
    
    # 性能指标
    avg_step_time_ms: float = 0.0
    total_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # 健康状态
    is_stable: bool = True
    stability_warnings: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status.value,
            "operation_mode": self.operation_mode.value,
            "start_time": self.start_time,
            "elapsed_time": self.elapsed_time,
            "simulated_time": self.simulated_time,
            "total_steps": self.total_steps,
            "integration_steps": self.integration_steps,
            "reflection_count": self.reflection_count,
            "sleep_count": self.sleep_count,
            "avg_step_time_ms": self.avg_step_time_ms,
            "total_memory_mb": self.total_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "is_stable": self.is_stable,
            "stability_warnings": self.stability_warnings,
            "error_count": self.error_count,
        }


@dataclass
class SystemResponse:
    """系统响应数据"""
    
    # 响应内容
    response_type: str = "text"  # 'text', 'action', 'state_update'
    content: Optional[str] = None
    
    # 状态信息
    state_before: Optional[SelfState] = None
    state_after: Optional[SelfState] = None
    
    # 元认知信息
    meta_cognitive_signal: Optional[torch.Tensor] = None
    confidence: float = 0.0
    intent_type: str = ""
    
    # 反思信息
    reflection_performed: bool = False
    reflection_result: Optional[Dict[str, Any]] = None
    
    # 性能信息
    processing_time_ms: float = 0.0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response_type": self.response_type,
            "content": self.content,
            "confidence": self.confidence,
            "intent_type": self.intent_type,
            "reflection_performed": self.reflection_performed,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class ChronosSystem:
    """
    完整 Chronos-Self 系统
    
    整合所有核心组件，实现完整的自我指涉动力学系统。
    
    核心组件：
    1. 表征系统（SemanticEncoder + LogicalEncoder + FusionModule）
    2. 积分引擎（IntegrationEngine）
    3. 默认模式网络（DefaultModeNetwork）
    4. 工作记忆（WorkingMemory）
    5. 元认知系统（MetaCognitiveSystem）
    6. 反思系统（ReflectionSystem）
    7. 训练系统（TrainingSystem）
    8. 验证系统（ValidationSystem）
    
    系统流程：
    输入 → 双通道编码 → 融合 → 积分引擎演化 → 元认知调控 → 反思修正 → 输出
    
    使用示例：
        system = ChronosSystem(config=ChronosSystemConfig())
        system.initialize()
        
        # 处理输入
        response = system.process_input(text="你好")
        
        # 获取状态
        state = system.get_current_state()
        
        # 运行验证
        validation_result = system.run_validation()
    """
    
    def __init__(
        self,
        config: Optional[ChronosSystemConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化 Chronos 系统
        
        Args:
            config: 系统配置
            global_config: 全局配置
            device: 计算设备
            seed: 随机种子
        """
        # 配置
        self.config = config or ChronosSystemConfig()
        self.global_config = global_config or ChronosConfig()
        
        # 设备和种子
        self.device = device or self.config.device or self.global_config.device
        self.seed = seed or self.config.random_seed or self.global_config.random_seed
        
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        
        # 系统状态
        self._system_state = SystemState()
        
        # 核心组件（待初始化）
        self.semantic_encoder: Optional[SemanticEncoder] = None
        self.logical_encoder: Optional[LogicalEncoder] = None
        self.fusion_module: Optional[FusionModule] = None
        self.integration_engine: Optional[IntegrationEngine] = None
        self.dmn: Optional[DefaultModeNetwork] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.meta_cognitive_system: Optional[MetaCognitiveSystem] = None
        self.reflection_system: Optional[ReflectionSystem] = None
        self.training_system: Optional[TrainingSystem] = None
        self.validation_system: Optional[ValidationSystem] = None
        
        # 当前状态
        self._current_self_state: Optional[SelfState] = None
        self._last_meta_cognitive_signal: Optional[torch.Tensor] = None
        
        # 性能监控
        self._step_times: List[float] = []
        self._performance_stats: Dict[str, Any] = {}
        
        # 初始化标志
        self._initialized = False
        
        logger.info(
            f"ChronosSystem created: "
            f"device={self.device}, "
            f"seed={self.seed}"
        )
    
    def initialize(self) -> None:
        """
        初始化所有核心组件
        
        包括：
        1. 表征系统（语义编码器 + 逻辑编码器 + 融合模块）
        2. 积分引擎
        3. 默认模式网络
        4. 工作记忆
        5. 元认知系统
        6. 反思系统
        7. 训练系统（可选）
        8. 验证系统（可选）
        """
        logger.info("=" * 80)
        logger.info("初始化 Chronos-Self 系统...")
        logger.info("=" * 80)
        
        self._system_state.status = SystemStatus.INITIALIZING
        
        try:
            # 1. 初始化积分引擎
            logger.info("[1/8] 初始化积分引擎...")
            self.integration_engine = IntegrationEngine(
                config=self.global_config,
                device=self.device,
                seed=self.seed
            )
            self.integration_engine.initialize()
            
            # 2. 初始化默认模式网络
            logger.info("[2/8] 初始化默认模式网络...")
            self.dmn = DefaultModeNetwork(
                chaos_config=self.global_config.chaos_injection,
                dim_config=self.global_config.dim,
                device=self.device,
                seed=self.seed
            )
            self.dmn.initialize()
            
            # 3. 初始化工作记忆
            logger.info("[3/8] 初始化工作记忆...")
            if self.config.enable_working_memory:
                self.working_memory = WorkingMemory(
                    capacity=self.config.working_memory_capacity,
                    fast_dim=self.global_config.dim.fast_variable_dim,
                    chunk_dim=self.global_config.dim.working_memory_dim,
                    device=self.device
                )
            
            # 4. 初始化语义编码器
            logger.info("[4/8] 初始化语义编码器...")
            if self.config.enable_semantic_encoder:
                self.semantic_encoder = create_semantic_encoder(
                    output_dim=self.global_config.dim.semantic_dim,
                    device=self.device,
                    config=self.global_config.encoder
                )
            
            # 5. 初始化逻辑编码器
            logger.info("[5/8] 初始化逻辑编码器...")
            if self.config.enable_logical_encoder:
                self.logical_encoder = create_logical_encoder(
                    config=self.global_config.encoder
                )
                self.logical_encoder.to(self.device)
            
            # 6. 初始化融合模块
            logger.info("[6/8] 初始化融合模块...")
            if self.config.enable_fusion_module:
                self.fusion_module = create_fusion_module(
                    config=self.global_config.encoder,
                    sem_dim=self.global_config.dim.semantic_dim,
                    log_dim=self.global_config.dim.physical_dim,
                    fusion_dim=self.global_config.dim.fusion_dim
                )
                self.fusion_module.to(self.device)
            
            # 7. 初始化元认知系统
            logger.info("[7/8] 初始化元认知系统...")
            if self.config.enable_meta_cognitive:
                self.meta_cognitive_system = MetaCognitiveSystem(
                    global_config=self.global_config,
                    device=self.device
                )
            
            # 8. 初始化反思系统
            logger.info("[8/8] 初始化反思系统...")
            if self.config.enable_reflection:
                reflection_config = ReflectionSystemConfig(
                    enable_realtime_reflection=self.config.enable_realtime_reflection,
                    enable_sleep_replay=self.config.enable_sleep_replay,
                    fast_dim=self.global_config.dim.fast_variable_dim,
                    slow_dim=self.global_config.dim.slow_variable_dim,
                )
                self.reflection_system = ReflectionSystem(
                    config=reflection_config,
                    global_config=self.global_config,
                    integration_engine=self.integration_engine,
                    device=self.device
                )
                self.reflection_system.initialize(self.integration_engine)
            
            # 初始化训练系统（可选）
            if self.config.enable_training_system:
                logger.info("[可选] 初始化训练系统...")
                self.training_system = TrainingSystem(
                    global_config=self.global_config,
                    integration_engine=self.integration_engine,
                    reflection_system=self.reflection_system,
                    meta_cognitive_system=self.meta_cognitive_system,
                    device=self.device
                )
                self.training_system.initialize()
            
            # 初始化验证系统（可选）
            if self.config.enable_validation_system:
                logger.info("[可选] 初始化验证系统...")
                self.validation_system = ValidationSystem(
                    config=self.global_config,
                    device=self.device
                )
            
            # 初始化当前自我状态
            self._current_self_state = SelfState(
                E_fast=torch.zeros(self.global_config.dim.fast_variable_dim, device=self.device),
                E_slow=torch.zeros(self.global_config.dim.slow_variable_dim, device=self.device),
                timestamp=0.0
            )
            
            # 更新系统状态
            self._system_state.status = SystemStatus.READY
            self._system_state.operation_mode = self.config.default_operation_mode
            self._initialized = True
            
            logger.info("=" * 80)
            logger.info("Chronos-Self 系统初始化完成！")
            logger.info(f"状态: {self._system_state.status.value}")
            logger.info(f"设备: {self.device}")
            logger.info("=" * 80)
            
        except Exception as e:
            self._system_state.status = SystemStatus.ERROR
            self._system_state.error_count += 1
            logger.error(f"系统初始化失败: {e}")
            raise
    
    def process_input(
        self,
        text: Optional[str] = None,
        physical_state: Optional[Dict[str, Any]] = None,
        external_input: Optional[ExternalInput] = None,
        dt: Optional[float] = None,
        apply_meta_cognitive: bool = True,
        apply_reflection: bool = True
    ) -> SystemResponse:
        """
        处理外部输入
        
        完整流程：
        1. 输入编码（语义 + 逻辑）
        2. 融合表征
        3. 积分引擎演化
        4. 元认知调控
        5. 反思修正
        6. 输出响应
        
        Args:
            text: 文本输入
            physical_state: 物理状态
            external_input: 外部输入对象
            dt: 时间步长
            apply_meta_cognitive: 是否应用元认知调控
            apply_reflection: 是否应用反思修正
        
        Returns:
            SystemResponse: 系统响应
        """
        if not self._initialized:
            raise ValueError("系统未初始化，请先调用 initialize()")
        
        if self._system_state.status not in [SystemStatus.READY, SystemStatus.RUNNING]:
            raise ValueError(f"系统状态异常: {self._system_state.status.value}")
        
        # 更新系统状态
        if self._system_state.status == SystemStatus.READY:
            self._system_state.status = SystemStatus.RUNNING
            self._system_state.start_time = time.time()
        
        # 开始计时
        start_process_time = time.time()
        
        # 创建响应对象
        response = SystemResponse()
        response.state_before = self._current_self_state.copy() if self._current_self_state else None
        
        # 时间步长
        dt = dt or self.config.default_dt
        
        # 1. 输入编码
        X_sem = None
        X_log = None
        intent_vector = None
        
        if external_input is not None:
            # 使用提供的外部输入
            X_sem = external_input.X_sem.to(self.device) if external_input.X_sem is not None else None
            X_log = external_input.X_log.to(self.device) if external_input.X_log is not None else None
        else:
            # 从文本和物理状态编码
            if text is not None and self.semantic_encoder is not None:
                # 语义编码
                intent_vector = self.semantic_encoder.forward(text)
                X_sem = intent_vector.combined_vector.to(self.device)
                
                # 提取意图信息
                response.intent_type = intent_vector.intent_type
                response.confidence = intent_vector.intent_confidence
                
                logger.debug(f"语义编码完成: intent={intent_vector.intent_type}, confidence={intent_vector.intent_confidence:.4f}")
            
            if physical_state is not None and self.logical_encoder is not None:
                # 逻辑编码
                X_log_tensor, _, _, _, _ = self.logical_encoder.encode_from_external_input(physical_state)
                X_log = X_log_tensor.to(self.device)
                
                logger.debug(f"逻辑编码完成: X_log_norm={torch.norm(X_log).item():.4f}")
        
        # 2. 融合表征
        X_fused = None
        if X_sem is not None and X_log is not None and self.fusion_module is not None:
            # 将张量转换为序列格式
            X_sem_seq = X_sem.unsqueeze(0).unsqueeze(0)  # (1, 1, sem_dim)
            X_log_seq = X_log.unsqueeze(0).unsqueeze(0)  # (1, 1, log_dim)
            
            # 融合
            fusion_output = self.fusion_module(X_sem_seq, X_log_seq, return_enriched=True)
            X_fused = fusion_output.X_fused.squeeze(0).squeeze(0)  # (fusion_dim,)
            
            logger.debug(f"融合完成: X_fused_norm={torch.norm(X_fused).item():.4f}")
        
        # 3. 创建外部输入对象
        if external_input is None:
            external_input = ExternalInput(
                X_sem=X_sem,
                X_log=X_log,
                importance=0.5,
                emotion_value=response.confidence,
                metadata={
                    "text": text,
                    "intent_type": response.intent_type,
                }
            )
        
        # 4. 获取元认知调控信号
        meta_cognitive_signal = None
        if apply_meta_cognitive and self.meta_cognitive_system is not None:
            # 运行元认知循环
            meta_output = self.meta_cognitive_system.forward(
                semantic_input=X_sem if X_sem is not None else torch.zeros(self.global_config.dim.semantic_dim, device=self.device),
                physical_input=X_log if X_log is not None else torch.zeros(self.global_config.dim.physical_dim, device=self.device),
                dt=dt,
                apply_regulation=True
            )
            
            if "l2_control" in meta_output:
                meta_cognitive_signal = meta_output["l2_control"]
                self._last_meta_cognitive_signal = meta_cognitive_signal
                response.meta_cognitive_signal = meta_cognitive_signal
                
                logger.debug(f"元认知调控完成: signal_norm={torch.norm(meta_cognitive_signal).item():.4f}")
        
        # 5. 积分引擎演化
        if self.integration_engine is not None:
            self._current_self_state = self.integration_engine.step(
                current_state=self._current_self_state,
                inputs=external_input,
                meta_cognitive_signal=meta_cognitive_signal,
                dt=dt
            )
            
            logger.debug(
                f"积分引擎演化完成: "
                f"timestamp={self._current_self_state.timestamp:.2f}, "
                f"E_fast_norm={self._current_self_state.get_fast_norm():.4f}"
            )
        
        # 6. 反思修正
        if apply_reflection and self.reflection_system is not None:
            reflection_result = self.reflection_system.add_online_step(
                state=self._current_self_state,
                inputs=external_input,
                metadata={"text": text}
            )
            
            response.reflection_performed = reflection_result.get("reflection_performed", False)
            response.reflection_result = reflection_result
            
            logger.debug(f"反思完成: performed={response.reflection_performed}")
        
        # 7. 更新工作记忆
        if self.working_memory is not None and X_fused is not None:
            # 创建新的组块
            chunk = self.working_memory.create_chunk(
                source_state=self._current_self_state.E_fast,
                chunk_type="semantic",
                initial_activation=response.confidence,
                metadata={"text": text, "intent": response.intent_type}
            )
            
            logger.debug(f"工作记忆更新: chunk_id={chunk.chunk_id}")
        
        # 8. 生成响应内容
        # 简化版本：基于意图类型生成响应
        if intent_vector is not None:
            response.content = self._generate_response_content(intent_vector, self._current_self_state)
        
        # 更新状态信息
        response.state_after = self._current_self_state.copy()
        
        # 更新系统统计
        self._system_state.total_steps += 1
        self._system_state.integration_steps += 1
        self._system_state.simulated_time += dt
        
        # 性能监控
        processing_time = (time.time() - start_process_time) * 1000
        response.processing_time_ms = processing_time
        self._step_times.append(processing_time)
        
        if len(self._step_times) > 100:
            self._system_state.avg_step_time_ms = np.mean(self._step_times[-100:])
        
        # 稳定性检查
        if self.config.auto_stability_correction:
            self._check_stability()
        
        # 性能日志
        if self._system_state.total_steps % self.config.performance_log_interval == 0:
            self._log_performance()
        
        logger.info(
            f"输入处理完成: "
            f"step={self._system_state.total_steps}, "
            f"time={processing_time:.2f}ms, "
            f"intent={response.intent_type}"
        )
        
        return response
    
    def _generate_response_content(
        self,
        intent_vector: Any,
        current_state: SelfState
    ) -> str:
        """
        生成响应内容
        
        Args:
            intent_vector: 意图向量
            current_state: 当前状态
        
        Returns:
            响应文本
        """
        # 简化版本：基于意图类型生成固定响应
        intent_type = intent_vector.intent_type
        
        responses = {
            "inform": "我收到了您的信息，正在处理...",
            "request": "我理解您的请求，会尽力帮助您。",
            "question": "这是一个好问题，让我思考一下...",
            "greet": "你好！我是 Chronos-Self 系统，很高兴与您交流。",
            "acknowledge": "好的，我明白了。",
            "command": "收到指令，正在执行...",
            "other": "我正在处理您的输入...",
        }
        
        response = responses.get(intent_type, "我正在思考...")
        
        # 添加状态信息（可选）
        if current_state:
            state_info = f"当前状态稳定度: {current_state.get_fast_norm():.4f}"
            response += f"\n({state_info})"
        
        return response
    
    def _check_stability(self) -> None:
        """检查系统稳定性"""
        if self._current_self_state is None:
            return
        
        # 检查快变量和慢变量的数值稳定性
        E_fast = self._current_self_state.E_fast
        E_slow = self._current_self_state.E_slow
        
        # 检查 NaN 和 Inf
        if torch.isnan(E_fast).any() or torch.isinf(E_fast).any():
            logger.warning("快变量出现 NaN 或 Inf，应用修正...")
            E_fast = torch.nan_to_num(E_fast, nan=0.0, posinf=1.0, neginf=-1.0)
            self._current_self_state.E_fast = E_fast
            self._system_state.stability_warnings += 1
        
        if torch.isnan(E_slow).any() or torch.isinf(E_slow).any():
            logger.warning("慢变量出现 NaN 或 Inf，应用修正...")
            E_slow = torch.nan_to_num(E_slow, nan=0.0, posinf=1.0, neginf=-1.0)
            self._current_self_state.E_slow = E_slow
            self._system_state.stability_warnings += 1
        
        # 检查状态范数
        fast_norm = self._current_self_state.get_fast_norm()
        if fast_norm > 1000.0:
            logger.warning(f"快变量范数过大: {fast_norm:.4f}")
            self._system_state.stability_warnings += 1
            self._system_state.is_stable = False
        
        if self._system_state.stability_warnings > 10:
            logger.error("稳定性警告过多，系统可能不稳定")
            self._system_state.status = SystemStatus.ERROR
    
    def _log_performance(self) -> None:
        """记录性能日志"""
        logger.info(
            f"[性能监控] "
            f"steps={self._system_state.total_steps}, "
            f"avg_time={self._system_state.avg_step_time_ms:.2f}ms, "
            f"simulated_time={self._system_state.simulated_time:.2f}s, "
            f"stable={self._system_state.is_stable}"
        )
    
    def get_current_state(self) -> SelfState:
        """获取当前自我状态"""
        return self._current_self_state.copy() if self._current_self_state else None
    
    def get_system_state(self) -> SystemState:
        """获取系统状态"""
        return self._system_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        stats = {
            "system_state": self._system_state.to_dict(),
            "initialized": self._initialized,
            "components": {
                "semantic_encoder": self.semantic_encoder is not None,
                "logical_encoder": self.logical_encoder is not None,
                "fusion_module": self.fusion_module is not None,
                "integration_engine": self.integration_engine is not None,
                "dmn": self.dmn is not None,
                "working_memory": self.working_memory is not None,
                "meta_cognitive_system": self.meta_cognitive_system is not None,
                "reflection_system": self.reflection_system is not None,
                "training_system": self.training_system is not None,
                "validation_system": self.validation_system is not None,
            },
            "performance": {
                "avg_step_time_ms": self._system_state.avg_step_time_ms,
                "total_steps": self._system_state.total_steps,
            },
            "current_self_state": {
                "E_fast_norm": self._current_self_state.get_fast_norm() if self._current_self_state else 0.0,
                "E_slow_norm": self._current_self_state.get_slow_norm() if self._current_self_state else 0.0,
                "timestamp": self._current_self_state.timestamp if self._current_self_state else 0.0,
            }
        }
        
        # 添加各组件的统计信息
        if self.integration_engine:
            stats["integration_engine_stats"] = self.integration_engine.get_state_monitoring()
        
        if self.meta_cognitive_system:
            stats["meta_cognitive_stats"] = self.meta_cognitive_system.get_statistics()
        
        if self.reflection_system:
            stats["reflection_stats"] = self.reflection_system.get_statistics()
        
        if self.working_memory:
            stats["working_memory_stats"] = self.working_memory.get_statistics()
        
        return stats
    
    def run_validation(
        self,
        mode: Optional[ValidationMode] = None,
        verbose: bool = True
    ) -> Any:
        """
        运行验证
        
        Args:
            mode: 验证模式
            verbose: 详细日志
        
        Returns:
            验证结果
        """
        if not self._initialized:
            raise ValueError("系统未初始化")
        
        if self.validation_system is None:
            raise ValueError("验证系统未启用")
        
        # 更新系统状态
        old_status = self._system_state.status
        self._system_state.status = SystemStatus.RUNNING
        self._system_state.operation_mode = OperationMode.VALIDATION
        
        logger.info("开始验证...")
        
        # 执行验证
        result = self.validation_system.run_validation(
            engine=self.integration_engine,
            mode=mode,
            initial_state=self._current_self_state,
            verbose=verbose
        )
        
        # 恢复系统状态
        self._system_state.status = old_status
        self._system_state.operation_mode = self.config.default_operation_mode
        
        logger.info(f"验证完成: passed={result.overall_passed}")
        
        return result
    
    def run_training(
        self,
        num_epochs: Optional[int] = None,
        callback: Optional[Any] = None
    ) -> Any:
        """
        运行训练
        
        Args:
            num_epochs: 训练轮数
            callback: 回调函数
        
        Returns:
            训练历史
        """
        if not self._initialized:
            raise ValueError("系统未初始化")
        
        if self.training_system is None:
            raise ValueError("训练系统未启用")
        
        # 更新系统状态
        old_status = self._system_state.status
        self._system_state.status = SystemStatus.RUNNING
        self._system_state.operation_mode = OperationMode.TRAINING
        
        logger.info("开始训练...")
        
        # 执行训练
        history = self.training_system.train(
            num_epochs=num_epochs,
            callback=callback
        )
        
        # 恢复系统状态
        self._system_state.status = old_status
        self._system_state.operation_mode = self.config.default_operation_mode
        
        logger.info(f"训练完成: epochs={history.total_epochs}")
        
        return history
    
    def trigger_sleep(
        self,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        触发睡眠重放
        
        Args:
            force: 强制触发
        
        Returns:
            睡眠结果
        """
        if not self._initialized:
            raise ValueError("系统未初始化")
        
        if self.reflection_system is None:
            raise ValueError("反思系统未启用")
        
        # 更新系统状态
        old_status = self._system_state.status
        self._system_state.status = SystemStatus.SLEEPING
        
        logger.info("触发睡眠重放...")
        
        # 执行睡眠
        result = self.reflection_system.perform_sleep(force=force)
        
        # 更新统计
        self._system_state.sleep_count += 1
        
        # 恢复系统状态
        self._system_state.status = old_status
        
        logger.info(f"睡眠完成: success={result.get('success', False)}")
        
        return result
    
    def save_state(self, filepath: Optional[str] = None) -> str:
        """
        保存系统状态
        
        Args:
            filepath: 文件路径
        
        Returns:
            保存的文件路径
        """
        filepath = filepath or f"{self.config.state_save_path}/state_{self._system_state.total_steps}.json"
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        state_data = {
            "system_state": self._system_state.to_dict(),
            "self_state": self._current_self_state.to_dict() if self._current_self_state else None,
            "statistics": self.get_statistics(),
            "config": {
                "device": self.device,
                "seed": self.seed,
            },
            "timestamp": time.time(),
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"系统状态保存至: {filepath}")
        
        return filepath
    
    def load_state(self, filepath: str) -> None:
        """
        加载系统状态
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # 恢复系统状态
        if state_data.get("self_state"):
            self._current_self_state = SelfState.from_dict(state_data["self_state"])
        
        logger.info(f"系统状态从 {filepath} 加载完成")
    
    def reset(self) -> None:
        """重置系统"""
        # 重置各组件
        if self.integration_engine:
            self.integration_engine.reset()
        
        if self.dmn:
            self.dmn.reset()
        
        if self.working_memory:
            self.working_memory.clear()
        
        if self.meta_cognitive_system:
            self.meta_cognitive_system.reset()
        
        if self.reflection_system:
            self.reflection_system.reset()
        
        if self.training_system:
            self.training_system.reset()
        
        # 重置当前状态
        self._current_self_state = SelfState(
            E_fast=torch.zeros(self.global_config.dim.fast_variable_dim, device=self.device),
            E_slow=torch.zeros(self.global_config.dim.slow_variable_dim, device=self.device),
            timestamp=0.0
        )
        
        # 重置系统状态
        self._system_state = SystemState()
        self._system_state.status = SystemStatus.READY
        
        # 清空性能监控
        self._step_times.clear()
        
        logger.info("系统已重置")
    
    def shutdown(self) -> None:
        """关闭系统"""
        # 保存最终状态
        if self.config.auto_save_state:
            self.save_state()
        
        # 更新系统状态
        self._system_state.status = SystemStatus.STOPPED
        
        logger.info("系统已关闭")
    
    def __repr__(self) -> str:
        return (
            f"ChronosSystem("
            f"status={self._system_state.status.value}, "
            f"initialized={self._initialized}, "
            f"steps={self._system_state.total_steps}, "
            f"device={self.device})"
        )


class ChronosSystemController:
    """
    Chronos 系统控制器
    
    提供系统启动、运行、停止、生命周期管理的统一接口。
    
    功能：
    1. 系统启动和初始化
    2. 运行模式切换
    3. 输入处理
    4. 状态查询
    5. 系统停止和清理
    
    使用示例：
        system = ChronosSystem()
        controller = ChronosSystemController(system)
        
        controller.start()
        response = controller.process_input("你好")
        controller.stop()
    """
    
    def __init__(
        self,
        system: ChronosSystem,
        auto_start: bool = False
    ):
        """
        初始化控制器
        
        Args:
            system: Chronos 系统实例
            auto_start: 是否自动启动
        """
        self.system = system
        
        # 控制器状态
        self._is_running = False
        self._start_time: Optional[float] = None
        
        # 回调函数
        self._on_step_callback: Optional[Any] = None
        self._on_error_callback: Optional[Any] = None
        
        if auto_start:
            self.start()
        
        logger.info(f"ChronosSystemController created: system={system}")
    
    def start(self) -> None:
        """
        启动系统
        
        包括：
        1. 系统初始化
        2. 状态检查
        3. 进入运行模式
        """
        if self._is_running:
            logger.warning("系统已在运行")
            return
        
        # 初始化系统（如果未初始化）
        if not self.system._initialized:
            self.system.initialize()
        
        # 更新状态
        self._is_running = True
        self._start_time = time.time()
        self.system._system_state.status = SystemStatus.RUNNING
        
        logger.info("系统已启动")
    
    def stop(self) -> None:
        """
        停止系统
        
        包括：
        1. 保存状态
        2. 清理资源
        3. 更新状态
        """
        if not self._is_running:
            logger.warning("系统未在运行")
            return
        
        # 关闭系统
        self.system.shutdown()
        
        # 更新状态
        self._is_running = False
        self.system._system_state.status = SystemStatus.STOPPED
        
        logger.info(f"系统已停止，运行时间: {time.time() - self._start_time:.2f}秒")
    
    def pause(self) -> None:
        """暂停系统"""
        if not self._is_running:
            return
        
        self.system._system_state.status = SystemStatus.PAUSED
        logger.info("系统已暂停")
    
    def resume(self) -> None:
        """恢复系统"""
        if self.system._system_state.status != SystemStatus.PAUSED:
            return
        
        self.system._system_state.status = SystemStatus.RUNNING
        logger.info("系统已恢复")
    
    def process_input(
        self,
        text: Optional[str] = None,
        physical_state: Optional[Dict[str, Any]] = None,
        external_input: Optional[ExternalInput] = None
    ) -> SystemResponse:
        """
        处理输入
        
        Args:
            text: 文本输入
            physical_state: 物理状态
            external_input: 外部输入对象
        
        Returns:
            SystemResponse: 系统响应
        """
        if not self._is_running:
            raise ValueError("系统未运行")
        
        response = self.system.process_input(
            text=text,
            physical_state=physical_state,
            external_input=external_input
        )
        
        # 回调
        if self._on_step_callback:
            self._on_step_callback(self.system._system_state.total_steps, response)
        
        return response
    
    def process_batch(
        self,
        inputs: List[Dict[str, Any]]
    ) -> List[SystemResponse]:
        """
        批量处理输入
        
        Args:
            inputs: 输入列表
        
        Returns:
            响应列表
        """
        responses = []
        
        for input_dict in inputs:
            response = self.process_input(
                text=input_dict.get("text"),
                physical_state=input_dict.get("physical_state"),
                external_input=input_dict.get("external_input")
            )
            responses.append(response)
        
        return responses
    
    def run_continuous(
        self,
        duration_hours: float,
        inputs_generator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        连续运行
        
        Args:
            duration_hours: 运行时长（小时）
            inputs_generator: 输入生成器
        
        Returns:
            运行统计
        """
        if not self._is_running:
            raise ValueError("系统未运行")
        
        # 更新模式
        self.system._system_state.operation_mode = OperationMode.CONTINUOUS
        
        logger.info(f"开始连续运行: duration={duration_hours}h")
        
        # 运行统计
        stats = {
            "start_time": time.time(),
            "duration_hours": duration_hours,
            "total_responses": 0,
            "errors": 0,
        }
        
        # 计算总步数
        dt = self.system.config.default_dt
        total_steps = int(duration_hours * 3600 / dt)
        
        for step_idx in range(total_steps):
            try:
                # 生成输入
                if inputs_generator:
                    input_dict = inputs_generator(step_idx)
                    response = self.process_input(**input_dict)
                else:
                    # 无输入运行
                    response = self.system.process_input(external_input=None)
                
                stats["total_responses"] += 1
                
                # 检查系统状态
                if self.system._system_state.status == SystemStatus.ERROR:
                    stats["errors"] += 1
                    logger.error(f"系统错误 at step {step_idx}")
                    break
                
            except Exception as e:
                stats["errors"] += 1
                logger.error(f"处理错误 at step {step_idx}: {e}")
                
                if self._on_error_callback:
                    self._on_error_callback(step_idx, e)
        
        # 更新统计
        stats["end_time"] = time.time()
        stats["actual_duration"] = stats["end_time"] - stats["start_time"]
        stats["system_state"] = self.system._system_state.to_dict()
        
        logger.info(f"连续运行完成: responses={stats['total_responses']}, errors={stats['errors']}")
        
        return stats
    
    def get_system_state(self) -> SystemState:
        """获取系统状态"""
        return self.system.get_system_state()
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.system.get_statistics()
    
    def set_callbacks(
        self,
        on_step: Optional[Any] = None,
        on_error: Optional[Any] = None
    ) -> None:
        """
        设置回调函数
        
        Args:
            on_step: 步骤回调
            on_error: 错误回调
        """
        self._on_step_callback = on_step
        self._on_error_callback = on_error
    
    def __repr__(self) -> str:
        return (
            f"ChronosSystemController("
            f"running={self._is_running}, "
            f"system={self.system})"
        )


def create_chronos_system_from_config(
    global_config: ChronosConfig,
    system_config: Optional[ChronosSystemConfig] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> ChronosSystem:
    """
    从配置创建 Chronos 系统
    
    Args:
        global_config: 全局配置
        system_config: 系统配置
        device: 计算设备
        seed: 随机种子
    
    Returns:
        ChronosSystem 实例
    """
    system = ChronosSystem(
        config=system_config,
        global_config=global_config,
        device=device,
        seed=seed
    )
    
    system.initialize()
    
    return system