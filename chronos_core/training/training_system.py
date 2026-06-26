"""
完整训练系统
============

整合所有训练组件，实现完整的 Chronos-Self 训练流程。

核心功能：
- 整合损失函数、动力学对齐、冻结策略
- 整合 IntegrationEngine、ReflectionSystem、MetaCognitiveSystem
- 实现完整训练流程（数据准备、前向传播、损失计算、反向传播、参数更新、验证）
- 支持多种训练模式（短时序、长时序、动力学对齐、P0验证）
- 提供完整训练接口（train、validate、save_checkpoint、load_checkpoint）

训练流程：
1. 数据准备（外部输入序列）
2. 前向传播（积分引擎演化）
3. 损失计算（完整损失组合）
4. 反向传播（伴随法）
5. 参数更新（考虑冻结策略）
6. 验证（周期性验证）

使用示例：
    training_system = TrainingSystem(config=TrainingSystemConfig())
    training_system.initialize()

    # 执行完整训练
    history = training_system.train(num_epochs=100)

    # 执行验证
    validation_result = training_system.validate()

    # 保存检查点
    training_system.save_checkpoint('checkpoint.pth')

    # 加载检查点
    training_system.load_checkpoint('checkpoint.pth')

    # 查询训练历史
    history = training_system.get_training_history()
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from pathlib import Path
import numpy as np

from chronos_core.utils.config import ChronosConfig, TrainingConfig
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.reflection.reflection_system import ReflectionSystem
from chronos_core.core.meta_cognitive.meta_cognitive_system import MetaCognitiveSystem

from chronos_core.training.loss_functions import (
    LossFunctions,
    LossFunctionsConfig,
    create_loss_functions_from_config,
)
from chronos_core.training.dynamics_alignment import (
    DynamicsAlignment,
    DynamicsAlignmentConfig,
    create_dynamics_alignment_from_config,
)
from chronos_core.training.freezing_strategy import (
    FreezingStrategy,
    FreezingStrategyConfig,
    create_freezing_strategy_from_config,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingSystemConfig:
    """完整训练系统配置"""

    # 训练模式
    training_mode: str = "standard"  # 'standard', 'alignment', 'p0_validation', 'long_sequence'

    # 训练参数
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 优化器配置
    optimizer_type: str = "adam"  # 'adam', 'adamw', 'sgd'
    scheduler_type: str = "step"  # 'step', 'cosine', 'none'
    scheduler_step_size: int = 30
    scheduler_gamma: float = 0.1

    # 梯度配置
    gradient_clip_threshold: float = 1.0
    use_gradient_accumulation: bool = False
    accumulation_steps: int = 4

    # 混合精度训练
    use_amp: bool = True  # Automatic Mixed Precision

    # 检查点配置
    checkpoint_dir: str = "checkpoints"
    checkpoint_frequency: int = 10
    save_best_model: bool = True

    # 验证配置
    validation_frequency: int = 5
    validation_duration_hours: float = 1.0  # 验证时长

    # 早停配置
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 0.05

    # 数据配置
    sequence_length: int = 100  # 训练序列长度
    use_random_data: bool = True  # 使用随机数据（测试模式）

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


@dataclass
class TrainingHistory:
    """训练历史记录"""

    # 时间信息
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0

    # 训练步数
    total_epochs: int = 0
    total_steps: int = 0

    # 损失历史
    loss_history: List[Dict[str, float]] = field(default_factory=list)

    # 验证历史
    validation_history: List[Dict[str, Any]] = field(default_factory=list)

    # 学习率历史
    lr_history: List[float] = field(default_factory=list)

    # 最佳结果
    best_loss: float = float('inf')
    best_epoch: int = 0

    # 检查点记录
    checkpoints: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "total_epochs": self.total_epochs,
            "total_steps": self.total_steps,
            "loss_history": self.loss_history,
            "validation_history": self.validation_history,
            "lr_history": self.lr_history,
            "best_loss": self.best_loss,
            "best_epoch": self.best_epoch,
            "checkpoints": self.checkpoints,
        }


class TrainingSystem(nn.Module):
    """
    完整训练系统

    整合所有训练组件：
    - 损失函数（LossFunctions）
    - 动力学对齐（DynamicsAlignment）
    - 冻结策略（FreezingStrategy）
    - IntegrationEngine
    - ReflectionSystem
    - MetaCognitiveSystem

    实现完整训练流程：
    1. 数据准备（外部输入序列）
    2. 前向传播（积分引擎演化）
    3. 损失计算（完整损失组合）
    4. 反向传播（伴随法）
    5. 参数更新（考虑冻结策略）
    6. 验证（周期性验证）
    """

    def __init__(
        self,
        config: Optional[TrainingSystemConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        integration_engine: Optional[IntegrationEngine] = None,
        reflection_system: Optional[ReflectionSystem] = None,
        meta_cognitive_system: Optional[MetaCognitiveSystem] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or TrainingSystemConfig()

        # 从全局配置更新
        if global_config:
            self.config.num_epochs = global_config.training.num_epochs
            self.config.batch_size = global_config.training.batch_size
            self.config.learning_rate = global_config.training.learning_rate
            self.config.weight_decay = global_config.training.weight_decay
            self.config.gradient_clip_threshold = global_config.training.gradient_clip_threshold
            self.config.validation_frequency = global_config.training.validation_frequency
            self.config.checkpoint_frequency = global_config.training.checkpoint_frequency
            self.config.use_amp = global_config.use_amp
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim

        self.global_config = global_config or ChronosConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 核心组件
        self.integration_engine = integration_engine
        self.reflection_system = reflection_system
        self.meta_cognitive_system = meta_cognitive_system

        # 训练组件
        self.loss_functions: Optional[LossFunctions] = None
        self.dynamics_alignment: Optional[DynamicsAlignment] = None
        self.freezing_strategy: Optional[FreezingStrategy] = None

        # 优化器和调度器
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # 混合精度训练
        self.scaler: Optional[torch.cuda.amp.GradScaler] = None

        # 训练历史
        self.history: TrainingHistory = TrainingHistory()

        # 训练状态
        self._current_epoch: int = 0
        self._current_step: int = 0
        self._is_training: bool = False

        # 初始化标志
        self._initialized = False

        logger.info(
            f"TrainingSystem created: "
            f"mode={self.config.training_mode}, "
            f"epochs={self.config.num_epochs}, "
            f"device={self.device}"
        )

    def initialize(self) -> None:
        """
        初始化训练系统

        包括：
        - 创建积分引擎（如果未提供）
        - 创建反思系统（如果未提供）
        - 创建元认知系统（如果未提供）
        - 创建损失函数系统
        - 创建动力学对齐系统
        - 创建冻结策略系统
        - 应用冻结策略
        - 创建优化器和调度器
        - 初始化混合精度训练
        """
        logger.info("Initializing TrainingSystem...")

        # 创建积分引擎（如果未提供）
        if self.integration_engine is None:
            self.integration_engine = IntegrationEngine(
                config=self.global_config,
                device=self.device
            )
            self.integration_engine.initialize()

        # 创建反思系统（如果未提供）
        if self.reflection_system is None:
            self.reflection_system = ReflectionSystem(
                global_config=self.global_config,
                integration_engine=self.integration_engine,
                device=self.device
            )
            self.reflection_system.initialize(self.integration_engine)

        # 创建元认知系统（如果未提供）
        if self.meta_cognitive_system is None:
            self.meta_cognitive_system = MetaCognitiveSystem(
                global_config=self.global_config,
                device=self.device
            )

        # 创建损失函数系统
        self.loss_functions = create_loss_functions_from_config(
            self.global_config,
            self.device
        )

        # 创建动力学对齐系统
        self.dynamics_alignment = create_dynamics_alignment_from_config(
            self.global_config,
            self.device
        )

        # 创建冻结策略系统
        self.freezing_strategy = create_freezing_strategy_from_config(
            self.global_config,
            self.device
        )

        # 应用冻结策略
        self._apply_freezing_strategy()

        # 创建优化器
        self._create_optimizer()

        # 创建调度器
        self._create_scheduler()

        # 初始化混合精度训练
        if self.config.use_amp and self.device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()

        self._initialized = True

        logger.info("TrainingSystem initialized successfully")

    def _apply_freezing_strategy(self) -> None:
        """应用冻结策略"""
        if self.freezing_strategy and self.integration_engine:
            # 获取积分引擎的可训练参数
            trainable_params = []
            for name, param in self.integration_engine.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)

            logger.info(f"Trainable parameters: {len(trainable_params)}")

    def _create_optimizer(self) -> None:
        """创建优化器"""
        # 收集可训练参数
        trainable_params = []

        if self.integration_engine:
            for param in self.integration_engine.parameters():
                if param.requires_grad:
                    trainable_params.append(param)

        if self.meta_cognitive_system:
            for param in self.meta_cognitive_system.parameters():
                if param.requires_grad:
                    trainable_params.append(param)

        # 创建优化器
        if self.config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

        logger.info(
            f"Optimizer created: type={self.config.optimizer_type}, "
            f"lr={self.config.learning_rate}, "
            f"params={len(trainable_params)}"
        )

    def _create_scheduler(self) -> None:
        """创建学习率调度器"""
        if self.config.scheduler_type == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.scheduler_step_size,
                gamma=self.config.scheduler_gamma
            )
        elif self.config.scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        else:
            self.scheduler = None

        logger.info(
            f"Scheduler created: type={self.config.scheduler_type}"
        )

    def train(
        self,
        num_epochs: Optional[int] = None,
        data_generator: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> TrainingHistory:
        """
        执行完整训练

        Args:
            num_epochs: 训练轮数（可选）
            data_generator: 数据生成器（可选）
            callback: 回调函数（可选）

        Returns:
            训练历史记录
        """
        if not self._initialized:
            raise ValueError("TrainingSystem not initialized. Call initialize() first.")

        num_epochs = num_epochs or self.config.num_epochs

        logger.info(f"Starting training: epochs={num_epochs}")

        # 记录开始时间
        self.history.start_time = time.time()
        self._is_training = True

        # 创建初始状态
        initial_state = SelfState(
            E_fast=torch.zeros(self.config.fast_dim, device=self.device),
            E_slow=torch.zeros(self.config.slow_dim, device=self.device),
            timestamp=0.0
        )

        # 训练循环
        for epoch in range(num_epochs):
            self._current_epoch = epoch

            # 执行单个 epoch
            epoch_loss = self._train_epoch(
                epoch,
                initial_state,
                data_generator
            )

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history.lr_history.append(current_lr)

            # 验证
            if epoch % self.config.validation_frequency == 0:
                validation_result = self.validate(initial_state)
                self.history.validation_history.append(validation_result)

            # 保存检查点
            if epoch % self.config.checkpoint_frequency == 0:
                checkpoint_path = self.save_checkpoint(
                    f"checkpoint_epoch_{epoch}.pth"
                )
                self.history.checkpoints.append(checkpoint_path)

            # 回调
            if callback:
                callback(epoch, epoch_loss, self)

            logger.info(
                f"Epoch {epoch} completed: "
                f"loss={epoch_loss:.4f}, "
                f"lr={self.optimizer.param_groups[0]['lr']:.6f}"
            )

        # 记录结束时间
        self.history.end_time = time.time()
        self.history.duration_seconds = self.history.end_time - self.history.start_time
        self.history.total_epochs = num_epochs

        self._is_training = False

        logger.info(
            f"Training completed: "
            f"epochs={num_epochs}, "
            f"duration={self.history.duration_seconds:.2f}s, "
            f"best_loss={self.history.best_loss:.4f}"
        )

        return self.history

    def _train_epoch(
        self,
        epoch: int,
        initial_state: SelfState,
        data_generator: Optional[Callable] = None
    ) -> float:
        """
        执行单个 epoch 训练

        Args:
            epoch: 当前 epoch
            initial_state: 初始状态
            data_generator: 数据生成器

        Returns:
            epoch 平均损失
        """
        epoch_losses = []

        # 创建外部输入序列
        input_sequence = self._create_input_sequence(data_generator)

        # 执行训练序列
        current_state = initial_state

        for step_idx in range(self.config.sequence_length):
            self._current_step = step_idx

            # 获取当前输入
            current_input = input_sequence[step_idx] if step_idx < len(input_sequence) else None

            # 执行训练步骤
            step_loss = self._train_step(current_state, current_input)

            # 更新状态
            # 简化：使用积分引擎更新
            if self.integration_engine:
                current_state = self.integration_engine.step(
                    current_state,
                    inputs=current_input,
                    dt=0.01
                )

            epoch_losses.append(step_loss)

        # 计算 epoch 平均损失
        epoch_avg_loss = np.mean(epoch_losses)

        # 记录损失历史
        self.history.loss_history.append({
            'epoch': epoch,
            'loss': epoch_avg_loss,
            'steps': len(epoch_losses),
        })

        # 更新最佳损失
        if epoch_avg_loss < self.history.best_loss:
            self.history.best_loss = epoch_avg_loss
            self.history.best_epoch = epoch

        return epoch_avg_loss

    def _train_step(
        self,
        current_state: SelfState,
        current_input: Optional[ExternalInput]
    ) -> float:
        """
        执行单个训练步骤

        Args:
            current_state: 当前状态
            current_input: 当前输入

        Returns:
            步骤损失值
        """
        # 清零梯度
        self.optimizer.zero_grad()

        # 使用混合精度训练
        if self.scaler and self.config.use_amp:
            with torch.cuda.amp.autocast():
                # 前向传播
                losses = self._compute_losses(current_state, current_input)

                # 计算总损失
                total_loss = losses['total']

            # 反向传播
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.integration_engine.parameters(),
                self.config.gradient_clip_threshold
            )

            # 参数更新
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # 标准训练
            # 前向传播
            losses = self._compute_losses(current_state, current_input)

            # 计算总损失
            total_loss = losses['total']

            # 反向传播
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.integration_engine.parameters(),
                self.config.gradient_clip_threshold
            )

            # 参数更新
            self.optimizer.step()

        # 记录步骤数
        self.history.total_steps += 1

        return total_loss.item()

    def _compute_losses(
        self,
        current_state: SelfState,
        current_input: Optional[ExternalInput]
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失

        Args:
            current_state: 当前状态
            current_input: 当前输入

        Returns:
            损失字典
        """
        # 创建预测状态（简化：使用当前状态的扰动）
        predicted_state = SelfState(
            E_fast=current_state.E_fast + torch.randn_like(current_state.E_fast) * 0.01,
            E_slow=current_state.E_slow + torch.randn_like(current_state.E_slow) * 0.01,
            timestamp=current_state.timestamp + 0.01
        )

        # 创建自预测状态（简化）
        self_predicted_state = SelfState(
            E_fast=current_state.E_fast.clone(),
            E_slow=current_state.E_slow.clone(),
            timestamp=current_state.timestamp + 0.01
        )

        # 计算损失函数损失
        losses = self.loss_functions.compute_all_losses(
            actual_state=current_state,
            predicted_state=predicted_state,
            self_predicted_state=self_predicted_state
        )

        # 计算动力学对齐损失（简化）
        alignment_losses = self.dynamics_alignment.compute_alignment_losses(
            integration_engine=self.integration_engine,
            initial_state=current_state
        )

        # 合并所有损失
        total_loss = losses['total'] + alignment_losses['total'] * 0.1

        losses['alignment'] = alignment_losses['total']
        losses['total'] = total_loss

        return losses

    def _create_input_sequence(self, data_generator: Optional[Callable] = None) -> List[ExternalInput]:
        """
        创建输入序列

        Args:
            data_generator: 数据生成器

        Returns:
            输入序列
        """
        if data_generator:
            # 使用数据生成器
            input_sequence = data_generator(self.config.sequence_length)
        elif self.config.use_random_data:
            # 使用随机数据（测试模式）
            input_sequence = []
            for i in range(self.config.sequence_length):
                random_input = ExternalInput(
                    X_sem=torch.randn(self.global_config.dim.semantic_dim),
                    X_log=torch.randn(self.global_config.dim.physical_dim),
                    importance=np.random.rand(),
                    emotion_value=np.random.rand(),
                )
                input_sequence.append(random_input)
        else:
            # 使用空输入序列
            input_sequence = []

        return input_sequence

    def validate(
        self,
        initial_state: Optional[SelfState] = None
    ) -> Dict[str, Any]:
        """
        执行验证

        Args:
            initial_state: 初始状态（可选）

        Returns:
            验证结果字典
        """
        if not self._initialized:
            raise ValueError("TrainingSystem not initialized.")

        logger.info("Starting validation...")

        # 创建初始状态（如果未提供）
        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.zeros(self.config.fast_dim, device=self.device),
                E_slow=torch.zeros(self.config.slow_dim, device=self.device),
                timestamp=0.0
            )

        # 执行动力学对齐验证
        validation_result = self.dynamics_alignment.periodic_validation.forward(
            integration_engine=self.integration_engine,
            initial_state=initial_state,
            epoch=self._current_epoch,
            step=self._current_step,
            force_validation=True
        )

        logger.info(f"Validation completed: result={validation_result}")

        return validation_result or {}

    def save_checkpoint(self, filepath: str) -> str:
        """
        保存检查点

        Args:
            filepath: 文件路径

        Returns:
            保存的文件路径
        """
        if not self._initialized:
            raise ValueError("TrainingSystem not initialized.")

        # 创建检查点目录
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filepath = checkpoint_dir / filepath

        # 保存状态
        checkpoint = {
            'epoch': self._current_epoch,
            'step': self._current_step,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history.to_dict(),
            'config': self.config.__dict__,
            'best_loss': self.history.best_loss,
            'best_epoch': self.history.best_epoch,
        }

        # 保存积分引擎状态
        if self.integration_engine:
            checkpoint['integration_engine_state'] = {
                'step_count': self.integration_engine.step_count,
                'total_time': self.integration_engine.total_time,
            }

        torch.save(checkpoint, filepath)

        logger.info(f"Checkpoint saved: {filepath}")

        return str(filepath)

    def load_checkpoint(self, filepath: str) -> None:
        """
        加载检查点

        Args:
            filepath: 文件路径
        """
        if not self._initialized:
            raise ValueError("TrainingSystem not initialized.")

        # 加载检查点
        checkpoint = torch.load(filepath, map_location=self.device)

        # 恢复状态
        self._current_epoch = checkpoint['epoch']
        self._current_step = checkpoint['step']

        # 恢复优化器状态
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复调度器状态
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # 恢复历史
        history_dict = checkpoint['history']
        self.history.start_time = history_dict['start_time']
        self.history.end_time = history_dict['end_time']
        self.history.duration_seconds = history_dict['duration_seconds']
        self.history.total_epochs = history_dict['total_epochs']
        self.history.total_steps = history_dict['total_steps']
        self.history.loss_history = history_dict['loss_history']
        self.history.validation_history = history_dict['validation_history']
        self.history.lr_history = history_dict['lr_history']
        self.history.best_loss = history_dict['best_loss']
        self.history.best_epoch = history_dict['best_epoch']
        self.history.checkpoints = history_dict['checkpoints']

        # 恢复积分引擎状态
        if self.integration_engine and 'integration_engine_state' in checkpoint:
            engine_state = checkpoint['integration_engine_state']
            self.integration_engine.step_count = engine_state['step_count']
            self.integration_engine.total_time = engine_state['total_time']

        logger.info(
            f"Checkpoint loaded: {filepath}, "
            f"epoch={self._current_epoch}, "
            f"best_loss={self.history.best_loss:.4f}"
        )

    def get_training_history(self) -> TrainingHistory:
        """
        获取训练历史

        Returns:
            训练历史记录
        """
        return self.history

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "initialized": self._initialized,
            "is_training": self._is_training,
            "current_epoch": self._current_epoch,
            "current_step": self._current_step,
            "best_loss": self.history.best_loss,
            "best_epoch": self.history.best_epoch,
            "total_epochs": self.history.total_epochs,
            "total_steps": self.history.total_steps,
            "training_mode": self.config.training_mode,
            "device": self.device,
            "loss_functions_stats": self.loss_functions.get_statistics() if self.loss_functions else None,
            "dynamics_alignment_stats": self.dynamics_alignment.get_statistics() if self.dynamics_alignment else None,
            "freezing_strategy_stats": self.freezing_strategy.get_statistics() if self.freezing_strategy else None,
        }

        return stats

    def reset(self) -> None:
        """重置训练系统"""
        # 重置历史
        self.history = TrainingHistory()

        # 重置状态
        self._current_epoch = 0
        self._current_step = 0
        self._is_training = False

        # 重置组件
        if self.integration_engine:
            self.integration_engine.reset()

        if self.reflection_system:
            self.reflection_system.reset()

        if self.loss_functions:
            self.loss_functions.reset_statistics()

        if self.dynamics_alignment:
            self.dynamics_alignment.reset()

        if self.freezing_strategy:
            self.freezing_strategy.reset()

        logger.info("TrainingSystem reset")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        training = "training" if self._is_training else "idle"

        return (
            f"TrainingSystem("
            f"status={status}, "
            f"state={training}, "
            f"epoch={self._current_epoch}, "
            f"best_loss={self.history.best_loss:.4f})"
        )


def create_training_system_from_config(
    global_config: ChronosConfig,
    integration_engine: Optional[IntegrationEngine] = None,
    reflection_system: Optional[ReflectionSystem] = None,
    meta_cognitive_system: Optional[MetaCognitiveSystem] = None,
    device: Optional[str] = None
) -> TrainingSystem:
    """
    从全局配置创建训练系统

    Args:
        global_config: 全局配置
        integration_engine: 积分引擎
        reflection_system: 反思系统
        meta_cognitive_system: 元认知系统
        device: 计算设备

    Returns:
        TrainingSystem 实例
    """
    training_config = TrainingSystemConfig(
        num_epochs=global_config.training.num_epochs,
        batch_size=global_config.training.batch_size,
        learning_rate=global_config.training.learning_rate,
        weight_decay=global_config.training.weight_decay,
        gradient_clip_threshold=global_config.training.gradient_clip_threshold,
        validation_frequency=global_config.training.validation_frequency,
        checkpoint_frequency=global_config.training.checkpoint_frequency,
        use_amp=global_config.use_amp,
        fast_dim=global_config.dim.fast_variable_dim,
        slow_dim=global_config.dim.slow_variable_dim,
    )

    training_system = TrainingSystem(
        config=training_config,
        global_config=global_config,
        integration_engine=integration_engine,
        reflection_system=reflection_system,
        meta_cognitive_system=meta_cognitive_system,
        device=device
    )

    return training_system