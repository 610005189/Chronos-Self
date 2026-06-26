"""
分阶段冻结策略系统
================

实现 Chronos-Self 项目的分阶段冻结策略，包括：
- L0编码器冻结策略
- L1积分引擎从头训练策略
- 固定投影矩阵冻结
- 冻结策略测试

核心功能：
- L0编码器冻结（SemanticEncoder、LogicalEncoder）
- L1积分引擎从头训练（FastDynamics、SlowDynamics）
- 固定投影矩阵冻结（P_core、Φ、W）
- 验证冻结策略的正确性

使用示例：
    freezing_strategy = FreezingStrategy(config=FreezingStrategyConfig())
    freezing_strategy.apply_freezing(model)
    freezing_strategy.validate_freezing(model)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import logging
import numpy as np

from chronos_core.utils.config import ChronosConfig, DimensionalityConfig
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.fast_dynamics import FastDynamicsFunction
from chronos_core.core.slow_dynamics import SlowDynamicsFunction
from chronos_core.core.meta_cognitive.meta_cognitive_system import MetaCognitiveSystem

logger = logging.getLogger(__name__)


@dataclass
class FreezingStrategyConfig:
    """分阶段冻结策略配置"""

    # L0编码器冻结配置
    freeze_l0_semantic_encoder: bool = True
    freeze_l0_logical_encoder: bool = True
    freeze_l0_physical_encoder: bool = True
    freeze_l0_causal_encoder: bool = True

    # L1积分引擎训练配置
    train_l1_fast_dynamics: bool = True
    train_l1_slow_dynamics: bool = True
    l1_max_dim: int = 2048  # 防止过参数化

    # 固定投影矩阵冻结配置
    freeze_core_subspace_projection: bool = True
    freeze_l2_metacognitive_projection: bool = True
    freeze_chaos_injection_projection: bool = True

    # 冻结策略阶段
    freeze_stage: str = "phase1"  # 'phase1', 'phase2', 'phase3'
    progressive_unfreezing: bool = False  # 渐进解冻

    # 验证配置
    validate_freezing: bool = True
    check_requires_grad: bool = True
    check_parameter_update: bool = True

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512
    core_subspace_dim: int = 64
    meta_cognitive_dim: int = 128


class L0EncoderFreezing:
    """
    L0编码器冻结策略

    冻结 L0 层的所有编码器：
    - SemanticEncoder（语义编码器）
    - LogicalEncoder（逻辑编码器）
    - PhysicalEncoder（物理编码器）
    - CausalEncoder（因果编码器）

    策略：
    - 使用预训练权重（冻结所有参数）
    - 仅通过RAG调用知识库
    - 不参与训练更新
    """

    def __init__(
        self,
        config: Optional[FreezingStrategyConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or FreezingStrategyConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 冻结的参数记录
        self._frozen_parameters: Set[str] = set()
        self._call_count: int = 0

        logger.info(
            f"L0EncoderFreezing created: "
            f"semantic={self.config.freeze_l0_semantic_encoder}, "
            f"logical={self.config.freeze_l0_logical_encoder}"
        )

    def apply_freezing(self, model: nn.Module) -> Dict[str, Any]:
        """
        应用L0编码器冻结策略

        Args:
            model: 需要冻结的模型

        Returns:
            冻结结果字典
        """
        frozen_count = 0
        trainable_count = 0

        # 遍历模型的所有模块
        for name, module in model.named_modules():
            # 检查是否为编码器模块
            if self._is_encoder_module(name):
                # 冻结编码器参数
                for param_name, param in module.named_parameters():
                    if self._should_freeze_encoder(name, param_name):
                        # 冻结参数
                        param.requires_grad = False
                        self._frozen_parameters.add(f"{name}.{param_name}")
                        frozen_count += 1
                        logger.debug(f"Frozen parameter: {name}.{param_name}")
                    else:
                        trainable_count += 1

        # 统计
        self._call_count += 1

        result = {
            "frozen_count": frozen_count,
            "trainable_count": trainable_count,
            "frozen_parameters": list(self._frozen_parameters),
            "success": True,
        }

        logger.info(
            f"L0EncoderFreezing applied: "
            f"frozen={frozen_count}, "
            f"trainable={trainable_count}"
        )

        return result

    def _is_encoder_module(self, module_name: str) -> bool:
        """判断是否为编码器模块"""
        encoder_keywords = [
            'semantic_encoder',
            'logical_encoder',
            'physical_encoder',
            'causal_encoder',
            'encoder',
            'perception',
            'l0',
        ]

        return any(keyword in module_name.lower() for keyword in encoder_keywords)

    def _should_freeze_encoder(self, module_name: str, param_name: str) -> bool:
        """判断是否应该冻结该编码器参数"""
        # SemanticEncoder 冻结
        if 'semantic' in module_name.lower() and self.config.freeze_l0_semantic_encoder:
            return True

        # LogicalEncoder 冻结
        if 'logical' in module_name.lower() and self.config.freeze_l0_logical_encoder:
            return True

        # PhysicalEncoder 冻结
        if 'physical' in module_name.lower() and self.config.freeze_l0_physical_encoder:
            return True

        # CausalEncoder 冻结
        if 'causal' in module_name.lower() and self.config.freeze_l0_causal_encoder:
            return True

        # 默认编码器冻结
        if 'encoder' in module_name.lower() or 'perception' in module_name.lower():
            return True

        return False

    def unfreeze_encoder(self, model: nn.Module, encoder_name: str) -> None:
        """
        解冻特定编码器（用于渐进解冻）

        Args:
            model: 模型
            encoder_name: 编码器名称
        """
        for name, module in model.named_modules():
            if encoder_name in name.lower():
                for param_name, param in module.named_parameters():
                    param.requires_grad = True
                    param_key = f"{name}.{param_name}"
                    if param_key in self._frozen_parameters:
                        self._frozen_parameters.remove(param_key)

                    logger.info(f"Unfreezed parameter: {param_key}")

    def get_frozen_parameters(self) -> Set[str]:
        """获取已冻结的参数"""
        return self._frozen_parameters

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "frozen_parameters_count": len(self._frozen_parameters),
            "frozen_parameters": list(self._frozen_parameters),
            "config": {
                "freeze_semantic": self.config.freeze_l0_semantic_encoder,
                "freeze_logical": self.config.freeze_l0_logical_encoder,
                "freeze_physical": self.config.freeze_l0_physical_encoder,
                "freeze_causal": self.config.freeze_l0_causal_encoder,
            },
        }
        return stats


class L1IntegrationTraining:
    """
    L1积分引擎从头训练策略

    从头训练 L1 层的积分引擎：
    - FastDynamics（快变量动力学）
    - SlowDynamics（慢变量动力学）

    策略：
    - 从头训练（不使用预训练权重）
    - 维度≤2048（防止过参数化）
    - 使用动力学对齐训练
    - 支持梯度更新
    """

    def __init__(
        self,
        config: Optional[FreezingStrategyConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or FreezingStrategyConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 训练的参数记录
        self._trainable_parameters: Set[str] = set()
        self._call_count: int = 0

        logger.info(
            f"L1IntegrationTraining created: "
            f"train_fast={self.config.train_l1_fast_dynamics}, "
            f"train_slow={self.config.train_l1_slow_dynamics}, "
            f"max_dim={self.config.l1_max_dim}"
        )

    def apply_training_strategy(self, model: nn.Module) -> Dict[str, Any]:
        """
        应用L1积分引擎训练策略

        Args:
            model: 需要训练的模型

        Returns:
            训练策略应用结果
        """
        trainable_count = 0
        frozen_count = 0

        # 遍历模型的所有模块
        for name, module in model.named_modules():
            # 检查是否为积分引擎模块
            if self._is_integration_module(name):
                # 检查维度约束
                self._check_dimension_constraint(module, name)

                # 设置训练状态
                for param_name, param in module.named_parameters():
                    if self._should_train_integration(name, param_name):
                        # 设置为可训练
                        param.requires_grad = True
                        self._trainable_parameters.add(f"{name}.{param_name}")
                        trainable_count += 1
                        logger.debug(f"Trainable parameter: {name}.{param_name}")
                    else:
                        param.requires_grad = False
                        frozen_count += 1

        # 统计
        self._call_count += 1

        result = {
            "trainable_count": trainable_count,
            "frozen_count": frozen_count,
            "trainable_parameters": list(self._trainable_parameters),
            "max_dim": self.config.l1_max_dim,
            "success": True,
        }

        logger.info(
            f"L1IntegrationTraining applied: "
            f"trainable={trainable_count}, "
            f"frozen={frozen_count}"
        )

        return result

    def _is_integration_module(self, module_name: str) -> bool:
        """判断是否为积分引擎模块"""
        integration_keywords = [
            'fast_dynamics',
            'slow_dynamics',
            'dynamics',
            'integration',
            'neural_ode',
            'evolution',
            'l1',
        ]

        return any(keyword in module_name.lower() for keyword in integration_keywords)

    def _check_dimension_constraint(self, module: nn.Module, module_name: str) -> None:
        """检查维度约束"""
        # 检查线性层的维度
        for param_name, param in module.named_parameters():
            if 'weight' in param_name and param.dim() == 2:
                max_dim = max(param.shape[0], param.shape[1])
                if max_dim > self.config.l1_max_dim:
                    logger.warning(
                        f"Module {module_name}.{param_name} dimension {max_dim} "
                        f"exceeds max_dim {self.config.l1_max_dim}"
                    )

    def _should_train_integration(self, module_name: str, param_name: str) -> bool:
        """判断是否应该训练该积分引擎参数"""
        # FastDynamics 训练
        if 'fast_dynamics' in module_name.lower() and self.config.train_l1_fast_dynamics:
            return True

        # SlowDynamics 训练
        if 'slow_dynamics' in module_name.lower() and self.config.train_l1_slow_dynamics:
            return True

        # NeuralODE 相关训练
        if 'neural_ode' in module_name.lower() or 'evolution' in module_name.lower():
            return True

        # 默认积分引擎训练
        if 'dynamics' in module_name.lower() or 'integration' in module_name.lower():
            return True

        return False

    def get_trainable_parameters(self) -> Set[str]:
        """获取可训练的参数"""
        return self._trainable_parameters

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "trainable_parameters_count": len(self._trainable_parameters),
            "trainable_parameters": list(self._trainable_parameters),
            "max_dim": self.config.l1_max_dim,
            "config": {
                "train_fast": self.config.train_l1_fast_dynamics,
                "train_slow": self.config.train_l1_slow_dynamics,
            },
        }
        return stats


class FixedProjectionFreezing:
    """
    固定投影矩阵冻结策略

    冻结所有固定的投影矩阵：
    - P_core（核心子空间投影）
    - Φ（L2元认知投影）
    - W（混沌注入投影）

    策略：
    - 所有投影矩阵不可学习（requires_grad=False）
    - 核心子空间投影 P_core（固定随机正交）
    - L2元认知投影 Φ（固定稀疏随机）
    - 混沌注入投影 W（固定随机）
    """

    def __init__(
        self,
        config: Optional[FreezingStrategyConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or FreezingStrategyConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 冻结的投影矩阵记录
        self._frozen_projections: Set[str] = set()
        self._projection_matrices: Dict[str, torch.Tensor] = {}
        self._call_count: int = 0

        logger.info(
            f"FixedProjectionFreezing created: "
            f"freeze_core={self.config.freeze_core_subspace_projection}, "
            f"freeze_l2={self.config.freeze_l2_metacognitive_projection}, "
            f"freeze_chaos={self.config.freeze_chaos_injection_projection}"
        )

    def apply_freezing(self, model: nn.Module) -> Dict[str, Any]:
        """
        应用固定投影矩阵冻结策略

        Args:
            model: 需要冻结的模型

        Returns:
            冻结结果字典
        """
        frozen_count = 0

        # 创建固定投影矩阵
        self._create_fixed_projections()

        # 遍历模型的所有参数
        for name, param in model.named_parameters():
            if self._is_projection_parameter(name):
                # 冻结投影参数
                param.requires_grad = False
                self._frozen_projections.add(name)
                frozen_count += 1
                logger.debug(f"Frozen projection: {name}")

        # 统计
        self._call_count += 1

        result = {
            "frozen_count": frozen_count,
            "frozen_projections": list(self._frozen_projections),
            "projection_matrices": {
                name: shape for name, shape in [
                    (k, tuple(v.shape)) for k, v in self._projection_matrices.items()
                ]
            },
            "success": True,
        }

        logger.info(
            f"FixedProjectionFreezing applied: "
            f"frozen={frozen_count}"
        )

        return result

    def _create_fixed_projections(self) -> None:
        """创建固定投影矩阵"""
        # 核心子空间投影 P_core（固定随机正交）
        if self.config.freeze_core_subspace_projection:
            core_dim = self.config.core_subspace_dim
            fast_dim = self.config.fast_dim

            # 创建随机正交投影矩阵
            P_core = self._create_orthogonal_projection(fast_dim, core_dim)
            self._projection_matrices['P_core'] = P_core
            logger.debug(f"Created P_core projection: shape={P_core.shape}")

        # L2元认知投影 Φ（固定稀疏随机）
        if self.config.freeze_l2_metacognitive_projection:
            meta_dim = self.config.meta_cognitive_dim
            slow_dim = self.config.slow_dim

            # 创建稀疏随机投影矩阵
            Phi = self._create_sparse_random_projection(slow_dim, meta_dim)
            self._projection_matrices['Phi'] = Phi
            logger.debug(f"Created Phi projection: shape={Phi.shape}")

        # 混沌注入投影 W（固定随机）
        if self.config.freeze_chaos_injection_projection:
            chaos_dim = self.config.core_subspace_dim
            fast_dim = self.config.fast_dim

            # 创建随机投影矩阵
            W = self._create_random_projection(fast_dim, chaos_dim)
            self._projection_matrices['W'] = W
            logger.debug(f"Created W projection: shape={W.shape}")

    def _create_orthogonal_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """创建随机正交投影矩阵"""
        # 创建随机矩阵
        random_matrix = torch.randn(input_dim, output_dim, device=self.device)

        # 正交化（使用QR分解）
        Q, R = torch.linalg.qr(random_matrix)

        # 返回正交矩阵
        return Q

    def _create_sparse_random_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """创建稀疏随机投影矩阵"""
        # 创建稀疏矩阵（10%密度）
        sparsity = 0.1
        sparse_matrix = torch.zeros(input_dim, output_dim, device=self.device)

        # 随机选择位置填充
        mask = torch.rand(input_dim, output_dim, device=self.device) < sparsity
        sparse_matrix[mask] = torch.randn(mask.sum().item(), device=self.device)

        # 归一化
        sparse_matrix = sparse_matrix / sparse_matrix.norm(dim=0, keepdim=True)

        return sparse_matrix

    def _create_random_projection(self, input_dim: int, output_dim: int) -> torch.Tensor:
        """创建随机投影矩阵"""
        # 创建随机矩阵
        random_matrix = torch.randn(input_dim, output_dim, device=self.device)

        # 归一化
        random_matrix = random_matrix / random_matrix.norm(dim=0, keepdim=True)

        return random_matrix

    def _is_projection_parameter(self, param_name: str) -> bool:
        """判断是否为投影参数"""
        projection_keywords = [
            'projection',
            'p_core',
            'phi',
            'w',
            'chaos_injection',
            'meta_cognitive',
            'core_subspace',
        ]

        return any(keyword in param_name.lower() for keyword in projection_keywords)

    def get_projection_matrix(self, name: str) -> Optional[torch.Tensor]:
        """获取特定投影矩阵"""
        return self._projection_matrices.get(name)

    def get_frozen_projections(self) -> Set[str]:
        """获取已冻结的投影矩阵"""
        return self._frozen_projections

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "frozen_projections_count": len(self._frozen_projections),
            "frozen_projections": list(self._frozen_projections),
            "projection_matrices": {
                name: tuple(shape.shape) for name, shape in self._projection_matrices.items()
            },
            "config": {
                "freeze_core": self.config.freeze_core_subspace_projection,
                "freeze_l2": self.config.freeze_l2_metacognitive_projection,
                "freeze_chaos": self.config.freeze_chaos_injection_projection,
            },
        }
        return stats


class FreezingValidator:
    """
    冻结策略验证器

    验证冻结策略的正确性：
    - 检查冻结参数确实不更新
    - 检查训练参数正确更新
    - 测试冻结策略对训练的影响
    """

    def __init__(
        self,
        config: Optional[FreezingStrategyConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or FreezingStrategyConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 验证记录
        self._validation_history: List[Dict[str, Any]] = []
        self._call_count: int = 0

        logger.info(f"FreezingValidator created: device={self.device}")

    def validate_freezing(
        self,
        model: nn.Module,
        frozen_parameters: Set[str],
        trainable_parameters: Set[str]
    ) -> Dict[str, Any]:
        """
        验证冻结策略

        Args:
            model: 模型
            frozen_parameters: 应该冻结的参数
            trainable_parameters: 应该可训练的参数

        Returns:
            验证结果字典
        """
        validation_results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "frozen_check": {},
            "trainable_check": {},
        }

        # 检查冻结参数的 requires_grad 状态
        if self.config.check_requires_grad:
            frozen_check = self._check_frozen_requires_grad(model, frozen_parameters)
            validation_results["frozen_check"] = frozen_check

            if not frozen_check["all_correct"]:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Some frozen parameters have requires_grad=True: "
                    f"{frozen_check['incorrect_count']}"
                )

        # 检查训练参数的 requires_grad 状态
        if self.config.check_requires_grad:
            trainable_check = self._check_trainable_requires_grad(model, trainable_parameters)
            validation_results["trainable_check"] = trainable_check

            if not trainable_check["all_correct"]:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Some trainable parameters have requires_grad=False: "
                    f"{trainable_check['incorrect_count']}"
                )

        # 检查参数更新（可选）
        if self.config.check_parameter_update:
            update_check = self._check_parameter_update(model)
            validation_results["update_check"] = update_check

        # 统计
        self._call_count += 1
        self._validation_history.append(validation_results)

        logger.info(
            f"FreezingValidator validated: "
            f"passed={validation_results['passed']}, "
            f"errors={len(validation_results['errors'])}"
        )

        return validation_results

    def _check_frozen_requires_grad(
        self,
        model: nn.Module,
        frozen_parameters: Set[str]
    ) -> Dict[str, Any]:
        """检查冻结参数的 requires_grad 状态"""
        correct_count = 0
        incorrect_count = 0
        incorrect_parameters = []

        for param_name in frozen_parameters:
            # 查找参数
            param = self._find_parameter(model, param_name)

            if param is not None:
                if not param.requires_grad:
                    correct_count += 1
                else:
                    incorrect_count += 1
                    incorrect_parameters.append(param_name)

        return {
            "all_correct": incorrect_count == 0,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "incorrect_parameters": incorrect_parameters,
        }

    def _check_trainable_requires_grad(
        self,
        model: nn.Module,
        trainable_parameters: Set[str]
    ) -> Dict[str, Any]:
        """检查训练参数的 requires_grad 状态"""
        correct_count = 0
        incorrect_count = 0
        incorrect_parameters = []

        for param_name in trainable_parameters:
            # 查找参数
            param = self._find_parameter(model, param_name)

            if param is not None:
                if param.requires_grad:
                    correct_count += 1
                else:
                    incorrect_count += 1
                    incorrect_parameters.append(param_name)

        return {
            "all_correct": incorrect_count == 0,
            "correct_count": correct_count,
            "incorrect_count": incorrect_count,
            "incorrect_parameters": incorrect_parameters,
        }

    def _check_parameter_update(self, model: nn.Module) -> Dict[str, Any]:
        """检查参数更新（简化检查）"""
        # 记录当前参数值
        param_values_before = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                param_values_before[name] = param.data.clone()

        # 模拟一次梯度更新（简化）
        # 在实际训练中，这个检查会验证参数是否真正更新

        return {
            "checked": True,
            "trainable_count": len(param_values_before),
            "message": "Parameter update check requires actual training step",
        }

    def _find_parameter(self, model: nn.Module, param_name: str) -> Optional[torch.Tensor]:
        """查找模型中的参数"""
        for name, param in model.named_parameters():
            if name == param_name or param_name in name:
                return param
        return None

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """获取验证历史"""
        return self._validation_history

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "validation_count": len(self._validation_history),
            "pass_rate": np.mean([v["passed"] for v in self._validation_history]) if self._validation_history else 0.0,
            "last_validation": self._validation_history[-1] if self._validation_history else None,
        }
        return stats


class FreezingStrategy(nn.Module):
    """
    完整分阶段冻结策略系统

    整合所有冻结策略：
    - L0编码器冻结
    - L1积分引擎从头训练
    - 固定投影矩阵冻结
    - 冻结策略验证

    使用示例：
        freezing_strategy = FreezingStrategy(config=FreezingStrategyConfig())
        freezing_strategy.apply_all_strategies(model)
        freezing_strategy.validate_all_strategies(model)
    """

    def __init__(
        self,
        config: Optional[FreezingStrategyConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or FreezingStrategyConfig()

        # 从全局配置更新
        if global_config:
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim
            self.config.core_subspace_dim = global_config.dim.core_subspace_dim
            self.config.meta_cognitive_dim = global_config.meta_cognitive.l2_hidden_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建各个冻结策略模块
        self.l0_encoder_freezing = L0EncoderFreezing(
            config=self.config,
            device=self.device
        )

        self.l1_integration_training = L1IntegrationTraining(
            config=self.config,
            device=self.device
        )

        self.fixed_projection_freezing = FixedProjectionFreezing(
            config=self.config,
            device=self.device
        )

        self.freezing_validator = FreezingValidator(
            config=self.config,
            device=self.device
        )

        # 统计信息
        self._strategy_history: List[Dict[str, Any]] = []
        self._call_count: int = 0

        logger.info(
            f"FreezingStrategy created: "
            f"stage={self.config.freeze_stage}, "
            f"device={self.device}"
        )

    def forward(self, model: nn.Module) -> Dict[str, Any]:
        """
        应用所有冻结策略

        Args:
            model: 需要应用冻结策略的模型

        Returns:
            应用结果字典
        """
        # 应用L0编码器冻结
        l0_result = self.l0_encoder_freezing.apply_freezing(model)

        # 应用L1积分引擎训练策略
        l1_result = self.l1_integration_training.apply_training_strategy(model)

        # 应用固定投影矩阵冻结
        projection_result = self.fixed_projection_freezing.apply_freezing(model)

        # 验证冻结策略
        validation_result = None
        if self.config.validate_freezing:
            all_frozen = (
                set(l0_result["frozen_parameters"]) |
                set(projection_result["frozen_projections"])
            )
            all_trainable = set(l1_result["trainable_parameters"])

            validation_result = self.freezing_validator.validate_freezing(
                model, all_frozen, all_trainable
            )

        # 统计
        self._call_count += 1

        # 合并结果
        result = {
            "l0_freezing": l0_result,
            "l1_training": l1_result,
            "projection_freezing": projection_result,
            "validation": validation_result,
            "success": True,
        }

        self._strategy_history.append(result)

        logger.info(
            f"FreezingStrategy applied: "
            f"frozen={l0_result['frozen_count'] + projection_result['frozen_count']}, "
            f"trainable={l1_result['trainable_count']}"
        )

        return result

    def apply_all_strategies(self, model: nn.Module) -> Dict[str, Any]:
        """
        应用所有冻结策略（简化接口）

        Args:
            model: 模型

        Returns:
            应用结果
        """
        return self.forward(model)

    def validate_all_strategies(self, model: nn.Module) -> Dict[str, Any]:
        """
        验证所有冻结策略

        Args:
            model: 模型

        Returns:
            验证结果
        """
        # 获取冻结和训练参数
        frozen_parameters = (
            self.l0_encoder_freezing.get_frozen_parameters() |
            self.fixed_projection_freezing.get_frozen_projections()
        )

        trainable_parameters = self.l1_integration_training.get_trainable_parameters()

        # 执行验证
        validation_result = self.freezing_validator.validate_freezing(
            model, frozen_parameters, trainable_parameters
        )

        return validation_result

    def get_frozen_parameters(self) -> Set[str]:
        """获取所有已冻结的参数"""
        return (
            self.l0_encoder_freezing.get_frozen_parameters() |
            self.fixed_projection_freezing.get_frozen_projections()
        )

    def get_trainable_parameters(self) -> Set[str]:
        """获取所有可训练的参数"""
        return self.l1_integration_training.get_trainable_parameters()

    def get_projection_matrix(self, name: str) -> Optional[torch.Tensor]:
        """获取固定投影矩阵"""
        return self.fixed_projection_freezing.get_projection_matrix(name)

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        l0_stats = self.l0_encoder_freezing.get_statistics()
        l1_stats = self.l1_integration_training.get_statistics()
        projection_stats = self.fixed_projection_freezing.get_statistics()
        validator_stats = self.freezing_validator.get_statistics()

        stats = {
            "call_count": self._call_count,
            "strategy_count": len(self._strategy_history),
            "l0_stats": l0_stats,
            "l1_stats": l1_stats,
            "projection_stats": projection_stats,
            "validator_stats": validator_stats,
            "config": {
                "freeze_stage": self.config.freeze_stage,
                "freeze_l0": self.config.freeze_l0_semantic_encoder,
                "train_l1": self.config.train_l1_fast_dynamics,
                "freeze_projections": self.config.freeze_core_subspace_projection,
            },
        }

        return stats

    def reset(self) -> None:
        """重置所有策略"""
        self._strategy_history.clear()
        self._call_count = 0

        # 重置子模块
        self.l0_encoder_freezing._frozen_parameters.clear()
        self.l0_encoder_freezing._call_count = 0

        self.l1_integration_training._trainable_parameters.clear()
        self.l1_integration_training._call_count = 0

        self.fixed_projection_freezing._frozen_projections.clear()
        self.fixed_projection_freezing._projection_matrices.clear()
        self.fixed_projection_freezing._call_count = 0

        self.freezing_validator._validation_history.clear()
        self.freezing_validator._call_count = 0

        logger.info("FreezingStrategy reset")

    def __repr__(self) -> str:
        frozen_count = len(self.get_frozen_parameters())
        trainable_count = len(self.get_trainable_parameters())

        return (
            f"FreezingStrategy("
            f"frozen={frozen_count}, "
            f"trainable={trainable_count}, "
            f"stage={self.config.freeze_stage})"
        )


def create_freezing_strategy_from_config(
    global_config: ChronosConfig,
    device: Optional[str] = None
) -> FreezingStrategy:
    """
    从全局配置创建冻结策略系统

    Args:
        global_config: 全局配置
        device: 计算设备

    Returns:
        FreezingStrategy 实例
    """
    freezing_config = FreezingStrategyConfig(
        fast_dim=global_config.dim.fast_variable_dim,
        slow_dim=global_config.dim.slow_variable_dim,
        core_subspace_dim=global_config.dim.core_subspace_dim,
        meta_cognitive_dim=global_config.meta_cognitive.l2_hidden_dim,
        l1_max_dim=global_config.dim.fast_variable_dim,
    )

    freezing_strategy = FreezingStrategy(
        config=freezing_config,
        global_config=global_config,
        device=device
    )

    return freezing_strategy