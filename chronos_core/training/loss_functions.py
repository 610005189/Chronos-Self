"""
训练损失函数系统
================

实现 Chronos-Self 项目的训练损失函数，包括：
- 预测损失（生存任务）
- 抗寂灭损失（自预测准确率）
- 惯性正则损失
- 完整损失组合（带权重λ, μ）

核心功能：
- L_pred = ||实际观测(t+1) - 预测(t+1)||²
- L_anti_decay = E[||E(t+Δt) - Ē(t+Δt)||²]
- L_inertia = ||Ė||²
- L = L_pred + λ·L_anti_decay + μ·L_inertia

使用示例：
    loss_system = LossFunctions(config=LossFunctionsConfig())
    losses = loss_system.compute_all_losses(
        actual_state, predicted_state,
        self_prediction_error, state_derivative
    )
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import numpy as np

from chronos_core.utils.config import ChronosConfig, CouplingStabilityConfig
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput

logger = logging.getLogger(__name__)


@dataclass
class LossFunctionsConfig:
    """损失函数配置"""

    # 抗寂灭权重 λ（防止状态坍塌，维持负熵）
    anti_quietus_weight: float = 0.01

    # 惯性权重 μ（维持状态连续性）
    inertia_weight: float = 0.001

    # 预测损失权重（生存任务）
    prediction_weight: float = 1.0

    # 损失计算模式
    loss_mode: str = "combined"  # 'combined', 'individual', 'weighted'

    # 数值稳定性参数
    min_loss_value: float = 1e-8
    max_loss_value: float = 1e6
    loss_clip_threshold: float = 100.0

    # 损失归一化
    normalize_losses: bool = True
    normalization_method: str = "l2"  # 'l1', 'l2', 'max'

    # 自预测误差计算
    self_prediction_window: int = 10  # 自预测窗口大小
    use_rolling_average: bool = True  # 使用滚动平均

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


class PredictionLoss(nn.Module):
    """
    预测损失（生存任务）

    计算系统预测外部输入的准确度：
        L_pred = ||实际观测(t+1) - 预测(t+1)||²

    这是生存任务的核心损失，确保系统能够准确预测外部输入。
    """

    def __init__(
        self,
        config: Optional[LossFunctionsConfig] = None,
        dim_config: Optional[Any] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or LossFunctionsConfig()
        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 统计信息
        self._loss_history: List[float] = []
        self._call_count: int = 0

        logger.info(
            f"PredictionLoss created: "
            f"fast_dim={self.config.fast_dim}, "
            f"slow_dim={self.config.slow_dim}"
        )

    def forward(
        self,
        actual_state: SelfState,
        predicted_state: SelfState,
        actual_input: Optional[ExternalInput] = None,
        predicted_input: Optional[ExternalInput] = None
    ) -> torch.Tensor:
        """
        计算预测损失

        Args:
            actual_state: 实际观测状态（t+1时刻）
            predicted_state: 预测状态（t+1时刻）
            actual_input: 实际外部输入（可选）
            predicted_input: 预测外部输入（可选）

        Returns:
            预测损失值
        """
        # 确保状态在设备上
        E_fast_actual = actual_state.E_fast.to(self.device)
        E_fast_pred = predicted_state.E_fast.to(self.device)

        E_slow_actual = actual_state.E_slow.to(self.device)
        E_slow_pred = predicted_state.E_slow.to(self.device)

        # 计算快变量预测误差
        fast_error = torch.norm(E_fast_actual - E_fast_pred, p=2)

        # 计算慢变量预测误差（权重较低）
        slow_error = torch.norm(E_slow_actual - E_slow_pred, p=2) * 0.1

        # 合并快慢变量误差
        state_prediction_loss = fast_error + slow_error

        # 如果有外部输入，计算输入预测误差
        input_prediction_loss = 0.0
        if actual_input is not None and predicted_input is not None:
            # 语义输入预测误差
            if actual_input.X_sem is not None and predicted_input.X_sem is not None:
                sem_actual = actual_input.X_sem.to(self.device)
                sem_pred = predicted_input.X_sem.to(self.device)
                sem_error = torch.norm(sem_actual - sem_pred, p=2)
                input_prediction_loss += sem_error

            # 物理输入预测误差
            if actual_input.X_log is not None and predicted_input.X_log is not None:
                log_actual = actual_input.X_log.to(self.device)
                log_pred = predicted_input.X_log.to(self.device)
                log_error = torch.norm(log_actual - log_pred, p=2)
                input_prediction_loss += log_error

        # 合并状态和输入预测损失
        total_loss = state_prediction_loss + input_prediction_loss * 0.5

        # 损失裁剪（数值稳定性）
        total_loss = torch.clamp(
            total_loss,
            min=self.config.min_loss_value,
            max=self.config.max_loss_value
        )

        # 统计
        self._call_count += 1
        self._loss_history.append(total_loss.item())

        if len(self._loss_history) > 1000:
            self._loss_history = self._loss_history[-1000:]

        logger.debug(
            f"PredictionLoss computed: "
            f"fast_error={fast_error.item():.4f}, "
            f"slow_error={slow_error.item():.4f}, "
            f"total_loss={total_loss.item():.4f}"
        )

        return total_loss

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._loss_history),
            "mean_loss": np.mean(self._loss_history) if self._loss_history else 0.0,
            "std_loss": np.std(self._loss_history) if self._loss_history else 0.0,
            "min_loss": np.min(self._loss_history) if self._loss_history else 0.0,
            "max_loss": np.max(self._loss_history) if self._loss_history else 0.0,
        }
        return stats


class AntiDecayLoss(nn.Module):
    """
    抗寂灭损失（自预测准确率）

    计算系统的自预测能力：
        L_anti_decay = E[||E(t+Δt) - Ē(t+Δt)||²]

    这是负熵维持的核心机制：
    - 系统越能预测自身演化，内部结构越稳定
    - 自预测准确率越高，负熵（内部秩序）越强
    - 这是真正的"自我维持"，而非简单的外部依赖

    物理意义：
    - 熵寂灭：系统无法预测自身，内部结构趋于混乱
    - 负熵维持：系统通过自预测维持内部秩序
    - 自指闭环：E(t) → 预测 → Ē(t+Δt) → 反馈 → E(t+Δt)
    """

    def __init__(
        self,
        config: Optional[LossFunctionsConfig] = None,
        dim_config: Optional[Any] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or LossFunctionsConfig()
        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 自预测误差历史（用于计算期望）
        self._self_prediction_errors: List[float] = []
        self._call_count: int = 0

        # 滚动窗口（用于计算平均误差）
        self._rolling_window: List[torch.Tensor] = []
        self._window_size = self.config.self_prediction_window

        logger.info(
            f"AntiDecayLoss created: "
            f"window_size={self._window_size}, "
            f"device={self.device}"
        )

    def forward(
        self,
        current_state: SelfState,
        self_predicted_state: SelfState,
        dt: Optional[float] = None,
        update_history: bool = True
    ) -> torch.Tensor:
        """
        计算抗寂灭损失

        Args:
            current_state: 当前实际状态 E(t+Δt)
            self_predicted_state: 系统自预测状态 Ē(t+Δt)
            dt: 时间步长（可选）
            update_history: 是否更新历史记录

        Returns:
            抗寂灭损失值
        """
        # 确保状态在设备上
        E_fast_actual = current_state.E_fast.to(self.device)
        E_fast_pred = self_predicted_state.E_fast.to(self.device)

        E_slow_actual = current_state.E_slow.to(self.device)
        E_slow_pred = self_predicted_state.E_slow.to(self.device)

        # 计算自预测误差
        fast_error = torch.norm(E_fast_actual - E_fast_pred, p=2)
        slow_error = torch.norm(E_slow_actual - E_slow_pred, p=2)

        # 合并误差（快变量更重要）
        self_prediction_error = fast_error + slow_error * 0.1

        # 如果使用滚动平均，计算窗口内的平均误差
        if self.config.use_rolling_average and len(self._rolling_window) > 0:
            # 将当前误差添加到窗口
            self._rolling_window.append(self_prediction_error.detach())

            if len(self._rolling_window) > self._window_size:
                self._rolling_window.pop(0)

            # 计算窗口内的平均误差
            window_tensor = torch.stack(self._rolling_window)
            anti_decay_loss = torch.mean(window_tensor)
        else:
            # 直接使用当前误差
            anti_decay_loss = self_prediction_error

        # 损失裁剪
        anti_decay_loss = torch.clamp(
            anti_decay_loss,
            min=self.config.min_loss_value,
            max=self.config.max_loss_value
        )

        # 更新历史记录
        if update_history:
            self._call_count += 1
            self._self_prediction_errors.append(self_prediction_error.item())

            if len(self._self_prediction_errors) > 1000:
                self._self_prediction_errors = self._self_prediction_errors[-1000:]

        logger.debug(
            f"AntiDecayLoss computed: "
            f"fast_error={fast_error.item():.4f}, "
            f"slow_error={slow_error.item():.4f}, "
            f"loss={anti_decay_loss.item():.4f}, "
            f"window_size={len(self._rolling_window)}"
        )

        return anti_decay_loss

    def compute_entropy_metric(self) -> float:
        """
        计算熵相关指标

        使用自预测误差的分布计算熵指标：
        - 低熵：自预测误差稳定，系统内部秩序强
        - 高熵：自预测误差波动大，系统趋于混乱

        Returns:
            熵指标值（0-1范围，越小越好）
        """
        if len(self._self_prediction_errors) < 10:
            return 1.0

        # 计算误差分布的熵
        errors = np.array(self._self_prediction_errors[-100:])
        errors_normalized = errors / (np.max(errors) + 1e-8)

        # 使用方差作为熵的代理指标
        entropy_metric = np.std(errors_normalized)

        return entropy_metric

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._self_prediction_errors),
            "mean_error": np.mean(self._self_prediction_errors) if self._self_prediction_errors else 0.0,
            "std_error": np.std(self._self_prediction_errors) if self._self_prediction_errors else 0.0,
            "rolling_window_size": len(self._rolling_window),
            "entropy_metric": self.compute_entropy_metric(),
        }
        return stats


class InertiaRegularizationLoss(nn.Module):
    """
    惯性正则损失

    计算状态变化的剧烈程度：
        L_inertia = ||Ė||²

    物理意义：
    - 防止状态变化过于剧烈
    - 维持系统的连续性和稳定性
    - 类似物理学中的"惯性"概念
    """

    def __init__(
        self,
        config: Optional[LossFunctionsConfig] = None,
        dim_config: Optional[Any] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or LossFunctionsConfig()
        if dim_config:
            self.config.fast_dim = dim_config.fast_variable_dim
            self.config.slow_dim = dim_config.slow_variable_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 统计信息
        self._derivative_history: List[float] = []
        self._call_count: int = 0

        logger.info(f"InertiaRegularizationLoss created: device={self.device}")

    def forward(
        self,
        state_derivative: torch.Tensor,
        prev_derivative: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算惯性正则损失

        Args:
            state_derivative: 当前状态导数 Ė(t)
            prev_derivative: 前一时刻状态导数（可选，用于计算二阶惯性）

        Returns:
            惯性正则损失值
        """
        # 确保导数在设备上
        state_derivative = state_derivative.to(self.device)

        # 计算导数的范数（一阶惯性）
        derivative_norm = torch.norm(state_derivative, p=2)

        # 计算二阶惯性（如果有前一时刻导数）
        second_order_inertia = 0.0
        if prev_derivative is not None:
            prev_derivative = prev_derivative.to(self.device)
            derivative_change = torch.norm(state_derivative - prev_derivative, p=2)
            second_order_inertia = derivative_change * 0.1

        # 合并一阶和二阶惯性
        inertia_loss = derivative_norm + second_order_inertia

        # 损失裁剪
        inertia_loss = torch.clamp(
            inertia_loss,
            min=self.config.min_loss_value,
            max=self.config.max_loss_value
        )

        # 统计
        self._call_count += 1
        self._derivative_history.append(derivative_norm.item())

        if len(self._derivative_history) > 1000:
            self._derivative_history = self._derivative_history[-1000:]

        logger.debug(
            f"InertiaRegularizationLoss computed: "
            f"derivative_norm={derivative_norm.item():.4f}, "
            f"inertia_loss={inertia_loss.item():.4f}"
        )

        return inertia_loss

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            "call_count": self._call_count,
            "history_length": len(self._derivative_history),
            "mean_derivative": np.mean(self._derivative_history) if self._derivative_history else 0.0,
            "std_derivative": np.std(self._derivative_history) if self._derivative_history else 0.0,
            "max_derivative": np.max(self._derivative_history) if self._derivative_history else 0.0,
        }
        return stats


class LossFunctions(nn.Module):
    """
    完整损失函数系统

    整合所有损失函数：
    - 预测损失（生存任务）
    - 抗寂灭损失（自预测准确率）
    - 惯性正则损失
    - 完整损失组合（带权重λ, μ）

    完整损失：
        L = L_pred + λ·L_anti_decay + μ·L_inertia

    默认权重：
        λ = 0.01（抗寂灭权重）
        μ = 0.001（惯性权重）
    """

    def __init__(
        self,
        config: Optional[LossFunctionsConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        coupling_config: Optional[CouplingStabilityConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or LossFunctionsConfig()

        # 从全局配置更新权重
        if global_config:
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim

        if coupling_config:
            self.config.anti_quietus_weight = coupling_config.anti_quietus_weight
            self.config.inertia_weight = coupling_config.inertia_weight

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 创建各个损失函数
        self.prediction_loss = PredictionLoss(
            config=self.config,
            device=self.device
        )

        self.anti_decay_loss = AntiDecayLoss(
            config=self.config,
            device=self.device
        )

        self.inertia_loss = InertiaRegularizationLoss(
            config=self.config,
            device=self.device
        )

        # 权重
        self.lambda_weight = self.config.anti_quietus_weight
        self.mu_weight = self.config.inertia_weight
        self.prediction_weight = self.config.prediction_weight

        # 统计信息
        self._total_loss_history: List[Dict[str, float]] = []
        self._call_count: int = 0

        logger.info(
            f"LossFunctions created: "
            f"λ={self.lambda_weight}, "
            f"μ={self.mu_weight}, "
            f"prediction_weight={self.prediction_weight}, "
            f"device={self.device}"
        )

    def forward(
        self,
        actual_state: SelfState,
        predicted_state: SelfState,
        self_predicted_state: Optional[SelfState] = None,
        state_derivative: Optional[torch.Tensor] = None,
        prev_derivative: Optional[torch.Tensor] = None,
        actual_input: Optional[ExternalInput] = None,
        predicted_input: Optional[ExternalInput] = None,
        mode: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失

        Args:
            actual_state: 实际观测状态（t+1时刻）
            predicted_state: 预测状态（t+1时刻）
            self_predicted_state: 系统自预测状态（可选）
            state_derivative: 状态导数（可选）
            prev_derivative: 前一时刻状态导数（可选）
            actual_input: 实际外部输入（可选）
            predicted_input: 预测外部输入（可选）
            mode: 损失计算模式（可选）

        Returns:
            损失字典：{
                'total': 总损失,
                'prediction': 预测损失,
                'anti_decay': 抗寂灭损失,
                'inertia': 惯性损失,
                'weights': 权重字典
            }
        """
        # 使用配置的模式或指定模式
        loss_mode = mode or self.config.loss_mode

        # 计算预测损失
        L_pred = self.prediction_loss(
            actual_state, predicted_state,
            actual_input, predicted_input
        )

        # 初始化损失字典
        losses = {
            'prediction': L_pred,
            'anti_decay': torch.tensor(0.0, device=self.device),
            'inertia': torch.tensor(0.0, device=self.device),
        }

        # 计算抗寂灭损失（如果有自预测状态）
        if self_predicted_state is not None:
            L_anti_decay = self.anti_decay_loss(
                actual_state, self_predicted_state
            )
            losses['anti_decay'] = L_anti_decay

        # 计算惯性损失（如果有状态导数）
        if state_derivative is not None:
            L_inertia = self.inertia_loss(
                state_derivative, prev_derivative
            )
            losses['inertia'] = L_inertia

        # 计算总损失
        if loss_mode == 'combined':
            # 完整损失组合：L = L_pred + λ·L_anti_decay + μ·L_inertia
            total_loss = (
                self.prediction_weight * losses['prediction'] +
                self.lambda_weight * losses['anti_decay'] +
                self.mu_weight * losses['inertia']
            )
        elif loss_mode == 'individual':
            # 仅计算预测损失（生存任务）
            total_loss = losses['prediction']
        elif loss_mode == 'weighted':
            # 使用指定权重（已经在各个损失中应用）
            total_loss = losses['prediction'] + losses['anti_decay'] + losses['inertia']
        else:
            # 默认组合模式
            total_loss = (
                self.prediction_weight * losses['prediction'] +
                self.lambda_weight * losses['anti_decay'] +
                self.mu_weight * losses['inertia']
            )

        # 损失裁剪（防止过大损失）
        total_loss = torch.clamp(
            total_loss,
            min=self.config.min_loss_value,
            max=self.config.loss_clip_threshold
        )

        # 添加总损失和权重到字典
        losses['total'] = total_loss
        losses['weights'] = {
            'prediction': self.prediction_weight,
            'anti_decay': self.lambda_weight,
            'inertia': self.mu_weight,
        }

        # 统计
        self._call_count += 1
        self._total_loss_history.append({
            'total': total_loss.item(),
            'prediction': losses['prediction'].item(),
            'anti_decay': losses['anti_decay'].item(),
            'inertia': losses['inertia'].item(),
        })

        if len(self._total_loss_history) > 1000:
            self._total_loss_history = self._total_loss_history[-1000:]

        logger.debug(
            f"LossFunctions computed: "
            f"total={total_loss.item():.4f}, "
            f"prediction={losses['prediction'].item():.4f}, "
            f"anti_decay={losses['anti_decay'].item():.4f}, "
            f"inertia={losses['inertia'].item():.4f}"
        )

        return losses

    def compute_all_losses(
        self,
        actual_state: SelfState,
        predicted_state: SelfState,
        self_predicted_state: Optional[SelfState] = None,
        state_derivative: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有损失（简化接口）

        Args:
            actual_state: 实际观测状态
            predicted_state: 预测状态
            self_predicted_state: 系统自预测状态
            state_derivative: 状态导数

        Returns:
            损失字典
        """
        return self.forward(
            actual_state=actual_state,
            predicted_state=predicted_state,
            self_predicted_state=self_predicted_state,
            state_derivative=state_derivative
        )

    def update_weights(
        self,
        lambda_weight: Optional[float] = None,
        mu_weight: Optional[float] = None,
        prediction_weight: Optional[float] = None
    ) -> None:
        """
        更新损失权重

        Args:
            lambda_weight: 抗寂灭权重
            mu_weight: 惯性权重
            prediction_weight: 预测权重
        """
        if lambda_weight is not None:
            self.lambda_weight = lambda_weight
            self.config.anti_quietus_weight = lambda_weight

        if mu_weight is not None:
            self.mu_weight = mu_weight
            self.config.inertia_weight = mu_weight

        if prediction_weight is not None:
            self.prediction_weight = prediction_weight
            self.config.prediction_weight = prediction_weight

        logger.info(
            f"Loss weights updated: "
            f"λ={self.lambda_weight}, "
            f"μ={self.mu_weight}, "
            f"prediction={self.prediction_weight}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        # 各个子损失统计
        prediction_stats = self.prediction_loss.get_statistics()
        anti_decay_stats = self.anti_decay_loss.get_statistics()
        inertia_stats = self.inertia_loss.get_statistics()

        # 总损失统计
        if self._total_loss_history:
            total_losses = [h['total'] for h in self._total_loss_history]
            prediction_losses = [h['prediction'] for h in self._total_loss_history]
            anti_decay_losses = [h['anti_decay'] for h in self._total_loss_history]
            inertia_losses = [h['inertia'] for h in self._total_loss_history]

            stats = {
                "call_count": self._call_count,
                "history_length": len(self._total_loss_history),
                "weights": {
                    "lambda": self.lambda_weight,
                    "mu": self.mu_weight,
                    "prediction": self.prediction_weight,
                },
                "total_loss": {
                    "mean": np.mean(total_losses),
                    "std": np.std(total_losses),
                    "min": np.min(total_losses),
                    "max": np.max(total_losses),
                },
                "prediction_loss": {
                    "mean": np.mean(prediction_losses),
                    "std": np.std(prediction_losses),
                },
                "anti_decay_loss": {
                    "mean": np.mean(anti_decay_losses),
                    "std": np.std(anti_decay_losses),
                },
                "inertia_loss": {
                    "mean": np.mean(inertia_losses),
                    "std": np.std(inertia_losses),
                },
                "prediction_stats": prediction_stats,
                "anti_decay_stats": anti_decay_stats,
                "inertia_stats": inertia_stats,
            }
        else:
            stats = {
                "call_count": self._call_count,
                "history_length": 0,
                "weights": {
                    "lambda": self.lambda_weight,
                    "mu": self.mu_weight,
                    "prediction": self.prediction_weight,
                },
            }

        return stats

    def reset_statistics(self) -> None:
        """重置统计信息"""
        self._total_loss_history.clear()
        self._call_count = 0

        # 重置子损失统计
        self.prediction_loss._loss_history.clear()
        self.prediction_loss._call_count = 0

        self.anti_decay_loss._self_prediction_errors.clear()
        self.anti_decay_loss._rolling_window.clear()
        self.anti_decay_loss._call_count = 0

        self.inertia_loss._derivative_history.clear()
        self.inertia_loss._call_count = 0

        logger.info("LossFunctions statistics reset")

    def __repr__(self) -> str:
        return (
            f"LossFunctions("
            f"λ={self.lambda_weight}, "
            f"μ={self.mu_weight}, "
            f"calls={self._call_count})"
        )


def create_loss_functions_from_config(
    global_config: ChronosConfig,
    device: Optional[str] = None
) -> LossFunctions:
    """
    从全局配置创建损失函数系统

    Args:
        global_config: 全局配置
        device: 计算设备

    Returns:
        LossFunctions 实例
    """
    loss_config = LossFunctionsConfig(
        anti_quietus_weight=global_config.coupling_stability.anti_quietus_weight,
        inertia_weight=global_config.coupling_stability.inertia_weight,
        fast_dim=global_config.dim.fast_variable_dim,
        slow_dim=global_config.dim.slow_variable_dim,
    )

    loss_system = LossFunctions(
        config=loss_config,
        global_config=global_config,
        coupling_config=global_config.coupling_stability,
        device=device
    )

    return loss_system