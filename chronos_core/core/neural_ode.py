"""
Neural ODE 求解器
================

实现连续时间动力学求解器，支持自适应步长和多种求解方法。

核心功能：
- 使用 torchdiffeq 库的 ODE 求解器
- 支持多种求解方法（adaptive、euler、rk4、dopri5）
- 自适应步长机制
- 数值稳定性检查（防止 NaN、Inf）
- 支持反向传播（伴随法）
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Callable, Union, List
import logging
from dataclasses import dataclass
import numpy as np

try:
    from torchdiffeq import odeint, odeint_adjoint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    logging.warning(
        "torchdiffeq not available. Neural ODE will use custom implementations. "
        "Install with: pip install torchdiffeq"
    )

from chronos_core.utils.config import NeuralODEConfig

logger = logging.getLogger(__name__)


@dataclass
class ODESolverConfig:
    """ODE 求解器配置"""
    # 求解方法
    method: str = "dopri5"  # 'euler', 'rk4', 'dopri5', 'adams', 'adaptive'

    # 容差参数（自适应方法）
    atol: float = 1e-6
    rtol: float = 1e-5

    # 步长参数（固定步长方法）
    step_size: float = 0.01

    # 最大步数
    max_steps: int = 1000

    # 安全系数（步长调整）
    safety_factor: float = 0.9

    # 最小步长
    min_step_size: float = 1e-6

    # 最大步长
    max_step_size: float = 0.1

    # 稳定性阈值
    stability_threshold: float = 1e6

    # 使用伴随法反向传播
    use_adjoint: bool = True


class DynamicsFunction(nn.Module):
    """
    动力学函数抽象基类

    所有连续时间动力学系统都需要继承此类，
    实现 forward 方法来定义状态演化函数。

    演化方程形式：
        dy/dt = f(y, t)

    其中：
        - y: 状态向量
        - t: 时间
        - f: 演化函数（由子类实现）
    """

    def __init__(self):
        super().__init__()

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        计算状态的时间导数

        Args:
            t: 时间标量
            y: 状态向量 (batch_size, state_dim) 或 (state_dim,)

        Returns:
            状态的时间导数 dy/dt
        """
        raise NotImplementedError("Subclasses must implement forward method")


class NeuralODESolver:
    """
    Neural ODE 求解器

    使用连续时间积分求解动力学系统。
    支持多种求解方法和自适应步长。

    主要功能：
    1. 连续时间积分
    2. 自适应步长控制
    3. 数值稳定性监测
    4. 支持反向传播（伴随法）

    使用示例：
        solver = NeuralODESolver(config=ODESolverConfig())
        dynamics_fn = MyDynamicsFunction()

        # 单步积分
        y_next = solver.step(dynamics_fn, y, t, dt)

        # 多步积分
        trajectory = solver.integrate(dynamics_fn, y, t_span)
    """

    def __init__(
        self,
        config: Optional[ODESolverConfig] = None,
        neural_ode_config: Optional[NeuralODEConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化 Neural ODE 求解器

        Args:
            config: ODE 求解器配置
            neural_ode_config: 来自全局配置的 Neural ODE 配置
            device: 计算设备
        """
        # 合并配置
        self.config = config or ODESolverConfig()

        if neural_ode_config:
            self.config.method = neural_ode_config.integration_method
            self.config.atol = neural_ode_config.atol
            self.config.rtol = neural_ode_config.rtol
            self.config.max_steps = neural_ode_config.max_steps
            self.config.step_size = neural_ode_config.dt

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 稳定性状态
        self.is_stable = True
        self.stability_warnings = []

        # 统计信息
        self.total_steps = 0
        self.total_integrations = 0
        self.step_size_history: List[float] = []

        # 检查 torchdiffeq 是否可用
        if not TORCHDIFFEQ_AVAILABLE:
            logger.warning(
                "torchdiffeq not available. Using custom solver implementations."
            )

        logger.info(
            f"NeuralODESolver initialized: method={self.config.method}, "
            f"atol={self.config.atol}, rtol={self.config.rtol}"
        )

    def step(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t: float,
        dt: float,
        return_intermediate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        执行单步积分

        Args:
            dynamics_fn: 动力学函数
            y: 当前状态 (state_dim,) 或 (batch_size, state_dim)
            t: 当前时间
            dt: 时间步长
            return_intermediate: 是否返回中间状态

        Returns:
            下一时刻的状态，或 (下一时刻状态, 中间状态列表)
        """
        # 确保状态在正确设备上
        y = y.to(self.device)

        # 数值稳定性检查
        if not self._check_stability(y):
            logger.warning(f"Instability detected at t={t}")
            return y  # 返回原状态，不进行积分

        # 选择求解方法
        if TORCHDIFFEQ_AVAILABLE:
            if self.config.method in ['dopri5', 'adams', 'adaptive']:
                result = self._adaptive_step(dynamics_fn, y, t, dt)
            else:
                result = self._fixed_step(dynamics_fn, y, t, dt)
        else:
            result = self._custom_step(dynamics_fn, y, t, dt)

        self.total_steps += 1

        # 再次检查稳定性
        if not self._check_stability(result):
            logger.warning(f"Instability after integration at t={t+dt}")
            self.is_stable = False

        return result

    def integrate(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t_span: torch.Tensor,
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        执行多步积分，生成完整轨迹

        Args:
            dynamics_fn: 动力学函数
            y: 初始状态
            t_span: 时间点序列 (num_steps,)
            method: 求解方法（可选，覆盖配置）

        Returns:
            状态轨迹 (num_steps, state_dim) 或 (num_steps, batch_size, state_dim)
        """
        method = method or self.config.method

        # 确保输入在正确设备上
        y = y.to(self.device)
        t_span = t_span.to(self.device)

        # 数值稳定性检查
        if not self._check_stability(y):
            logger.error("Initial state is unstable!")
            # 返回轨迹，但保持初始状态
            num_steps = t_span.shape[0]
            return y.unsqueeze(0).repeat(num_steps, *([1] * y.dim()))

        # 使用 torchdiffeq（如果可用）
        if TORCHDIFFEQ_AVAILABLE:
            try:
                if self.config.use_adjoint:
                    trajectory = odeint_adjoint(
                        dynamics_fn,
                        y,
                        t_span,
                        method=method,
                        atol=self.config.atol,
                        rtol=self.config.rtol,
                        options={'max_num_steps': self.config.max_steps}
                    )
                else:
                    trajectory = odeint(
                        dynamics_fn,
                        y,
                        t_span,
                        method=method,
                        atol=self.config.atol,
                        rtol=self.config.rtol,
                        options={'max_num_steps': self.config.max_steps}
                    )

                # 检查轨迹稳定性
                self._check_trajectory_stability(trajectory)

                self.total_integrations += 1
                return trajectory

            except Exception as e:
                logger.error(f"torchdiffeq integration failed: {e}")
                # 使用自定义求解器
                return self._custom_integrate(dynamics_fn, y, t_span)

        # 自定义求解器
        return self._custom_integrate(dynamics_fn, y, t_span)

    def _adaptive_step(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        自适应步长积分（使用 torchdiffeq）

        Args:
            dynamics_fn: 动力学函数
            y: 当前状态
            t: 当前时间
            dt: 时间步长

        Returns:
            下一时刻的状态
        """
        # 时间区间
        t_span = torch.tensor([t, t + dt], device=self.device)

        try:
            result = odeint(
                dynamics_fn,
                y,
                t_span,
                method=self.config.method,
                atol=self.config.atol,
                rtol=self.config.rtol,
                options={'max_num_steps': self.config.max_steps}
            )

            # 返回最终状态
            return result[-1]

        except Exception as e:
            logger.warning(f"Adaptive step failed: {e}")
            # 回退到固定步长
            return self._fixed_step(dynamics_fn, y, t, dt)

    def _fixed_step(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        固定步长积分（使用 torchdiffeq）

        Args:
            dynamics_fn: 动力学函数
            y: 当前状态
            t: 当前时间
            dt: 时间步长

        Returns:
            下一时刻的状态
        """
        method = self.config.method
        if method == 'adaptive':
            method = 'euler'  # 自适应不可用时使用欧拉法

        # 时间区间
        t_span = torch.tensor([t, t + dt], device=self.device)

        try:
            result = odeint(
                dynamics_fn,
                y,
                t_span,
                method=method,
                options={
                    'step_size': self.config.step_size,
                    'max_num_steps': self.config.max_steps
                }
            )

            return result[-1]

        except Exception as e:
            logger.warning(f"Fixed step failed: {e}")
            # 使用自定义求解器
            return self._custom_step(dynamics_fn, y, t, dt)

    def _custom_step(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t: float,
        dt: float
    ) -> torch.Tensor:
        """
        自定义求解器（当 torchdiffeq 不可用时）

        使用改进的 RK4 方法。

        Args:
            dynamics_fn: 动力学函数
            y: 当前状态
            t: 当前时间
            dt: 时间步长

        Returns:
            下一时刻的状态
        """
        # RK4 方法
        t_tensor = torch.tensor(t, device=self.device)

        # k1 = f(y, t)
        k1 = dynamics_fn(t_tensor, y)

        # k2 = f(y + dt/2 * k1, t + dt/2)
        k2 = dynamics_fn(t_tensor + dt/2, y + dt/2 * k1)

        # k3 = f(y + dt/2 * k2, t + dt/2)
        k3 = dynamics_fn(t_tensor + dt/2, y + dt/2 * k2)

        # k4 = f(y + dt * k3, t + dt)
        k4 = dynamics_fn(t_tensor + dt, y + dt * k3)

        # RK4 更新
        y_next = y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6

        return y_next

    def _custom_integrate(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """
        自定义多步积分

        Args:
            dynamics_fn: 动力学函数
            y: 初始状态
            t_span: 时间点序列

        Returns:
            状态轨迹
        """
        num_steps = t_span.shape[0]
        state_dim = y.shape[-1] if y.dim() > 1 else y.shape[0]

        # 初始化轨迹
        trajectory = torch.zeros((num_steps, *y.shape), device=self.device)
        trajectory[0] = y

        # 逐步积分
        current_y = y
        for i in range(1, num_steps):
            dt = (t_span[i] - t_span[i-1]).item()
            current_y = self._custom_step(dynamics_fn, current_y, t_span[i-1].item(), dt)
            trajectory[i] = current_y

        return trajectory

    def _check_stability(self, y: torch.Tensor) -> bool:
        """
        检查数值稳定性

        Args:
            y: 状态向量

        Returns:
            是否稳定
        """
        # 检查 NaN
        if torch.isnan(y).any():
            warning = f"NaN detected in state"
            self.stability_warnings.append(warning)
            logger.warning(warning)
            return False

        # 检查 Inf
        if torch.isinf(y).any():
            warning = f"Inf detected in state"
            self.stability_warnings.append(warning)
            logger.warning(warning)
            return False

        # 检查范数是否过大
        norm = torch.norm(y).item()
        if norm > self.config.stability_threshold:
            warning = f"State norm too large: {norm:.4e} > {self.config.stability_threshold}"
            self.stability_warnings.append(warning)
            logger.warning(warning)
            return False

        return True

    def _check_trajectory_stability(self, trajectory: torch.Tensor) -> bool:
        """
        检查轨迹稳定性

        Args:
            trajectory: 状态轨迹

        Returns:
            是否稳定
        """
        # 检查每个时间点
        for i in range(trajectory.shape[0]):
            if not self._check_stability(trajectory[i]):
                logger.warning(f"Instability at trajectory step {i}")
                return False

        # 检查整体范数趋势
        norms = torch.norm(trajectory, dim=-1)
        if torch.isnan(norms).any() or torch.isinf(norms).any():
            logger.warning("NaN or Inf in trajectory norms")
            return False

        # 检查是否发散（范数持续增大）
        if norms.shape[0] > 10:
            initial_norm = norms[:5].mean().item()
            final_norm = norms[-5:].mean().item()
            if final_norm > initial_norm * 10:
                logger.warning(
                    f"Trajectory appears to diverge: "
                    f"initial_norm={initial_norm:.4e}, final_norm={final_norm:.4e}"
                )
                return False

        return True

    def adaptive_integrate(
        self,
        dynamics_fn: DynamicsFunction,
        y: torch.Tensor,
        t_start: float,
        t_end: float,
        initial_dt: Optional[float] = None
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        自适应步长积分（自动调整步长）

        Args:
            dynamics_fn: 动力学函数
            y: 初始状态
            t_start: 开始时间
            t_end: 结束时间
            initial_dt: 初始步长

        Returns:
            (最终状态, 步长历史)
        """
        initial_dt = initial_dt or self.config.step_size
        t = t_start
        current_y = y.to(self.device)
        dt = initial_dt

        step_history = []

        while t < t_end:
            # 调整步长以不超过结束时间
            if t + dt > t_end:
                dt = t_end - t

            # 尝试积分
            try:
                y_candidate = self._custom_step(dynamics_fn, current_y, t, dt)

                # 误差估计（使用半步比较）
                y_half = self._custom_step(dynamics_fn, current_y, t, dt/2)
                y_double_half = self._custom_step(dynamics_fn, y_half, t + dt/2, dt/2)

                # 误差
                error = torch.norm(y_candidate - y_double_half).item()

                # 自适应步长调整
                if error < self.config.atol:
                    # 误差小，增大步长
                    dt = min(dt * 1.5, self.config.max_step_size)
                    current_y = y_double_half  # 使用更精确的结果
                elif error > self.config.rtol:
                    # 误差大，减小步长
                    dt = max(dt / 2, self.config.min_step_size)
                    # 重试当前步
                    continue
                else:
                    # 误差适中，保持步长
                    current_y = y_candidate

                t += dt
                step_history.append(dt)
                self.step_size_history.append(dt)

                # 检查稳定性
                if not self._check_stability(current_y):
                    logger.warning(f"Instability at t={t}")
                    break

            except Exception as e:
                logger.error(f"Adaptive integration error at t={t}: {e}")
                break

        return current_y, step_history

    def get_statistics(self) -> Dict:
        """
        获取求解器统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "total_steps": self.total_steps,
            "total_integrations": self.total_integrations,
            "is_stable": self.is_stable,
            "stability_warnings_count": len(self.stability_warnings),
            "method": self.config.method,
            "atol": self.config.atol,
            "rtol": self.config.rtol,
            "torchdiffeq_available": TORCHDIFFEQ_AVAILABLE,
            "average_step_size": np.mean(self.step_size_history) if self.step_size_history else 0.0
        }

        return stats

    def reset(self) -> None:
        """重置求解器状态"""
        self.is_stable = True
        self.stability_warnings.clear()
        self.total_steps = 0
        self.total_integrations = 0
        self.step_size_history.clear()

        logger.info("NeuralODESolver reset")

    def set_method(self, method: str) -> None:
        """
        设置求解方法

        Args:
            method: 求解方法名称
        """
        valid_methods = ['euler', 'rk4', 'dopri5', 'adams', 'adaptive']
        if method in valid_methods:
            self.config.method = method
            logger.info(f"Solver method set to: {method}")
        else:
            logger.warning(f"Unknown method: {method}. Valid methods: {valid_methods}")

    def __repr__(self) -> str:
        return (
            f"NeuralODESolver(method={self.config.method}, "
            f"steps={self.total_steps}, stable={self.is_stable})"
        )


def create_solver_from_config(
    neural_ode_config: NeuralODEConfig,
    device: Optional[str] = None
) -> NeuralODESolver:
    """
    从全局配置创建 ODE 求解器

    Args:
        neural_ode_config: Neural ODE 配置
        device: 计算设备

    Returns:
        NeuralODESolver 实例
    """
    solver_config = ODESolverConfig(
        method=neural_ode_config.integration_method,
        atol=neural_ode_config.atol,
        rtol=neural_ode_config.rtol,
        step_size=neural_ode_config.dt,
        max_steps=neural_ode_config.max_steps
    )

    solver = NeuralODESolver(config=solver_config, device=device)

    return solver