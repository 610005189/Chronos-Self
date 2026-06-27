"""
完整积分引擎
============

整合快变量动力学、慢变量动力学、耦合机制、稳定性保障，
实现完整的连续时间动力学系统。

核心功能：
- 接收外部输入和内部信号
- 更新快变量（Neural ODE）
- 更新慢变量（低频更新）
- 应用稳定性约束
- 输出新的自我状态
- 支持批量积分和长时序运行
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List, Any
import logging
from dataclasses import dataclass, field
import time

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    NeuralODEConfig,
    CouplingStabilityConfig,
    MemoryTemporalConfig,
    MetaCognitiveConfig,
    ChaosInjectionConfig
)

from .state import SelfState
from .external_input import ExternalInput
from .dmn_system import DefaultModeNetwork
from .fast_dynamics import FastDynamicsSystem, FastDynamicsConfig, FastDynamicsFunction
from .slow_dynamics import SlowDynamicsSystem, SlowDynamicsConfig, SlowDynamicsFunction
from .coupling import CouplingAndStabilitySystem, CouplingConfig
from .neural_ode import NeuralODESolver, ODESolverConfig
from .state_manager import StateManager

from chronos_core.representation.fusion import FusionModule, FusionOutput

logger = logging.getLogger(__name__)


@dataclass
class IntegrationEngineConfig:
    """积分引擎配置"""

    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512

    # 时间参数
    default_dt: float = 0.01  # 默认时间步长
    slow_update_frequency: int = 100  # 慢变量更新频率

    # 求解器参数
    solver_method: str = "dopri5"
    solver_atol: float = 1e-6
    solver_rtol: float = 1e-5

    # IMEX 求解器参数
    solver_type: str = "euler"  # euler | imex | rk4 | verlet
    imex_update_interval: int = 100  # J0 更新间隔（步数）
    imex_j0_eps: float = 1e-6  # J0 估算的扰动大小
    imex_j0_clamp_min: float = 0.0  # J0 对角元素最小值（确保非负刚性）

    # 稳定性参数
    stability_check_interval: int = 100
    auto_stability_actions: bool = True

    # 批量参数
    max_batch_size: int = 32

    # 监控参数
    log_interval: int = 1000  # 日志记录间隔
    history_limit: int = 10000  # 历史记录限制


class IntegrationEngine(nn.Module):
    """
    完整积分引擎

    整合所有动力学组件，实现完整的自我状态演化系统。

    主要功能：
    1. 整合快变量、慢变量、耦合机制
    2. 实现完整的时间积分流程
    3. 支持外部输入和内部信号
    4. 稳定性监测和保障
    5. 支持批量积分和长时序运行
    6. 提供状态监测接口

    使用示例：
        engine = IntegrationEngine(config=ChronosConfig())
        engine.initialize()

        # 单步积分
        new_state = engine.step(current_state, inputs, dt)

        # 多步积分
        trajectory = engine.integrate(initial_state, inputs_sequence, t_span)

        # 长时序运行
        engine.run_continuous(duration_hours=72.0)
    """

    def __init__(
        self,
        config: Optional[ChronosConfig] = None,
        engine_config: Optional[IntegrationEngineConfig] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化积分引擎

        Args:
            config: 全局配置
            engine_config: 引擎配置
            device: 计算设备
            seed: 随机种子
        """
        super().__init__()

        # 配置
        self.global_config = config or ChronosConfig()
        self.engine_config = engine_config or IntegrationEngineConfig()

        # 合并维度配置
        self.engine_config.fast_dim = self.global_config.dim.fast_variable_dim
        self.engine_config.slow_dim = self.global_config.dim.slow_variable_dim
        self.engine_config.slow_update_frequency = self.global_config.memory_temporal.slow_update_frequency

        # 合并 IMEX 相关配置（从 global_config.numerics 中读取）
        self.engine_config.solver_type = self.global_config.numerics.solver_type
        self.engine_config.imex_update_interval = self.global_config.numerics.imex_update_interval
        # imex_j0_eps 和 imex_j0_clamp_min 使用默认值

        # 设备和种子
        self.device = device or self.global_config.device
        self.seed = seed or self.global_config.random_seed

        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # 核心组件
        self.fast_dynamics: Optional[FastDynamicsSystem] = None
        self.slow_dynamics: Optional[SlowDynamicsSystem] = None
        self.coupling_system: Optional[CouplingAndStabilitySystem] = None
        self.ode_solver: Optional[NeuralODESolver] = None
        self.dmn: Optional[DefaultModeNetwork] = None
        self.fusion_module: Optional[FusionModule] = None

        # 状态管理器
        self.state_manager = StateManager(
            default_device=torch.device(self.device) if self.device else None,
            max_history_length=self.engine_config.history_limit
        )

        # 状态缓存
        self._current_E_fast_prev: Optional[torch.Tensor] = None
        self._current_E_slow_prev: Optional[torch.Tensor] = None

        # IMEX 求解器缓存
        self._J0_cache: Optional[torch.Tensor] = None  # 线性主部（对角矩阵）
        self._J0_update_counter: int = 0  # J0 更新计数器
        self._last_input_dict: Optional[Dict[str, torch.Tensor]] = None  # 最近的输入字典（用于估算 J0）

        # 运行统计
        self.step_count = 0
        self.total_time = 0.0
        self.fast_updates = 0
        self.slow_updates = 0
        self.stability_warnings = 0

        # 历史记录
        self.state_history: List[SelfState] = []

        # 初始化标志
        self._initialized = False

        logger.info(
            f"IntegrationEngine created: "
            f"fast_dim={self.engine_config.fast_dim}, "
            f"slow_dim={self.engine_config.slow_dim}, "
            f"device={self.device}"
        )

    def initialize(self) -> None:
        """
        初始化所有组件

        包括：
        - 快变量动力学系统
        - 慢变量动力学系统
        - 耦合与稳定性系统
        - ODE 求解器
        - DMN 系统（混沌注入）
        """
        # 创建快变量动力学系统
        self.fast_dynamics = FastDynamicsSystem(
            dim_config=self.global_config.dim,
            meta_config=self.global_config.meta_cognitive,
            device=self.device
        )
        self.fast_dynamics.initialize()
        self.add_module('fast_dynamics', self.fast_dynamics)

        # 创建慢变量动力学系统
        self.slow_dynamics = SlowDynamicsSystem(
            dim_config=self.global_config.dim,
            coupling_config=self.global_config.coupling_stability,
            temporal_config=self.global_config.memory_temporal,
            device=self.device
        )
        self.slow_dynamics.initialize()
        self.add_module('slow_dynamics', self.slow_dynamics)

        # 创建耦合与稳定性系统
        self.coupling_system = CouplingAndStabilitySystem(
            coupling_config=self.global_config.coupling_stability,
            dim_config=self.global_config.dim,
            device=self.device
        )
        self.coupling_system.initialize()

        # 创建 ODE 求解器
        self.ode_solver = NeuralODESolver(
            neural_ode_config=self.global_config.neural_ode,
            device=self.device
        )

        # 创建 DMN 系统
        self.dmn = DefaultModeNetwork(
            chaos_config=self.global_config.chaos_injection,
            dim_config=self.global_config.dim,
            device=self.device,
            seed=self.seed
        )
        self.dmn.initialize()

        # 创建 FusionModule（双向交叉注意力融合）
        self.fusion_module = FusionModule(
            sem_dim=self.global_config.dim.semantic_dim,
            log_dim=self.global_config.dim.physical_dim,
            fusion_dim=self.global_config.dim.fusion_dim,
            num_heads=8,
            dropout=0.1
        )
        self.fusion_module.to(self.device)
        self.add_module('fusion_module', self.fusion_module)

        self._initialized = True

        logger.info(
            "IntegrationEngine initialized: "
            f"fast_dynamics={self.fast_dynamics}, "
            f"slow_dynamics={self.slow_dynamics}, "
            f"coupling={self.coupling_system}"
        )

    def step(
        self,
        current_state: SelfState,
        inputs: Optional[ExternalInput] = None,
        meta_cognitive_signal: Optional[torch.Tensor] = None,
        dt: Optional[float] = None,
        return_intermediate: bool = False
    ) -> SelfState:
        """
        执行单步积分

        Args:
            current_state: 当前自我状态
            inputs: 外部输入（可选）
            meta_cognitive_signal: 元认知调控信号（可选）
            dt: 时间步长（可选）
            return_intermediate: 是否返回中间状态

        Returns:
            新的自我状态
        """
        if not self._initialized:
            raise ValueError("Engine not initialized. Call initialize() first.")

        # 使用默认时间步长
        dt = dt or self.engine_config.default_dt

        # 确保状态在正确设备上
        E_fast = current_state.E_fast.to(self.device)
        E_slow = current_state.E_slow.to(self.device)

        # 保存前一时刻状态（用于稳定性监测）
        self._current_E_fast_prev = E_fast.clone()
        self._current_E_slow_prev = E_slow.clone()

        # 1. 获取混沌注入信号（来自 DMN）
        B_chaos = self.dmn.step(dt, E_fast)

        # 2. 构建输入信号字典
        input_dict = self._build_input_dict(
            inputs=inputs,
            meta_cognitive_signal=meta_cognitive_signal,
            B_chaos=B_chaos
        )

        # 3. 更新耦合系数
        coupling_coeff = self.coupling_system.update_coupling(E_fast)

        # 4. 更新快变量（Neural ODE）
        E_fast_new = self._update_fast_variable(
            E_fast, E_slow, input_dict, dt, current_state.timestamp
        )

        # 5. 检查是否需要更新慢变量
        if self.slow_dynamics.should_update_slow():
            # 更新慢变量（低频更新）
            E_slow_new = self._update_slow_variable(
                E_slow, E_fast_new, coupling_coeff, dt, current_state.timestamp
            )
            self.slow_updates += 1
        else:
            # 慢变量保持不变
            E_slow_new = E_slow

        # 6. 稳定性监测
        stability_report = self.coupling_system.monitor_stability(
            E_fast_new, E_slow_new,
            E_fast_prev=self._current_E_fast_prev,
            E_slow_prev=self._current_E_slow_prev,
            step_count=self.step_count
        )

        # 7. 应用稳定性保障措施（如果需要）
        if self.engine_config.auto_stability_actions and stability_report.get("actions_taken"):
            self.coupling_system.apply_stability_actions(
                stability_report["actions_taken"],
                state_manager=self.state_manager
            )

        # 8. 创建新状态
        new_state = SelfState(
            E_fast=E_fast_new.detach().cpu(),
            E_slow=E_slow_new.detach().cpu(),
            timestamp=current_state.timestamp + dt,
            metadata={
                "step_count": self.step_count,
                "coupling_coeff": coupling_coeff,
                "stability_report": stability_report
            }
        )

        # 9. 更新统计
        self.step_count += 1
        self.fast_updates += 1
        self.total_time += dt

        if not stability_report.get("is_stable", True):
            self.stability_warnings += 1

        # 10. 记录历史（可选）
        if len(self.state_history) < self.engine_config.history_limit:
            self.state_history.append(new_state.copy())

        # 日志记录
        if self.step_count % self.engine_config.log_interval == 0:
            logger.info(
                f"Step {self.step_count}: time={self.total_time:.2f}, "
                f"E_fast_norm={new_state.get_fast_norm():.4f}, "
                f"E_slow_norm={new_state.get_slow_norm():.4f}, "
                f"coupling={coupling_coeff:.4f}"
            )

        return new_state

    def _build_input_dict(
        self,
        inputs: Optional[ExternalInput],
        meta_cognitive_signal: Optional[torch.Tensor],
        B_chaos: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        构建输入信号字典

        Args:
            inputs: 外部输入
            meta_cognitive_signal: 元认知调控信号
            B_chaos: 混沌注入信号

        Returns:
            输入信号字典
        """
        input_dict = {}

        # 外部输入
        if inputs is not None:
            X_sem = inputs.X_sem.to(self.device)
            X_log = inputs.X_log.to(self.device)

            # 确保维度匹配配置
            target_sem_dim = self.global_config.dim.semantic_dim
            target_phys_dim = self.global_config.dim.physical_dim

            if X_sem.shape[0] != target_sem_dim:
                if X_sem.shape[0] > target_sem_dim:
                    X_sem = X_sem[:target_sem_dim]
                else:
                    padding = torch.zeros(target_sem_dim - X_sem.shape[0], device=self.device)
                    X_sem = torch.cat([X_sem, padding], dim=0)
                logger.debug(f"Adjusted X_sem dimension: {inputs.X_sem.shape[0]} -> {target_sem_dim}")

            if X_log.shape[0] != target_phys_dim:
                if X_log.shape[0] > target_phys_dim:
                    X_log = X_log[:target_phys_dim]
                else:
                    padding = torch.zeros(target_phys_dim - X_log.shape[0], device=self.device)
                    X_log = torch.cat([X_log, padding], dim=0)
                logger.debug(f"Adjusted X_log dimension: {inputs.X_log.shape[0]} -> {target_phys_dim}")

            input_dict['X_sem'] = X_sem
            input_dict['X_log'] = X_log

            # 使用 FusionModule 进行双向交叉注意力融合
            # 添加 batch 和 seq_len 维度：(dim,) -> (1, 1, dim)
            X_sem_seq = X_sem.unsqueeze(0).unsqueeze(0)
            X_log_seq = X_log.unsqueeze(0).unsqueeze(0)

            # 执行融合
            fusion_output: FusionOutput = self.fusion_module(
                X_sem_seq, X_log_seq,
                return_enriched=True,
                need_attention_weights=False
            )

            # 提取融合结果并移除 batch 和 seq_len 维度
            input_dict['X_fused'] = fusion_output.X_fused.squeeze(0).squeeze(0)
            input_dict['X_sem_enriched'] = fusion_output.X_sem_enriched.squeeze(0).squeeze(0)
            input_dict['X_log_enriched'] = fusion_output.X_log_enriched.squeeze(0).squeeze(0)

        # 元认知调控信号
        if meta_cognitive_signal is not None:
            input_dict['C_meta'] = meta_cognitive_signal.to(self.device)

        # 混沌注入信号
        input_dict['B_chaos'] = B_chaos

        return input_dict

    def _update_fast_variable(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        input_dict: Dict[str, torch.Tensor],
        dt: float,
        t: float
    ) -> torch.Tensor:
        """
        更新快变量状态

        根据 solver_type 配置选择不同的求解器：
        - euler: 原始 euler 方法（默认）
        - imex: IMEX 算子分裂求解器（新增）
        - rk4: RK4 方法（如有）
        - verlet: Verlet 方法（可选）

        Args:
            E_fast: 当前快变量
            E_slow: 慢变量
            input_dict: 输入字典
            dt: 时间步长
            t: 当前时间

        Returns:
            新的快变量状态
        """
        # 缓存输入字典（用于 IMEX 求解器估算 J0）
        self._last_input_dict = input_dict

        # 根据配置选择求解器
        solver_type = self.engine_config.solver_type

        if solver_type == "imex":
            # 使用 IMEX 求解器
            E_fast_new = self.step_imex(
                E_fast, E_slow, input_dict, dt, t
            )
        elif solver_type == "euler":
            # 使用原始 euler 方法
            E_fast_new = self.fast_dynamics.step(
                E_fast, E_slow, input_dict, dt, t
            )
        elif solver_type == "rk4":
            # 使用 RK4 求解器
            E_fast_new = self.step_rk4(
                E_fast, E_slow, input_dict, dt, t
            )
        elif solver_type == "verlet":
            # Verlet 方法（辛积分器）
            E_fast_new = self.step_verlet(
                E_fast, E_slow, input_dict, dt, t
            )
        else:
            # 未知求解器类型，回退到 euler
            logger.warning(f"Unknown solver_type '{solver_type}', falling back to euler")
            E_fast_new = self.fast_dynamics.step(
                E_fast, E_slow, input_dict, dt, t
            )

        return E_fast_new

    def _update_slow_variable(
        self,
        E_slow: torch.Tensor,
        E_fast: torch.Tensor,
        coupling_coeff: float,
        dt: float,
        t: float
    ) -> torch.Tensor:
        """
        更新慢变量状态

        Args:
            E_slow: 当前慢变量
            E_fast: 快变量
            coupling_coeff: 耦合系数
            dt: 时间步长（快变量时间步长）
            t: 当前时间

        Returns:
            新的慢变量状态
        """
        # 慢变量时间步长（相对于快变量）
        dt_slow = dt * self.engine_config.slow_update_frequency

        # 使用慢变量动力学系统更新
        E_slow_new = self.slow_dynamics.step(
            E_slow, E_fast, dt_slow, t
        )

        return E_slow_new

    def _estimate_j0(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        input_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        从当前状态估算线性主部 J0（对角近似）

        使用有限差分法估算雅可比矩阵的对角部分，代表刚性模态。
        为了提高效率，使用采样策略和批量计算。

        Args:
            E_fast: 当前快变量状态
            E_slow: 慢变量状态
            input_dict: 输入字典

        Returns:
            J0: 对角矩阵（刚性部分）
        """
        eps = self.engine_config.imex_j0_eps
        fast_dim = E_fast.shape[-1]

        # 采样策略：为了提高效率，只估算部分维度
        # 如果维度小于 100，则全部计算
        # 如果维度大于 100，则采样 100 个维度
        sample_size = min(100, fast_dim)
        sample_indices = torch.linspace(0, fast_dim - 1, sample_size).long() \
            if fast_dim > sample_size else torch.arange(fast_dim)

        # 获取动力学函数
        dynamics_fn = self.fast_dynamics.dynamics_fn

        # 批量计算：一次性计算所有采样维度的扰动
        # 构建 batch: [2 * sample_size, fast_dim]
        # 每个采样维度有两个扰动（+eps 和 -eps）
        batch_size = 2 * sample_size
        batch_E = E_fast.unsqueeze(0).repeat(batch_size, 1)  # [batch_size, fast_dim]

        # 为每个采样维度添加扰动
        for idx, i in enumerate(sample_indices):
            batch_E[2 * idx, i] += eps  # +eps
            batch_E[2 * idx + 1, i] -= eps  # -eps

        # 批量计算动力学
        with torch.no_grad():  # 不需要梯度
            # 提取输入参数
            X_sem = input_dict.get('X_sem')
            X_log = input_dict.get('X_log')
            X_fused = input_dict.get('X_fused')
            C_meta = input_dict.get('C_meta')
            B_chaos = input_dict.get('B_chaos')

            # 需要将 E_slow 也扩展为 batch
            batch_E_slow = E_slow.unsqueeze(0).repeat(batch_size, 1)

            # 批量计算动力学（需要修改 dynamics_fn 以支持批量输入）
            # 但当前 dynamics_fn 可能不支持批量输入，所以我们需要逐个计算
            # 为了提高效率，我们可以分批计算
            batch_f = []
            chunk_size = 10  # 每次计算 10 个扰动
            for chunk_start in range(0, batch_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, batch_size)
                chunk_E = batch_E[chunk_start:chunk_end]

                # 对 chunk 中的每个元素计算动力学
                chunk_f = []
                for j in range(chunk_E.shape[0]):
                    f_j = dynamics_fn.forward(
                        torch.tensor(0.0, device=self.device),
                        chunk_E[j],
                        E_slow=E_slow,
                        X_sem=X_sem,
                        X_log=X_log,
                        X_fused=X_fused,
                        C_meta=C_meta,
                        B_chaos=B_chaos
                    )
                    chunk_f.append(f_j)

                batch_f.extend(chunk_f)

            # 转换为 tensor
            batch_f = torch.stack(batch_f)  # [batch_size, fast_dim]

        # 计算采样维度的雅可比对角元素
        sampled_J0_diag = torch.zeros(sample_size, device=self.device, dtype=E_fast.dtype)

        for idx, i in enumerate(sample_indices):
            f_plus = batch_f[2 * idx]
            f_minus = batch_f[2 * idx + 1]

            # 计算对角元素（中心差分）
            sampled_J0_diag[idx] = ((f_plus - f_minus) / (2 * eps))[i].mean()

        # 应用非负约束（确保刚性模态）
        sampled_J0_diag = sampled_J0_diag.clamp(min=self.engine_config.imex_j0_clamp_min)

        # 如果使用了采样策略，则通过插值或平均得到所有维度的对角元素
        if fast_dim > sample_size:
            # 方法 1：使用平均值（简单但有效）
            # 假设所有维度的刚性相似
            mean_diag = sampled_J0_diag.mean()
            J0_diag = torch.full((fast_dim,), mean_diag, device=self.device, dtype=E_fast.dtype)

            # 方法 2：使用线性插值（更精确）
            # 从 sampled_J0_diag 插值到所有维度
            # sampled_indices: [sample_size]
            # sampled_values: [sample_size]
            # 目标 indices: [fast_dim]
            # 使用 PyTorch 的插值功能
            # 但 PyTorch 没有直接的一维插值函数，我们可以使用自定义实现
            # 为了简化，使用方法 1（平均值）
        else:
            # 如果没有采样，直接使用计算结果
            J0_diag = sampled_J0_diag

        # 构建对角矩阵
        J0 = torch.diag(J0_diag)

        logger.debug(
            f"J0 estimated: sample_size={sample_size}, "
            f"diag_range=[{J0_diag.min().item():.4f}, {J0_diag.max().item():.4f}], "
            f"mean={J0_diag.mean().item():.4f}"
        )

        return J0

    def step_imex(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        input_dict: Dict[str, torch.Tensor],
        dt: float,
        t: float
    ) -> torch.Tensor:
        """
        IMEX 算子分裂求解器步骤

        将动力学分裂为线性部分（隐式求解）和非线性部分（显式求解）：
        - dE_fast/dt = J0 @ E_fast + N(E_fast, ...)
        - 其中 J0 是线性主部（对角矩阵，代表刚性）
        - N 是非线性残差

        Args:
            E_fast: 当前快变量状态
            E_slow: 慢变量状态
            input_dict: 输入字典
            dt: 时间步长
            t: 当前时间

        Returns:
            新的快变量状态
        """
        # 1. 检查是否需要更新 J0
        if (self._J0_cache is None or
            self._J0_update_counter >= self.engine_config.imex_update_interval):
            # 更新 J0
            self._J0_cache = self._estimate_j0(E_fast, E_slow, input_dict)
            self._J0_update_counter = 0
            logger.debug(f"J0 updated at step {self.step_count}")

        # 2. 隐式求解线性部分（解析解）
        # (I - dt * J0) @ E_linear = E_fast
        # E_linear = (I - dt * J0)^{-1} @ E_fast
        J0 = self._J0_cache
        I = torch.eye(J0.shape[0], device=self.device, dtype=J0.dtype)

        # 计算 (I - dt * J0) 的逆
        # 对于对角矩阵，逆矩阵也是对角矩阵，可以简化计算
        # 但为了通用性，这里使用矩阵求逆
        A = I - dt * J0

        # 检查矩阵是否可逆（避免数值不稳定）
        try:
            # 对于对角矩阵，使用更高效的方法
            if torch.allclose(J0 - torch.diag(torch.diag(J0)), torch.zeros_like(J0)):
                # 对角矩阵情况：直接计算逆对角元素
                A_diag_inv = 1.0 / torch.diag(A).clamp(min=1e-10)  # 防止除零
                E_linear = A_diag_inv * E_fast  # 对角矩阵乘法
            else:
                # 非对角矩阵情况（如果未来扩展）
                A_inv = torch.inverse(A)
                E_linear = A_inv @ E_fast
        except Exception as e:
            # 如果求逆失败，回退到 euler 方法
            logger.warning(f"IMEX matrix inversion failed: {e}, falling back to euler")
            return self.fast_dynamics.step(E_fast, E_slow, input_dict, dt, t)

        # 3. 计算非线性残差
        # N_theta = F_theta(E_fast, ...) - J0 @ E_fast
        dynamics_fn = self.fast_dynamics.dynamics_fn

        # 计算完整动力学 F_theta
        X_sem = input_dict.get('X_sem')
        X_log = input_dict.get('X_log')
        X_fused = input_dict.get('X_fused')
        C_meta = input_dict.get('C_meta')
        B_chaos = input_dict.get('B_chaos')

        F_theta = dynamics_fn.forward(
            torch.tensor(t, device=self.device),
            E_fast,
            E_slow=E_slow,
            X_sem=X_sem,
            X_log=X_log,
            X_fused=X_fused,
            C_meta=C_meta,
            B_chaos=B_chaos
        )

        # 计算非线性部分
        N_theta = F_theta - J0 @ E_fast

        # 4. 显式求解非线性部分（euler 步）
        # E_nonlinear = E_fast + dt * N_theta
        E_nonlinear = E_fast + dt * N_theta

        # 5. 组合更新
        # E_new = E_linear + (E_nonlinear - E_fast)
        # 这相当于：E_new = E_linear + dt * N_theta
        E_new = E_linear + (E_nonlinear - E_fast)

        # 6. 状态范数裁剪（防止发散）
        norm = torch.norm(E_new).item()
        if norm > self.fast_dynamics.config.state_norm_threshold:
            scale = self.fast_dynamics.config.state_norm_threshold / norm
            E_new = E_new * scale
            logger.debug(f"IMEX state norm clipped: {norm:.4e} -> {self.fast_dynamics.config.state_norm_threshold}")

        # 7. 更新计数器
        self._J0_update_counter += 1

        # 8. 稳定性检查
        self.fast_dynamics._check_stability(E_new)

        return E_new

    def step_verlet(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        input_dict: Dict[str, torch.Tensor],
        dt: float,
        t: float
    ) -> torch.Tensor:
        """
        Verlet-like 近似辛积分器步骤

        实现辛积分器（Verlet 模式），提供长期稳定性保障。
        辛积分器能够保持系统的辛结构，降低长期能量漂移误差。

        算法步骤：
        1. 位置半更新：E_half = E_fast + 0.5 * dt * F_theta(E_fast, E_slow, ...)
        2. 速度全更新：E_new = E_fast + dt * F_theta(E_half, E_slow, ...)
        3. 位置半修正：E_new = E_new + 0.5 * dt * F_theta(E_new, E_slow, ...)

        Args:
            E_fast: 当前快变量状态
            E_slow: 慢变量状态
            input_dict: 输入字典（包含 X_sem, X_log, X_fused, C_meta, B_chaos）
            dt: 时间步长
            t: 当前时间

        Returns:
            新的快变量状态
        """
        dynamics_fn = self.fast_dynamics.dynamics_fn

        # 提取输入参数
        X_sem = input_dict.get('X_sem')
        X_log = input_dict.get('X_log')
        X_fused = input_dict.get('X_fused')
        C_meta = input_dict.get('C_meta')
        B_chaos = input_dict.get('B_chaos')

        # 辅助函数：计算动力学 F_theta
        def compute_F_theta(state):
            return dynamics_fn.forward(
                torch.tensor(t, device=self.device),
                state,
                E_slow=E_slow,
                X_sem=X_sem,
                X_log=X_log,
                X_fused=X_fused,
                C_meta=C_meta,
                B_chaos=B_chaos
            )

        # 1. 位置半更新（使用当前状态）
        # E_half = E_fast + 0.5 * dt * F_theta(E_fast, E_slow, ...)
        F_initial = compute_F_theta(E_fast)
        E_half = E_fast + 0.5 * dt * F_initial

        # 2. 速度全更新（使用半位置）
        # E_new = E_fast + dt * F_theta(E_half, E_slow, ...)
        F_half = compute_F_theta(E_half)
        E_new = E_fast + dt * F_half

        # 3. 位置半修正（使用新状态）
        # E_new = E_new + 0.5 * dt * F_theta(E_new, E_slow, ...)
        F_new = compute_F_theta(E_new)
        E_new = E_new + 0.5 * dt * F_new

        # 4. 状态范数裁剪（防止发散）
        norm = torch.norm(E_new).item()
        if norm > self.fast_dynamics.config.state_norm_threshold:
            scale = self.fast_dynamics.config.state_norm_threshold / norm
            E_new = E_new * scale
            logger.debug(f"Verlet state norm clipped: {norm:.4e} -> {self.fast_dynamics.config.state_norm_threshold}")

        # 5. 稳定性检查
        self.fast_dynamics._check_stability(E_new)

        return E_new

    def step_rk4(
        self,
        E_fast: torch.Tensor,
        E_slow: torch.Tensor,
        input_dict: Dict[str, torch.Tensor],
        dt: float,
        t: float
    ) -> torch.Tensor:
        """
        经典四阶 Runge-Kutta (RK4) 求解器步骤

        RK4 是一种显式数值积分方法，具有四阶精度。算法步骤：
        k1 = f(t, y)
        k2 = f(t + dt/2, y + dt/2 * k1)
        k3 = f(t + dt/2, y + dt/2 * k2)
        k4 = f(t + dt, y + dt * k3)
        y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        Args:
            E_fast: 当前快变量状态
            E_slow: 慢变量状态
            input_dict: 输入字典（包含 X_sem, X_log, X_fused, C_meta, B_chaos）
            dt: 时间步长
            t: 当前时间

        Returns:
            新的快变量状态
        """
        dynamics_fn = self.fast_dynamics.dynamics_fn

        X_sem = input_dict.get('X_sem')
        X_log = input_dict.get('X_log')
        X_fused = input_dict.get('X_fused')
        C_meta = input_dict.get('C_meta')
        B_chaos = input_dict.get('B_chaos')

        def compute_F_theta(state, time):
            return dynamics_fn.forward(
                torch.tensor(time, device=self.device),
                state,
                E_slow=E_slow,
                X_sem=X_sem,
                X_log=X_log,
                X_fused=X_fused,
                C_meta=C_meta,
                B_chaos=B_chaos
            )

        # RK4 算法步骤
        k1 = compute_F_theta(E_fast, t)

        k2_state = E_fast + 0.5 * dt * k1
        k2 = compute_F_theta(k2_state, t + 0.5 * dt)

        k3_state = E_fast + 0.5 * dt * k2
        k3 = compute_F_theta(k3_state, t + 0.5 * dt)

        k4_state = E_fast + dt * k3
        k4 = compute_F_theta(k4_state, t + dt)

        # 组合更新
        E_new = E_fast + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # 状态范数裁剪（防止发散）
        norm = torch.norm(E_new).item()
        if norm > self.fast_dynamics.config.state_norm_threshold:
            scale = self.fast_dynamics.config.state_norm_threshold / norm
            E_new = E_new * scale
            logger.debug(f"RK4 state norm clipped: {norm:.4e} -> {self.fast_dynamics.config.state_norm_threshold}")

        # 稳定性检查
        self.fast_dynamics._check_stability(E_new)

        return E_new

    def integrate(
        self,
        initial_state: SelfState,
        inputs_sequence: Optional[List[ExternalInput]] = None,
        t_span: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        record_trajectory: bool = True
    ) -> List[SelfState]:
        """
        执行多步积分

        Args:
            initial_state: 初始状态
            inputs_sequence: 输入序列（可选）
            t_span: 时间点序列（可选）
            num_steps: 步数（可选）
            record_trajectory: 是否记录轨迹

        Returns:
            状态轨迹列表
        """
        if not self._initialized:
            raise ValueError("Engine not initialized.")

        # 生成时间序列
        if t_span is None:
            if num_steps is None:
                num_steps = 1000

            dt = self.engine_config.default_dt
            t_span = torch.tensor(
                [i * dt for i in range(num_steps)],
                device=self.device
            )

        num_steps = t_span.shape[0]

        # 初始化轨迹
        trajectory = [initial_state.copy()]
        current_state = initial_state

        # 逐步积分
        for i in range(1, num_steps):
            # 获取输入（如果有）
            inputs = None
            if inputs_sequence is not None and len(inputs_sequence) > i:
                inputs = inputs_sequence[i]

            # 执行单步
            dt_i = (t_span[i] - t_span[i-1]).item()
            current_state = self.step(
                current_state, inputs, dt=dt_i
            )

            if record_trajectory:
                trajectory.append(current_state.copy())

        logger.info(
            f"Integration completed: steps={num_steps}, "
            f"duration={t_span[-1].item():.2f}s"
        )

        return trajectory

    def run_continuous(
        self,
        duration_hours: float,
        initial_state: Optional[SelfState] = None,
        inputs_generator: Optional[callable] = None,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        连续运行引擎（长时间尺度）

        Args:
            duration_hours: 运行时长（小时）
            initial_state: 初始状态（可选）
            inputs_generator: 输入生成器（可选）
            callback: 回调函数（可选）

        Returns:
            运行统计结果
        """
        if not self._initialized:
            raise ValueError("Engine not initialized.")

        # 创建初始状态（如果未提供）
        if initial_state is None:
            initial_state = SelfState(
                E_fast=torch.zeros(self.engine_config.fast_dim),
                E_slow=torch.zeros(self.engine_config.slow_dim),
                timestamp=0.0
            )

        # 计算总步数（假设 dt=0.01s，1小时=3600秒）
        dt = self.engine_config.default_dt
        total_steps = int(duration_hours * 3600 / dt)

        logger.info(
            f"Starting continuous run: duration={duration_hours}h, "
            f"total_steps={total_steps}"
        )

        start_time = time.time()

        # 运行统计
        stats = {
            "start_time": start_time,
            "duration_hours": duration_hours,
            "total_steps": total_steps,
            "actual_steps": 0,
            "final_time": 0.0,
            "fast_updates": 0,
            "slow_updates": 0,
            "stability_warnings": 0,
            "is_stable": True,
            "is_edge_of_chaos": True,
            "state_history_length": 0
        }

        current_state = initial_state

        for step_idx in range(total_steps):
            # 生成输入（如果有生成器）
            inputs = None
            if inputs_generator is not None:
                inputs = inputs_generator(step_idx, current_state)

            # 执行单步
            current_state = self.step(current_state, inputs, dt=dt)

            # 回调
            if callback is not None:
                callback(step_idx, current_state, self)

            # 检查稳定性
            if not self.coupling_system.check_edge_of_chaos():
                stats["is_edge_of_chaos"] = False

            # 检查是否需要停止（严重不稳定）
            if self.stability_warnings > 10:
                logger.warning("Too many stability warnings, stopping run")
                stats["is_stable"] = False
                break

        # 记录最终统计
        stats["actual_steps"] = self.step_count
        stats["final_time"] = self.total_time
        stats["fast_updates"] = self.fast_updates
        stats["slow_updates"] = self.slow_updates
        stats["stability_warnings"] = self.stability_warnings
        stats["state_history_length"] = len(self.state_history)
        stats["actual_duration_hours"] = self.total_time / 3600

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = elapsed

        logger.info(
            f"Continuous run completed: "
            f"steps={stats['actual_steps']}, "
            f"hours={stats['actual_duration_hours']:.2f}, "
            f"stable={stats['is_stable']}, "
            f"edge_of_chaos={stats['is_edge_of_chaos']}"
        )

        return stats

    def get_state_monitoring(self) -> Dict:
        """
        获取状态监测报告

        Returns:
            监测报告字典
        """
        report = {
            "step_count": self.step_count,
            "total_time": self.total_time,
            "fast_updates": self.fast_updates,
            "slow_updates": self.slow_updates,
            "stability_warnings": self.stability_warnings,
            "history_length": len(self.state_history),
            "coupling_report": self.coupling_system.get_stability_report(),
            "fast_dynamics_stats": self.fast_dynamics.get_statistics(),
            "slow_dynamics_stats": self.slow_dynamics.get_statistics(),
            "dmn_stats": self.dmn.get_statistics(),
            "solver_stats": self.ode_solver.get_statistics()
        }

        return report

    def reset(self) -> None:
        """重置引擎"""
        # 重置组件
        if self.fast_dynamics:
            self.fast_dynamics.reset()
        if self.slow_dynamics:
            self.slow_dynamics.reset()
        if self.coupling_system:
            self.coupling_system.reset()
        if self.dmn:
            self.dmn.reset()
        if self.ode_solver:
            self.ode_solver.reset()

        # 重置状态管理器
        if self.state_manager:
            active_state_id = self.state_manager.get_active_state_id()
            if active_state_id:
                self.state_manager.reset(active_state_id)

        # 重置统计
        self.step_count = 0
        self.total_time = 0.0
        self.fast_updates = 0
        self.slow_updates = 0
        self.stability_warnings = 0

        # 清空历史
        self.state_history.clear()

        # 清空缓存
        self._current_E_fast_prev = None
        self._current_E_slow_prev = None

        # 清空 IMEX 相关缓存
        self._J0_cache = None
        self._J0_update_counter = 0
        self._last_input_dict = None

        logger.info("IntegrationEngine reset")

    def save_state(self, filepath: str) -> None:
        """
        保存当前状态到文件

        Args:
            filepath: 文件路径
        """
        import json

        if self.state_history:
            current_state = self.state_history[-1]
        else:
            current_state = SelfState(
                E_fast=torch.zeros(self.engine_config.fast_dim),
                E_slow=torch.zeros(self.engine_config.slow_dim)
            )

        state_data = {
            "state": current_state.to_dict(),
            "statistics": self.get_state_monitoring(),
            "config": {
                "fast_dim": self.engine_config.fast_dim,
                "slow_dim": self.engine_config.slow_dim,
                "device": self.device
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"State saved to {filepath}")

    def load_state(self, filepath: str) -> SelfState:
        """
        从文件加载状态

        Args:
            filepath: 文件路径

        Returns:
            加载的状态
        """
        import json

        with open(filepath, 'r') as f:
            state_data = json.load(f)

        state = SelfState.from_dict(state_data["state"])

        logger.info(f"State loaded from {filepath}")

        return state

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"IntegrationEngine(status={status}, "
            f"steps={self.step_count}, time={self.total_time:.2f}s)"
        )


def create_integration_engine_from_config(
    config: ChronosConfig,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> IntegrationEngine:
    """
    从全局配置创建积分引擎

    Args:
        config: 全局配置
        device: 计算设备
        seed: 随机种子

    Returns:
        IntegrationEngine 实例
    """
    engine_config = IntegrationEngineConfig(
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
        slow_update_frequency=config.memory_temporal.slow_update_frequency,
        solver_method=config.neural_ode.integration_method,
        solver_atol=config.neural_ode.atol,
        solver_rtol=config.neural_ode.rtol,
        # IMEX 相关配置
        solver_type=config.numerics.solver_type,
        imex_update_interval=config.numerics.imex_update_interval,
        imex_j0_eps=1e-6,  # 默认值
        imex_j0_clamp_min=0.0  # 默认值
    )

    engine = IntegrationEngine(
        config=config,
        engine_config=engine_config,
        device=device,
        seed=seed
    )

    engine.initialize()
    return engine