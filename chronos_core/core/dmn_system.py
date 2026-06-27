"""
默认模式网络（DMN）系统
======================

整合混沌吸引子、注入机制、增益控制，实现完整的自持动力学系统。

核心功能：
- 支持无外部输入时的持续动力学
- 自适应增益控制
- 多吸引子随机切换
- 系统状态监测和混沌注入效果评估
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from dataclasses import dataclass, field
import time

from chronos_core.utils.config import ChaosInjectionConfig, DimensionalityConfig

from .chaos import (
    AttractorManager,
    LorenzAttractor,
    RosslerAttractor,
    ChuaAttractor
)
from .chaos_injector import ChaosInjector, CoreSubspaceProjector

logger = logging.getLogger(__name__)


@dataclass
class DMNConfig:
    """DMN 系统配置"""
    # 维度配置
    fast_variable_dim: int = 2048
    slow_variable_dim: int = 512
    core_subspace_dim: int = 64

    # 混沌注入配置
    base_gain: float = 0.1
    target_variance: float = 1.0
    gain_smoothing: float = 0.95

    # 吸引子切换配置
    switch_interval: int = 5000
    transition_steps: int = 100
    attractor_smoothing: float = 0.95

    # 监控配置
    stability_threshold: float = 100.0
    monitoring_interval: int = 100

    # 验证配置
    max_continuous_hours: float = 72.0  # 最大连续运行时间（小时）


@dataclass
class DMNState:
    """DMN 系统状态"""
    # 核心状态
    E_fast: torch.Tensor = None
    E_slow: torch.Tensor = None
    E_fast_core: torch.Tensor = None

    # 时间追踪
    simulation_time: float = 0.0
    step_count: int = 0

    # 统计信息
    injection_count: int = 0
    attractor_switches: int = 0
    variance_history: List[float] = field(default_factory=list)
    gain_history: List[float] = field(default_factory=list)

    # 稳定性状态
    is_stable: bool = True
    last_stability_check: float = 0.0


class DefaultModeNetwork:
    """
    默认模式网络系统

    整合混沌吸引子库、注入机制、增益控制，提供完整的内源性动力学支持。

    主要功能：
    1. 无外部输入时的持续动力学维持
    2. 自适应增益计算（基于核心子空间方差）
    3. 多吸引子随机切换机制
    4. 切换时的平滑插值过渡
    5. 系统状态监测和稳定性保障

    使用示例：
        dmn = DefaultModeNetwork(config=DMNConfig())
        dmn.initialize()

        # 无输入持续运行
        for _ in range(10000):
            B = dmn.step()  # 获取混沌注入信号
            E_fast += B  # 将注入信号加入快变量
    """

    def __init__(
        self,
        config: Optional[DMNConfig] = None,
        chaos_config: Optional[ChaosInjectionConfig] = None,
        dim_config: Optional[DimensionalityConfig] = None,
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        初始化 DMN 系统

        Args:
            config: DMN 配置
            chaos_config: 混沌注入配置（来自全局配置）
            dim_config: 维度配置（来自全局配置）
            device: 计算设备
            seed: 随机种子
        """
        # 合并配置
        self.config = config or DMNConfig()

        if chaos_config:
            self.config.base_gain = chaos_config.chaos_injection_gain
            self.config.switch_interval = chaos_config.attractor_switch_interval
            self.config.attractor_smoothing = chaos_config.attractor_transition_smoothing

        if dim_config:
            self.config.fast_variable_dim = dim_config.fast_variable_dim
            self.config.slow_variable_dim = dim_config.slow_variable_dim
            self.config.core_subspace_dim = dim_config.core_subspace_dim

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # 随机种子
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 核心组件
        self.attractor_manager: Optional[AttractorManager] = None
        self.chaos_injector: Optional[ChaosInjector] = None
        self.core_projector: Optional[CoreSubspaceProjector] = None

        # 系统状态
        self.state: Optional[DMNState] = None

        # 监控数据
        self.monitoring_data: Dict = {}

        # 初始化标志
        self._initialized = False

        logger.info(
            f"DefaultModeNetwork created: core_dim={self.config.core_subspace_dim}, "
            f"switch_interval={self.config.switch_interval}"
        )

    def initialize(self) -> None:
        """
        初始化 DMN 系统所有组件

        包括：
        - 吸引子管理器
        - 混沌注入器
        - 核心子空间投影器
        - 系统状态
        """
        # 创建吸引子管理器
        self.attractor_manager = AttractorManager(
            core_subspace_dim=self.config.core_subspace_dim,
            switch_interval=self.config.switch_interval,
            transition_steps=self.config.transition_steps,
            smoothing_factor=self.config.attractor_smoothing,
            device=self.device,
            seed=self.seed
        )

        # 注册默认吸引子
        self.attractor_manager.register_default_attractors()

        # 创建混沌注入器
        self.chaos_injector = ChaosInjector(
            core_subspace_dim=self.config.core_subspace_dim,
            chaos_dim=3,
            base_gain=self.config.base_gain,
            target_variance=self.config.target_variance,
            device=self.device,
            seed=self.seed
        )

        # 创建核心子空间投影器
        self.core_projector = CoreSubspaceProjector(
            full_dim=self.config.fast_variable_dim,
            core_dim=self.config.core_subspace_dim,
            device=self.device,
            seed=self.seed + 1 if self.seed is not None else None
        )

        # 初始化状态
        self.state = DMNState(
            E_fast=torch.zeros(self.config.fast_variable_dim, device=self.device),
            E_slow=torch.zeros(self.config.slow_variable_dim, device=self.device),
            E_fast_core=torch.zeros(self.config.core_subspace_dim, device=self.device),
            simulation_time=0.0,
            step_count=0
        )

        # 初始化吸引子状态
        self.attractor_manager.initialize_states(seed=self.seed)

        self._initialized = True

        logger.info("DefaultModeNetwork initialized successfully")

    def step(
        self,
        dt: Optional[float] = None,
        E_fast: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        执行单步 DMN 演化

        Args:
            dt: 时间步长
            E_fast: 外部提供的快变量状态（如果为None，使用内部状态）

        Returns:
            混沌注入信号 B (fast_variable_dim,)
        """
        if not self._initialized:
            raise ValueError("DMN system not initialized. Call initialize() first.")

        dt = dt or 0.01

        # 更新快变量状态（如果提供了外部状态）
        if E_fast is not None:
            self.state.E_fast = E_fast.to(self.device)

        # 投影到核心子空间
        self.state.E_fast_core = self.core_projector.project(self.state.E_fast)

        # 执行吸引子演化
        z_chaos = self.attractor_manager.step(dt)

        # 计算混沌注入信号
        B_core = self.chaos_injector.inject(z_chaos, self.state.E_fast_core)

        # 扩展到全维度
        B_full = self.core_projector.inject_to_full(B_core)

        # 更新状态统计
        self._update_state(B_full, dt)

        # 稳定性检查
        if self.state.step_count % self.config.monitoring_interval == 0:
            self._check_stability()

        return B_full

    def _update_state(self, B: torch.Tensor, dt: float) -> None:
        """更新系统状态"""
        # 将注入信号加入快变量
        self.state.E_fast = self.state.E_fast + B * dt

        # 更新时间
        self.state.simulation_time += dt
        self.state.step_count += 1
        self.state.injection_count += 1

        # 记录方差和增益历史
        variance = torch.var(self.state.E_fast_core).item()
        self.state.variance_history.append(variance)
        self.state.gain_history.append(self.chaos_injector.current_gain)

        # 限制历史长度
        max_history = 1000
        if len(self.state.variance_history) > max_history:
            self.state.variance_history = self.state.variance_history[-max_history:]
        if len(self.state.gain_history) > max_history:
            self.state.gain_history = self.state.gain_history[-max_history:]

    def _check_stability(self) -> None:
        """检查系统稳定性"""
        # 检查快变量范数
        E_fast_norm = torch.norm(self.state.E_fast).item()

        # 检查是否发散
        if E_fast_norm > self.config.stability_threshold:
            self.state.is_stable = False
            logger.warning(
                f"DMN system becoming unstable: E_fast_norm={E_fast_norm:.4e}, "
                f"threshold={self.config.stability_threshold}"
            )
            # 自动降低增益
            self.chaos_injector.current_gain *= 0.5

        # 检查 NaN 和 Inf
        if torch.isnan(self.state.E_fast).any() or torch.isinf(self.state.E_fast).any():
            self.state.is_stable = False
            logger.error("DMN state contains NaN or Inf values!")
            # 重置快变量
            self.state.E_fast = torch.zeros_like(self.state.E_fast)

        self.state.last_stability_check = self.state.simulation_time

        # 更新吸引子切换计数
        self.state.attractor_switches = len(self.attractor_manager.switch_history)

    def run_continuous(
        self,
        duration_hours: float,
        dt: float = 0.01,
        callback: Optional[callable] = None
    ) -> Dict:
        """
        连续运行 DMN 系统（无外部输入）

        验证系统在无输入时能否维持持续动力学。

        Args:
            duration_hours: 运行时长（小时）
            dt: 时间步长
            callback: 每步回调函数

        Returns:
            运行统计结果
        """
        if not self._initialized:
            raise ValueError("DMN system not initialized. Call initialize() first.")

        # 计算总步数
        # 假设 dt 是秒级，1小时 = 3600秒
        total_steps = int(duration_hours * 3600 / dt)

        logger.info(
            f"Starting continuous DMN run: duration={duration_hours}h, "
            f"total_steps={total_steps}"
        )

        start_time = time.time()

        # 运行统计
        stats = {
            "start_time": start_time,
            "duration_hours": duration_hours,
            "total_steps": total_steps,
            "final_time": 0.0,
            "final_gain": 0.0,
            "final_variance": 0.0,
            "attractor_switches": 0,
            "stability_maintained": True,
            "E_fast_norm_final": 0.0,
            "variance_trajectory": [],
            "gain_trajectory": []
        }

        for step_idx in range(total_steps):
            # 执行单步
            B = self.step(dt)

            # 回调处理
            if callback is not None:
                callback(step_idx, B, self.state)

            # 定期记录轨迹
            if step_idx % 1000 == 0:
                stats["variance_trajectory"].append(
                    torch.var(self.state.E_fast_core).item()
                )
                stats["gain_trajectory"].append(self.chaos_injector.current_gain)

                logger.debug(
                    f"Step {step_idx}/{total_steps}: "
                    f"time={self.state.simulation_time:.2f}s, "
                    f"gain={self.chaos_injector.current_gain:.4f}, "
                    f"E_norm={torch.norm(self.state.E_fast).item():.4f}"
                )

            # 稳定性检查
            if not self.state.is_stable:
                stats["stability_maintained"] = False
                logger.warning(f"Stability lost at step {step_idx}")
                break

        # 记录最终状态
        stats["final_time"] = self.state.simulation_time
        stats["final_gain"] = self.chaos_injector.current_gain
        stats["final_variance"] = torch.var(self.state.E_fast_core).item()
        stats["attractor_switches"] = self.state.attractor_switches
        stats["E_fast_norm_final"] = torch.norm(self.state.E_fast).item()
        stats["actual_duration_hours"] = self.state.simulation_time / 3600

        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = elapsed

        logger.info(
            f"Continuous DMN run completed: "
            f"steps={self.state.step_count}, "
            f"hours={stats['actual_duration_hours']:.2f}, "
            f"switches={stats['attractor_switches']}, "
            f"stable={stats['stability_maintained']}"
        )

        return stats

    def get_injection_signal(self) -> torch.Tensor:
        """
        获取当前的混沌注入信号（不执行演化）

        Returns:
            注入信号 (fast_variable_dim,)
        """
        if not self._initialized:
            raise ValueError("DMN system not initialized.")

        z = self.attractor_manager.get_state()
        B_core = self.chaos_injector.inject(z)
        B_full = self.core_projector.inject_to_full(B_core)
        return B_full

    def get_statistics(self) -> Dict:
        """
        获取 DMN 系统完整统计信息

        Returns:
            统计信息字典
        """
        if not self._initialized:
            return {"status": "not_initialized"}

        stats = {
            "system": {
                "initialized": self._initialized,
                "simulation_time": self.state.simulation_time,
                "step_count": self.state.step_count,
                "injection_count": self.state.injection_count,
                "is_stable": self.state.is_stable
            },
            "attractor": self.attractor_manager.get_statistics(),
            "injector": self.chaos_injector.get_statistics(),
            "state": {
                "E_fast_norm": torch.norm(self.state.E_fast).item() if self.state.E_fast is not None else 0,
                "E_slow_norm": torch.norm(self.state.E_slow).item() if self.state.E_slow is not None else 0,
                "E_fast_core_norm": torch.norm(self.state.E_fast_core).item() if self.state.E_fast_core is not None else 0,
                "E_fast_core_variance": torch.var(self.state.E_fast_core).item() if self.state.E_fast_core is not None else 0,
                "average_variance": np.mean(self.state.variance_history) if self.state.variance_history else 0.0,
                "average_gain": np.mean(self.state.gain_history) if self.state.gain_history else self.config.base_gain
            },
            "config": {
                "core_subspace_dim": self.config.core_subspace_dim,
                "fast_variable_dim": self.config.fast_variable_dim,
                "switch_interval": self.config.switch_interval,
                "base_gain": self.config.base_gain,
                "target_variance": self.config.target_variance
            }
        }

        return stats

    def set_state(
        self,
        E_fast: torch.Tensor,
        E_slow: Optional[torch.Tensor] = None
    ) -> None:
        """
        设置系统状态

        Args:
            E_fast: 快变量状态
            E_slow: 慢变量状态（可选）
        """
        if not self._initialized:
            raise ValueError("DMN system not initialized.")

        self.state.E_fast = E_fast.to(self.device)
        if E_slow is not None:
            self.state.E_slow = E_slow.to(self.device)

        # 更新核心子空间投影
        self.state.E_fast_core = self.core_projector.project(self.state.E_fast)

        logger.debug(f"DMN state updated: E_fast_norm={torch.norm(E_fast).item():.4f}")

    def reset(self, seed: Optional[int] = None) -> None:
        """
        重置 DMN 系统

        Args:
            seed: 新的随机种子
        """
        if seed is not None:
            self.seed = seed
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 重置组件
        if self.attractor_manager:
            self.attractor_manager.reset(seed)

        if self.chaos_injector:
            self.chaos_injector.reset(seed)

        # 重置状态
        if self.state:
            self.state.E_fast = torch.zeros(self.config.fast_variable_dim, device=self.device)
            self.state.E_slow = torch.zeros(self.config.slow_variable_dim, device=self.device)
            self.state.E_fast_core = torch.zeros(self.config.core_subspace_dim, device=self.device)
            self.state.simulation_time = 0.0
            self.state.step_count = 0
            self.state.injection_count = 0
            self.state.attractor_switches = 0
            self.state.variance_history.clear()
            self.state.gain_history.clear()
            self.state.is_stable = True

        logger.info("DefaultModeNetwork reset")

    def force_attractor_switch(self, target_name: Optional[str] = None) -> None:
        """
        强制切换吸引子

        Args:
            target_name: 目标吸引子名称
        """
        if self.attractor_manager:
            self.attractor_manager.force_switch(target_name)
            self.state.attractor_switches = len(self.attractor_manager.switch_history)

    def set_gain(self, gain: float) -> None:
        """
        手动设置注入增益

        Args:
            gain: 新的增益值
        """
        if self.chaos_injector:
            self.chaos_injector.set_gain(gain)

    def save_state(self, filepath: str) -> None:
        """
        保存 DMN 系统状态

        Args:
            filepath: 文件路径
        """
        import json

        state_data = {
            "dmn_state": {
                "E_fast": self.state.E_fast.detach().cpu().numpy().tolist(),
                "E_slow": self.state.E_slow.detach().cpu().numpy().tolist(),
                "simulation_time": self.state.simulation_time,
                "step_count": self.state.step_count,
                "injection_count": self.state.injection_count,
                "attractor_switches": self.state.attractor_switches,
                "variance_history": self.state.variance_history[-100:] if self.state.variance_history else [],
                "gain_history": self.state.gain_history[-100:] if self.state.gain_history else [],
                "is_stable": self.state.is_stable
            },
            "attractor_manager": self.attractor_manager.to_dict(),
            "injector": {
                "current_gain": self.chaos_injector.current_gain,
                "total_injections": self.chaos_injector.total_injections,
                "projection_matrix": self.chaos_injector.W.detach().cpu().numpy().tolist()
            },
            "config": {
                "core_subspace_dim": self.config.core_subspace_dim,
                "fast_variable_dim": self.config.fast_variable_dim,
                "base_gain": self.config.base_gain,
                "switch_interval": self.config.switch_interval
            }
        }

        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)

        logger.info(f"DMN state saved to {filepath}")

    def load_state(self, filepath: str) -> None:
        """
        加载 DMN 系统状态

        Args:
            filepath: 文件路径
        """
        import json

        with open(filepath, 'r') as f:
            state_data = json.load(f)

        # 恢复状态
        dmn_state = state_data["dmn_state"]
        self.state.E_fast = torch.tensor(dmn_state["E_fast"], dtype=torch.float32, device=self.device)
        self.state.E_slow = torch.tensor(dmn_state["E_slow"], dtype=torch.float32, device=self.device)
        self.state.simulation_time = dmn_state["simulation_time"]
        self.state.step_count = dmn_state["step_count"]
        self.state.injection_count = dmn_state["injection_count"]
        self.state.attractor_switches = dmn_state["attractor_switches"]
        self.state.variance_history = dmn_state["variance_history"]
        self.state.gain_history = dmn_state["gain_history"]
        self.state.is_stable = dmn_state["is_stable"]

        # 恢复核心子空间投影
        self.state.E_fast_core = self.core_projector.project(self.state.E_fast)

        # 恢复注入器状态
        injector_state = state_data["injector"]
        self.chaos_injector.current_gain = injector_state["current_gain"]
        self.chaos_injector.total_injections = injector_state["total_injections"]

        logger.info(f"DMN state loaded from {filepath}")

    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"DefaultModeNetwork(status={status}, "
            f"core_dim={self.config.core_subspace_dim}, "
            f"steps={self.state.step_count if self.state else 0}, "
            f"time={self.state.simulation_time if self.state else 0:.2f}s)"
        )


def create_dmn_from_config(
    chaos_config: ChaosInjectionConfig,
    dim_config: DimensionalityConfig,
    device: Optional[str] = None,
    seed: Optional[int] = None
) -> DefaultModeNetwork:
    """
    从全局配置创建 DMN 系统

    Args:
        chaos_config: 混沌注入配置
        dim_config: 维度配置
        device: 计算设备
        seed: 随机种子

    Returns:
        DefaultModeNetwork 实例
    """
    dmn_config = DMNConfig(
        fast_variable_dim=dim_config.fast_variable_dim,
        slow_variable_dim=dim_config.slow_variable_dim,
        core_subspace_dim=dim_config.core_subspace_dim,
        base_gain=chaos_config.chaos_injection_gain,
        switch_interval=chaos_config.attractor_switch_interval,
        attractor_smoothing=chaos_config.attractor_transition_smoothing
    )

    dmn = DefaultModeNetwork(
        config=dmn_config,
        chaos_config=chaos_config,
        dim_config=dim_config,
        device=device,
        seed=seed
    )

    dmn.initialize()
    return dmn