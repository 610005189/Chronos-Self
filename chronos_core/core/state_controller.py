"""
状态控制器
============

实现系统的状态模式切换和平滑过渡功能。

核心功能：
- 定义三种状态模式：REST（静息）、WORK（工作）、EXPLORE（探索）
- 实现参数平滑插值过渡
- 支持手动状态切换
- 状态切换响应时间 < 100 步
"""

import torch
from enum import Enum
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)


class StateMode(Enum):
    """状态模式枚举
    
    REST: 静息状态
        - 低活跃度，高稳定性
        - 适合系统恢复和能量保存
        
    WORK: 工作状态
        - 中等活跃度，稳定性平衡
        - 适合任务执行和认知加工
        
    EXPLORE: 探索状态
        - 高活跃度，适度混沌
        - 适合探索、学习和创新
    """
    REST = "rest"
    WORK = "work"
    EXPLORE = "explore"


@dataclass
class StateParameters:
    """状态参数配置
    
    定义每种状态下系统的动力学参数。
    """
    # 动力学参数
    decay_rate: float = 0.85  # 自然衰减率
    gamma: float = 0.5  # 线性耗散系数（激活函数斜率限制）
    dynamics_scale: float = 1.0  # 演化函数输出缩放因子
    noise_scale: float = 0.00001  # 内部噪声强度
    
    # E/I 平衡参数
    ei_ratio: float = 4.0  # E维数 / I维数
    alpha: float = 0.1  # 抑制反馈增益
    e_target: float = 0.0  # 兴奋目标均值
    
    # 稳定性参数
    state_norm_threshold: float = 100.0  # 状态范数阈值（紧急裁剪）
    state_norm_clip: float = 0.0  # 状态范数主动截断（>0时启用）
    max_gradient_norm: float = 10.0  # 梯度裁剪阈值
    
    # 逐层谱约束参数
    target_spectral_norm: float = 1.9  # 每层权重的目标谱范数
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'decay_rate': self.decay_rate,
            'gamma': self.gamma,
            'dynamics_scale': self.dynamics_scale,
            'noise_scale': self.noise_scale,
            'ei_ratio': self.ei_ratio,
            'alpha': self.alpha,
            'e_target': self.e_target,
            'state_norm_threshold': self.state_norm_threshold,
            'state_norm_clip': self.state_norm_clip,
            'max_gradient_norm': self.max_gradient_norm,
            'target_spectral_norm': self.target_spectral_norm,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'StateParameters':
        """从字典创建"""
        return cls(**data)

    def copy(self) -> 'StateParameters':
        """创建副本"""
        return StateParameters(**self.to_dict())


# 预定义状态参数配置
STATE_PARAMS_CONFIG: Dict[StateMode, StateParameters] = {
    StateMode.REST: StateParameters(
        decay_rate=0.0,  # 无衰减（逐层谱约束已控制稳定性）
        gamma=0.0,  # 关闭激活函数斜率限制（逐层谱约束已足够）
        dynamics_scale=0.8,  # 低动力学强度
        noise_scale=0.000001,  # 极低噪声
        ei_ratio=4.0,
        alpha=0.0,  # E/I平衡关闭
        e_target=0.0,
        state_norm_threshold=50.0,  # 紧急裁剪阈值
        state_norm_clip=0.1,  # 主动范数截断
        max_gradient_norm=5.0,
        target_spectral_norm=2.1,  # 更高谱范数 = 更稳定（更大全局收缩）
    ),
    
    StateMode.WORK: StateParameters(
        decay_rate=0.5,  # 适度衰减，确保系统有界
        gamma=0.0,  # 关闭激活函数斜率限制
        dynamics_scale=7.0,  # 中等动力学强度，产生温和混沌（λ_max≈0.17）
        noise_scale=0.00001,  # 默认噪声
        ei_ratio=4.0,
        alpha=0.0,  # E/I平衡关闭
        e_target=0.0,
        state_norm_threshold=200.0,  # 紧急裁剪阈值
        state_norm_clip=0.0,  # 关闭主动范数截断，让系统自然演化
        max_gradient_norm=200.0,  # 足够大，不裁剪真实动力学
        target_spectral_norm=1.5,  # 逐层谱约束（温和混沌边缘）
    ),
    
    StateMode.EXPLORE: StateParameters(
        decay_rate=0.1,  # 低衰减，允许更多探索
        gamma=0.0,  # 关闭激活函数斜率限制
        dynamics_scale=3.0,  # 高动力学强度，增强探索性（λ_max≈0.13）
        noise_scale=0.00005,  # 较高噪声，增强探索
        ei_ratio=4.0,
        alpha=0.0,  # E/I平衡关闭
        e_target=0.0,
        state_norm_threshold=200.0,  # 高阈值
        state_norm_clip=0.0,  # 关闭主动范数截断，让系统自由探索
        max_gradient_norm=200.0,  # 足够大，不裁剪真实动力学
        target_spectral_norm=1.5,  # 谱约束
    ),
}


@dataclass
class TransitionState:
    """过渡状态信息"""
    current_mode: StateMode = StateMode.REST
    target_mode: Optional[StateMode] = None
    transition_progress: float = 0.0  # 0.0 到 1.0
    transition_steps: int = 0
    total_transition_steps: int = 50  # 默认过渡步数
    start_params: Optional[StateParameters] = None
    target_params: Optional[StateParameters] = None
    current_params: Optional[StateParameters] = None
    
    is_transitioning: bool = field(default=False, init=False)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.current_params is None:
            self.current_params = STATE_PARAMS_CONFIG[self.current_mode]


class StateController:
    """状态控制器
    
    管理系统状态模式的切换和平滑过渡。
    
    核心功能：
    1. 状态切换：switch_state(mode) 开始状态切换
    2. 参数过渡：step_transition() 执行一步参数插值
    3. 参数获取：get_current_params() 获取当前参数
    4. 状态查询：is_transitioning() 检查是否正在过渡
    
    使用示例：
        controller = StateController()
        controller.switch_state(StateMode.WORK, transition_steps=50)
        
        # 在演化循环中
        for i in range(100):
            params = controller.get_current_params()
            # 使用 params 更新动力学系统
            controller.step_transition()
    """
    
    def __init__(
        self,
        initial_mode: StateMode = StateMode.REST,
        default_transition_steps: int = 50,
        interpolation_method: str = "linear"
    ):
        """
        初始化状态控制器
        
        Args:
            initial_mode: 初始状态模式
            default_transition_steps: 默认过渡步数
            interpolation_method: 插值方法（"linear" 或 "cosine")
        """
        self.default_transition_steps = default_transition_steps
        self.interpolation_method = interpolation_method
        
        # 初始化状态
        self._transition_state = TransitionState(
            current_mode=initial_mode,
            current_params=STATE_PARAMS_CONFIG[initial_mode].copy(),
            total_transition_steps=default_transition_steps
        )
        
        # 统计信息
        self._transition_count = 0
        self._total_transition_time = 0
        self._mode_history: list = [(initial_mode, 0)]
        self._step_count = 0
        
        logger.info(
            f"StateController initialized: mode={initial_mode.value}, "
            f"default_transition_steps={default_transition_steps}"
        )
    
    def switch_state(
        self,
        target_mode: StateMode,
        transition_steps: Optional[int] = None,
        force: bool = False
    ) -> bool:
        """
        开始状态切换
        
        Args:
            target_mode: 目标状态模式
            transition_steps: 过渡步数（None 使用默认值）
            force: 是否强制切换（立即完成）
            
        Returns:
            是否成功开始切换
        """
        # 如果正在过渡，不允许新切换（除非强制）
        if self._transition_state.is_transitioning and not force:
            logger.warning(
                f"Cannot switch to {target_mode.value}: "
                f"currently transitioning to {self._transition_state.target_mode.value}"
            )
            return False
        
        # 如果目标状态与当前状态相同，直接返回
        if target_mode == self._transition_state.current_mode and not self._transition_state.is_transitioning:
            logger.debug(f"Already in {target_mode.value} mode, no transition needed")
            return True
        
        # 设置过渡参数
        steps = transition_steps or self.default_transition_steps
        
        # 强制切换：立即完成
        if force:
            self._transition_state.current_mode = target_mode
            self._transition_state.current_params = STATE_PARAMS_CONFIG[target_mode].copy()
            self._transition_state.is_transitioning = False
            self._transition_state.target_mode = None
            
            # 记录历史
            self._mode_history.append((target_mode, self._step_count))
            self._transition_count += 1
            
            logger.info(f"Force switched to {target_mode.value}")
            return True
        
        # 开始正常过渡
        self._transition_state.target_mode = target_mode
        self._transition_state.total_transition_steps = steps
        self._transition_state.transition_steps = 0
        self._transition_state.transition_progress = 0.0
        self._transition_state.start_params = self._transition_state.current_params.copy()
        self._transition_state.target_params = STATE_PARAMS_CONFIG[target_mode].copy()
        self._transition_state.is_transitioning = True
        
        logger.info(
            f"Starting transition: {self._transition_state.current_mode.value} → {target_mode.value}, "
            f"steps={steps}"
        )
        
        return True
    
    def step_transition(self) -> Tuple[StateParameters, bool]:
        """
        执行一步参数过渡
        
        Returns:
            (current_params, transition_complete): 当前参数和过渡是否完成
        """
        self._step_count += 1
        
        # 如果不在过渡中，直接返回当前参数
        if not self._transition_state.is_transitioning:
            return self._transition_state.current_params, False
        
        # 更新过渡进度
        self._transition_state.transition_steps += 1
        
        # 计算过渡进度（0.0 到 1.0）
        progress = self._transition_state.transition_steps / self._transition_state.total_transition_steps
        self._transition_state.transition_progress = min(progress, 1.0)
        
        # 执行参数插值
        interpolated_params = self._interpolate_params(
            self._transition_state.start_params,
            self._transition_state.target_params,
            self._transition_state.transition_progress
        )
        
        self._transition_state.current_params = interpolated_params
        
        # 检查是否完成过渡
        if self._transition_state.transition_progress >= 1.0:
            # 过渡完成
            self._transition_state.current_mode = self._transition_state.target_mode
            self._transition_state.is_transitioning = False
            self._transition_state.target_mode = None
            
            # 记录历史和统计
            self._mode_history.append((self._transition_state.current_mode, self._step_count))
            self._transition_count += 1
            self._total_transition_time += self._transition_state.total_transition_steps
            
            logger.info(
                f"Transition complete: now in {self._transition_state.current_mode.value}, "
                f"steps={self._transition_state.transition_steps}"
            )
            
            return interpolated_params, True
        
        return interpolated_params, False
    
    def _interpolate_params(
        self,
        start: StateParameters,
        target: StateParameters,
        progress: float
    ) -> StateParameters:
        """
        参数插值
        
        Args:
            start: 起始参数
            target: 目标参数
            progress: 过渡进度（0.0 到 1.0）
            
        Returns:
            插值后的参数
        """
        if self.interpolation_method == "linear":
            # 线性插值
            weight = progress
        elif self.interpolation_method == "cosine":
            # 余弦插值（平滑开始和结束）
            weight = 0.5 * (1 - math.cos(math.pi * progress))
        else:
            weight = progress
        
        # 插值各参数
        return StateParameters(
            decay_rate=start.decay_rate + weight * (target.decay_rate - start.decay_rate),
            gamma=start.gamma + weight * (target.gamma - start.gamma),
            dynamics_scale=start.dynamics_scale + weight * (target.dynamics_scale - start.dynamics_scale),
            noise_scale=start.noise_scale + weight * (target.noise_scale - start.noise_scale),
            ei_ratio=start.ei_ratio + weight * (target.ei_ratio - start.ei_ratio),
            alpha=start.alpha + weight * (target.alpha - start.alpha),
            e_target=start.e_target + weight * (target.e_target - start.e_target),
            state_norm_threshold=start.state_norm_threshold + weight * (target.state_norm_threshold - start.state_norm_threshold),
            max_gradient_norm=start.max_gradient_norm + weight * (target.max_gradient_norm - start.max_gradient_norm),
        )
    
    def get_current_params(self) -> StateParameters:
        """获取当前参数"""
        return self._transition_state.current_params
    
    def get_current_mode(self) -> StateMode:
        """获取当前状态模式"""
        return self._transition_state.current_mode
    
    def is_transitioning(self) -> bool:
        """检查是否正在过渡"""
        return self._transition_state.is_transitioning
    
    def get_transition_progress(self) -> float:
        """获取过渡进度"""
        return self._transition_state.transition_progress
    
    def get_target_mode(self) -> Optional[StateMode]:
        """获取目标模式（过渡中）"""
        return self._transition_state.target_mode
    
    def get_remaining_steps(self) -> int:
        """获取剩余过渡步数"""
        if not self._transition_state.is_transitioning:
            return 0
        return self._transition_state.total_transition_steps - self._transition_state.transition_steps
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'current_mode': self._transition_state.current_mode.value,
            'target_mode': self._transition_state.target_mode.value if self._transition_state.target_mode else None,
            'is_transitioning': self._transition_state.is_transitioning,
            'transition_progress': self._transition_state.transition_progress,
            'transition_steps_remaining': self.get_remaining_steps(),
            'total_transitions': self._transition_count,
            'total_transition_time': self._total_transition_time,
            'mode_history': [(mode.value, step) for mode, step in self._mode_history[-10:]],
            'step_count': self._step_count,
        }
    
    def reset(self, mode: Optional[StateMode] = None) -> None:
        """
        重置状态控制器
        
        Args:
            mode: 重置后的状态模式（None 保持当前）
        """
        target_mode = mode or self._transition_state.current_mode
        
        self._transition_state = TransitionState(
            current_mode=target_mode,
            current_params=STATE_PARAMS_CONFIG[target_mode].copy(),
            total_transition_steps=self.default_transition_steps
        )
        
        self._transition_count = 0
        self._total_transition_time = 0
        self._mode_history = [(target_mode, 0)]
        self._step_count = 0
        
        logger.info(f"StateController reset to {target_mode.value}")
    
    def __repr__(self) -> str:
        """字符串表示"""
        status = "transitioning" if self._transition_state.is_transitioning else "stable"
        return (
            f"StateController(mode={self._transition_state.current_mode.value}, "
            f"status={status}, "
            f"progress={self._transition_state.transition_progress:.2f})"
        )