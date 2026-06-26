"""
完整反思系统（Reflection System）
===================================

整合实时反思机制和睡眠重放系统，实现完整的自我修正和学习进化流程。

核心功能：
- 整合实时反思和睡眠重放
- 实现反思流程：在线运行 -> 触发睡眠 -> 离线重放 -> 重新运行
- 提供反思接口：手动触发、查询历史、监测效果
- 状态管理：反思状态保存、关键帧管理、反思日志记录

反思流程：
1. 在线运行：实时反思机制
   - 维护最近 T 步计算图
   - 使用截断伴随法进行梯度回传
   - 实时修正近期演化轨迹

2. 达到24小时：触发睡眠期
   - 每24模拟小时自动触发
   - 支持手动触发
   - 记录睡眠历史

3. 离线重放：睡眠重放更新
   - 从关键帧数据库加载关键帧
   - 计算重放一致性损失和预测改善损失
   - 使用伴随法梯度回传更新参数
   - 确保不修改关键帧本身

4. 重新运行：应用更新后的参数
   - 应用睡眠更新后的参数
   - 继续在线运行
   - 监测反思效果

使用示例：
    reflection_system = ReflectionSystem(config=ReflectionSystemConfig())
    reflection_system.initialize(integration_engine)
    
    # 开始在线运行
    reflection_system.start_online_running()
    
    # 添加步骤（实时反思）
    reflection_system.add_online_step(state, inputs)
    
    # 检查是否需要睡眠
    if reflection_system.should_sleep():
        # 执行睡眠重放
        result = reflection_system.perform_sleep()
    
    # 查询反思历史
    history = reflection_system.get_reflection_history()
    
    # 监测反思效果
    performance = reflection_system.monitor_performance()
"""

import torch
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from datetime import datetime
import json
from pathlib import Path

from chronos_core.utils.config import ChronosConfig
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.reflection.realtime_reflection import (
    RealtimeReflection,
    RealtimeReflectionConfig,
    create_realtime_reflection_from_config,
)
from chronos_core.core.reflection.sleep_replay import (
    SleepReplay,
    SleepReplayConfig,
    KeyframeData,
    create_sleep_replay_from_config,
)
from chronos_core.core.reflection.sleep_updater import (
    SleepUpdater,
    SleepUpdaterConfig,
    create_sleep_updater_from_config,
)


logger = logging.getLogger(__name__)


class ReflectionMode(Enum):
    """反思模式枚举"""
    
    ONLINE = "online"  # 在线运行模式（实时反思）
    SLEEP = "sleep"  # 睡眠模式（离线重放）
    OFFLINE = "offline"  # 离线模式（手动反思）
    PAUSED = "paused"  # 暂停模式


class ReflectionState(Enum):
    """反思状态枚举"""
    
    IDLE = "idle"  # 空闲状态
    RUNNING = "running"  # 运行状态
    REFLECTING = "reflecting"  # 反思状态
    SLEEPING = "sleeping"  # 睡眠状态
    UPDATING = "updating"  # 更新状态
    ERROR = "error"  # 错误状态


@dataclass
class ReflectionSystemConfig:
    """完整反思系统配置"""
    
    # 实时反思配置
    enable_realtime_reflection: bool = True  # 启用实时反思
    realtime_reflection_interval: int = 100  # 实时反思间隔
    
    # 睡眠重放配置
    enable_sleep_replay: bool = True  # 启用睡眠重放
    auto_sleep_trigger: bool = True  # 自动睡眠触发
    
    # 状态管理
    save_reflection_history: bool = True  # 保存反思历史
    reflection_history_path: str = "data/reflection_history"  # 反思历史路径
    max_history_entries: int = 10000  # 最大历史条目数
    
    # 监测配置
    enable_performance_monitoring: bool = True  # 启用性能监测
    monitoring_interval: int = 1000  # 监测间隔
    
    # 状态维度
    fast_dim: int = 2048
    slow_dim: int = 512


@dataclass
class ReflectionHistoryEntry:
    """反思历史记录条目"""
    
    # 基本信息
    entry_id: str
    timestamp: float
    created_time: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 反思类型
    reflection_type: str = "realtime"  # 'realtime', 'sleep', 'manual'
    
    # 反思结果
    success: bool = True
    result_summary: Dict[str, Any] = field(default_factory=dict)
    
    # 性能指标
    elapsed_time_ms: float = 0.0
    keyframes_processed: int = 0
    gradients_computed: int = 0
    
    # 状态变化
    state_before: Optional[Dict[str, Any]] = None
    state_after: Optional[Dict[str, Any]] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "created_time": self.created_time,
            "reflection_type": self.reflection_type,
            "success": self.success,
            "result_summary": self.result_summary,
            "elapsed_time_ms": self.elapsed_time_ms,
            "keyframes_processed": self.keyframes_processed,
            "gradients_computed": self.gradients_computed,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "metadata": self.metadata,
        }


class ReflectionSystem:
    """
    完整反思系统
    
    整合实时反思和睡眠重放，实现完整的自我修正和学习进化流程。
    
    主要功能：
    1. 整合实时反思机制和睡眠重放系统
    2. 实现反思流程管理
    3. 提供统一的反思接口
    4. 状态管理和历史记录
    5. 性能监测和效果评估
    
    使用示例：
        system = ReflectionSystem(config=ReflectionSystemConfig())
        system.initialize(integration_engine)
        
        # 在线运行
        system.start_online_running()
        system.add_online_step(state, inputs)
        
        # 睡眠重放
        if system.should_sleep():
            system.perform_sleep()
        
        # 手动反思
        system.manual_reflection()
        
        # 查询历史
        history = system.get_reflection_history()
    """
    
    def __init__(
        self,
        config: Optional[ReflectionSystemConfig] = None,
        global_config: Optional[ChronosConfig] = None,
        integration_engine: Optional[IntegrationEngine] = None,
        device: Optional[str] = None
    ):
        """
        初始化反思系统
        
        Args:
            config: 反思系统配置
            global_config: 全局配置
            integration_engine: 积分引擎
            device: 计算设备
        """
        self.config = config or ReflectionSystemConfig()
        self.global_config = global_config or ChronosConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从全局配置更新
        if global_config:
            self.config.fast_dim = global_config.dim.fast_variable_dim
            self.config.slow_dim = global_config.dim.slow_variable_dim
        
        # 积分引擎
        self.integration_engine = integration_engine
        
        # 核心组件
        self.realtime_reflection: Optional[RealtimeReflection] = None
        self.sleep_replay: Optional[SleepReplay] = None
        self.sleep_updater: Optional[SleepUpdater] = None
        
        # 系统状态
        self._mode: ReflectionMode = ReflectionMode.ONLINE
        self._state: ReflectionState = ReflectionState.IDLE
        self._online_step_count: int = 0
        self._sleep_count: int = 0
        
        # 反思历史
        self._reflection_history: List[ReflectionHistoryEntry] = []
        
        # 性能监测
        self._performance_metrics: Dict[str, Any] = {
            "total_online_steps": 0,
            "total_reflections": 0,
            "total_sleeps": 0,
            "avg_reflection_time_ms": 0.0,
            "avg_sleep_time_ms": 0.0,
            "avg_keyframes_per_sleep": 0.0,
            "performance_improvement": 0.0,
        }
        
        # 统计信息
        self._stats: Dict[str, Any] = {
            "initialized": False,
            "online_time_hours": 0.0,
            "last_reflection_time": 0.0,
            "last_sleep_time": 0.0,
        }
        
        # 初始化标志
        self._initialized = False
        
        logger.info(
            f"ReflectionSystem created: "
            f"realtime={self.config.enable_realtime_reflection}, "
            f"sleep={self.config.enable_sleep_replay}"
        )
    
    def initialize(
        self,
        integration_engine: Optional[IntegrationEngine] = None
    ) -> None:
        """
        初始化反思系统
        
        Args:
            integration_engine: 积分引擎
        """
        if integration_engine is not None:
            self.integration_engine = integration_engine
        
        # 创建实时反思机制
        if self.config.enable_realtime_reflection:
            self.realtime_reflection = create_realtime_reflection_from_config(
                self.global_config,
                self.integration_engine,
                self.device,
            )
        
        # 创建睡眠重放系统
        if self.config.enable_sleep_replay:
            self.sleep_replay = create_sleep_replay_from_config(
                self.global_config,
                self.integration_engine,
                self.device,
            )
        
        # 创建睡眠更新器
        if self.config.enable_sleep_replay:
            self.sleep_updater = create_sleep_updater_from_config(
                self.global_config,
                self.integration_engine,
                self.device,
            )
        
        self._initialized = True
        self._state = ReflectionState.RUNNING
        self._stats["initialized"] = True  # 更新统计信息中的初始化状态
        
        logger.info("ReflectionSystem initialized")
    
    def start_online_running(self) -> None:
        """
        开始在线运行模式
        
        进入在线运行状态，启用实时反思机制。
        """
        if not self._initialized:
            raise ValueError("ReflectionSystem not initialized.")
        
        self._mode = ReflectionMode.ONLINE
        self._state = ReflectionState.RUNNING
        
        logger.info("Online running started")
    
    def add_online_step(
        self,
        state: SelfState,
        inputs: Optional[ExternalInput] = None,
        response_data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加在线运行步骤
        
        在线运行期间添加新的步骤，实时反思机制会自动处理。
        
        Args:
            state: 当前自我状态
            inputs: 外部输入
            response_data: 系统响应数据
            metadata: 元数据
            
        Returns:
            步骤处理结果
        """
        if not self._initialized:
            raise ValueError("ReflectionSystem not initialized.")
        
        if self._mode != ReflectionMode.ONLINE:
            logger.warning("Not in online mode, step added but reflection may not be performed")
        
        # 更新步数计数
        self._online_step_count += 1
        self._performance_metrics["total_online_steps"] += 1
        
        # 更新累积时间（假设 dt=0.01s）
        self._stats["online_time_hours"] += 0.01 / 3600
        
        # 添加到实时反思机制
        result = {"reflection_performed": False}
        
        if self.realtime_reflection is not None:
            # 添加步骤到计算图缓冲区
            self.realtime_reflection.add_step(
                state=state,
                inputs={
                    'X_sem': inputs.X_sem if inputs is not None else None,
                    'X_log': inputs.X_log if inputs is not None else None,
                },
                metadata=metadata,
            )
            
            # 检查是否应该执行反思
            if self.realtime_reflection.should_reflect(self._online_step_count):
                # 执行实时反思
                reflection_result = self.realtime_reflection.reflect(
                    apply_correction=True
                )
                
                result["reflection_performed"] = True
                result["reflection_result"] = reflection_result
                
                # 记录反思历史
                self._record_reflection_history(
                    reflection_type="realtime",
                    result_summary=reflection_result,
                    state=state,
                )
                
                # 更新性能指标
                self._performance_metrics["total_reflections"] += 1
        
        # 添加到睡眠重放系统（记录关键帧）
        if self.sleep_replay is not None and inputs is not None:
            # 判断是否为关键帧
            is_keyframe = inputs.is_high_emotional() or inputs.importance > 0.8
            
            if is_keyframe:
                keyframe_id = self.sleep_replay.add_keyframe(
                    state=state,
                    inputs=inputs,
                    response_data=response_data,
                    metadata=metadata,
                )
                
                result["keyframe_added"] = True
                result["keyframe_id"] = keyframe_id
        
        logger.debug(
            f"Online step added: step_count={self._online_step_count}, "
            f"reflection_performed={result['reflection_performed']}"
        )
        
        return result
    
    def should_sleep(self) -> bool:
        """
        判断是否应该进入睡眠期
        
        检查是否达到睡眠触发条件（24小时间隔）。
        
        Returns:
            是否应该睡眠
        """
        if not self._initialized:
            return False
        
        if not self.config.enable_sleep_replay:
            return False
        
        if self._mode != ReflectionMode.ONLINE:
            return False
        
        # 使用睡眠重放系统的判断
        if self.sleep_replay is not None:
            current_time = time.time()
            should_sleep = self.sleep_replay.should_sleep(current_time)
            
            return should_sleep
        
        return False
    
    def perform_sleep(
        self,
        duration_minutes: Optional[float] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        执行睡眠重放
        
        进入睡眠模式，执行离线重放和参数更新。
        
        Args:
            duration_minutes: 睡眠持续时间
            force: 是否强制触发
            
        Returns:
            睡眠结果字典
        """
        if not self._initialized:
            raise ValueError("ReflectionSystem not initialized.")
        
        if not self.config.enable_sleep_replay:
            logger.warning("Sleep replay not enabled")
            return {"success": False, "reason": "sleep_disabled"}
        
        # 切换到睡眠模式
        self._mode = ReflectionMode.SLEEP
        self._state = ReflectionState.SLEEPING
        
        logger.info("Entering sleep mode")
        
        # 记录睡眠开始
        sleep_start_time = time.time()
        
        # 执行睡眠重放
        sleep_result = {"success": False}
        
        if self.sleep_replay is not None:
            sleep_result = self.sleep_replay.trigger_sleep(
                force=force,
                duration_minutes=duration_minutes,
            )
        
        # 执行睡眠更新（如果重放成功）
        update_result = {"success": False}
        
        if sleep_result.get("success") and self.sleep_updater is not None:
            self._state = ReflectionState.UPDATING
            
            # 获取重放的关键帧
            keyframes = []
            if self.sleep_replay.keyframe_db:
                keyframes = self.sleep_replay.keyframe_db.get_recent_keyframes()
            
            # 执行参数更新
            update_result = self.sleep_updater.perform_sleep_update(keyframes)
        
        # 更新统计
        self._sleep_count += 1
        self._performance_metrics["total_sleeps"] += 1
        
        elapsed_time = (time.time() - sleep_start_time) * 1000
        self._performance_metrics["avg_sleep_time_ms"] = elapsed_time
        
        # 记录睡眠历史
        self._record_reflection_history(
            reflection_type="sleep",
            result_summary={
                "sleep_result": sleep_result,
                "update_result": update_result,
            },
            elapsed_time_ms=elapsed_time,
        )
        
        # 切换回在线模式
        self._mode = ReflectionMode.ONLINE
        self._state = ReflectionState.RUNNING
        
        logger.info(
            f"Sleep completed: sleep_count={self._sleep_count}, "
            f"elapsed={elapsed_time:.2f}ms"
        )
        
        return {
            "success": sleep_result.get("success") and update_result.get("success"),
            "sleep_result": sleep_result,
            "update_result": update_result,
            "elapsed_time_ms": elapsed_time,
        }
    
    def manual_reflection(
        self,
        reflection_type: str = "realtime",
        window_size: Optional[int] = None,
        apply_correction: bool = True
    ) -> Dict[str, Any]:
        """
        手动触发反思
        
        支持手动触发实时反思或睡眠重放。
        
        Args:
            reflection_type: 反思类型（'realtime', 'sleep'）
            window_size: 反思窗口大小
            apply_correction: 是否应用修正
            
        Returns:
            反思结果字典
        """
        if not self._initialized:
            raise ValueError("ReflectionSystem not initialized.")
        
        logger.info(f"Manual reflection triggered: type={reflection_type}")
        
        # 切换到反思状态
        self._state = ReflectionState.REFLECTING
        
        reflection_start_time = time.time()
        reflection_result = {"success": False}
        
        if reflection_type == "realtime":
            # 手动实时反思
            if self.realtime_reflection is not None:
                reflection_result = self.realtime_reflection.reflect(
                    window_size=window_size,
                    apply_correction=apply_correction,
                )
        
        elif reflection_type == "sleep":
            # 手动睡眠重放
            reflection_result = self.perform_sleep(force=True)
        
        else:
            logger.warning(f"Unknown reflection type: {reflection_type}")
        
        elapsed_time = (time.time() - reflection_start_time) * 1000
        
        # 记录反思历史
        self._record_reflection_history(
            reflection_type="manual",
            result_summary=reflection_result,
            elapsed_time_ms=elapsed_time,
        )
        
        # 恢复运行状态
        self._state = ReflectionState.RUNNING
        
        return reflection_result
    
    def get_reflection_history(
        self,
        limit: Optional[int] = None,
        reflection_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取反思历史
        
        Args:
            limit: 数量限制
            reflection_type: 反思类型过滤
            
        Returns:
            反思历史列表
        """
        history = self._reflection_history
        
        # 类型过滤
        if reflection_type:
            history = [h for h in history if h.reflection_type == reflection_type]
        
        # 数量限制
        if limit:
            history = history[-limit:]
        
        return [h.to_dict() for h in history]
    
    def monitor_performance(self) -> Dict[str, Any]:
        """
        监测反思性能
        
        评估反思机制的效果和性能。
        
        Returns:
            性能监测结果
        """
        performance = self._performance_metrics.copy()
        
        # 收集组件统计
        performance["realtime_reflection_stats"] = {}
        if self.realtime_reflection:
            performance["realtime_reflection_stats"] = self.realtime_reflection.get_statistics()
        
        performance["sleep_replay_stats"] = {}
        if self.sleep_replay:
            performance["sleep_replay_stats"] = self.sleep_replay.get_statistics()
        
        performance["sleep_updater_stats"] = {}
        if self.sleep_updater:
            performance["sleep_updater_stats"] = self.sleep_updater.get_statistics()
        
        # 计算改进指标
        if self._sleep_count > 0:
            # 简化：使用平均损失作为改进指标
            avg_update_loss = performance.get("sleep_updater_stats", {}).get("avg_update_loss", 0.0)
            performance["performance_improvement"] = max(0, 1.0 - avg_update_loss)
        
        return performance
    
    def query_keyframes(
        self,
        query_type: str = "recent",
        n: Optional[int] = 100,
        **kwargs
    ) -> List[KeyframeData]:
        """
        查询关键帧
        
        Args:
            query_type: 查询类型（'recent', 'high_emotional', 'time_range'）
            n: 数量限制
            **kwargs: 其他查询参数
            
        Returns:
            关键帧列表
        """
        if self.sleep_replay is None or self.sleep_replay.keyframe_db is None:
            return []
        
        keyframe_db = self.sleep_replay.keyframe_db
        
        if query_type == "recent":
            return keyframe_db.get_recent_keyframes(n=n)
        
        elif query_type == "high_emotional":
            threshold = kwargs.get("threshold", 0.7)
            return keyframe_db.query_high_emotional(threshold=threshold, limit=n)
        
        elif query_type == "time_range":
            start_time = kwargs.get("start_time", 0.0)
            end_time = kwargs.get("end_time", time.time())
            return keyframe_db.query_by_time_range(start_time, end_time)
        
        else:
            logger.warning(f"Unknown query type: {query_type}")
            return []
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前反思状态
        
        Returns:
            当前状态字典
        """
        return {
            "mode": self._mode.value,
            "state": self._state.value,
            "online_step_count": self._online_step_count,
            "sleep_count": self._sleep_count,
            "online_time_hours": self._stats["online_time_hours"],
            "initialized": self._initialized,
        }
    
    def save_reflection_state(self, filepath: Optional[str] = None) -> None:
        """
        保存反思状态
        
        Args:
            filepath: 文件路径
        """
        filepath = filepath or f"{self.config.reflection_history_path}/reflection_state.json"
        
        # 确保目录存在
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # 构建状态数据
        state_data = {
            "current_state": self.get_current_state(),
            "performance_metrics": self._performance_metrics,
            "reflection_history_count": len(self._reflection_history),
            "stats": self._stats,
            "timestamp": datetime.now().isoformat(),
        }
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reflection state saved to {filepath}")
    
    def load_reflection_state(self, filepath: Optional[str] = None) -> None:
        """
        加载反思状态
        
        Args:
            filepath: 文件路径
        """
        filepath = filepath or f"{self.config.reflection_history_path}/reflection_state.json"
        
        if not Path(filepath).exists():
            logger.warning(f"Reflection state file not found: {filepath}")
            return
        
        # 读取文件
        with open(filepath, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        # 恢复状态
        self._performance_metrics = state_data.get("performance_metrics", {})
        self._stats = state_data.get("stats", {})
        
        logger.info(f"Reflection state loaded from {filepath}")
    
    def _record_reflection_history(
        self,
        reflection_type: str,
        result_summary: Dict[str, Any],
        state: Optional[SelfState] = None,
        elapsed_time_ms: Optional[float] = None,
    ) -> None:
        """
        记录反思历史
        
        Args:
            reflection_type: 反思类型
            result_summary: 结果摘要
            state: 状态快照
            elapsed_time_ms: 耗时
        """
        if not self.config.save_reflection_history:
            return
        
        # 创建历史条目
        entry = ReflectionHistoryEntry(
            entry_id=f"ref_{len(self._reflection_history)}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=time.time(),
            reflection_type=reflection_type,
            success=result_summary.get("success", False),
            result_summary=result_summary,
            elapsed_time_ms=elapsed_time_ms or 0.0,
            keyframes_processed=result_summary.get("keyframes_count", 0),
        )
        
        # 记录状态快照
        if state is not None:
            entry.state_before = {
                "E_fast_norm": state.get_fast_norm(),
                "E_slow_norm": state.get_slow_norm(),
                "timestamp": state.timestamp,
            }
        
        # 添加到历史
        self._reflection_history.append(entry)
        
        # 限制历史长度
        if len(self._reflection_history) > self.config.max_history_entries:
            self._reflection_history = self._reflection_history[-self.config.max_history_entries:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self._stats.copy()
        stats["performance_metrics"] = self._performance_metrics
        stats["reflection_history_count"] = len(self._reflection_history)
        stats["current_state"] = self.get_current_state()
        
        return stats
    
    def reset(self) -> None:
        """
        重置反思系统
        """
        # 重置组件
        if self.realtime_reflection:
            self.realtime_reflection.reset()
        
        if self.sleep_replay:
            self.sleep_replay.reset()
        
        if self.sleep_updater:
            self.sleep_updater.reset()
        
        # 重置状态
        self._mode = ReflectionMode.ONLINE
        self._state = ReflectionState.IDLE
        self._online_step_count = 0
        self._sleep_count = 0
        
        # 清空历史
        self._reflection_history.clear()
        
        # 重置性能指标
        self._performance_metrics = {
            "total_online_steps": 0,
            "total_reflections": 0,
            "total_sleeps": 0,
            "avg_reflection_time_ms": 0.0,
            "avg_sleep_time_ms": 0.0,
            "avg_keyframes_per_sleep": 0.0,
            "performance_improvement": 0.0,
        }
        
        # 重置统计
        self._stats = {
            "initialized": self._initialized,
            "online_time_hours": 0.0,
            "last_reflection_time": 0.0,
            "last_sleep_time": 0.0,
        }
        
        logger.info("ReflectionSystem reset")
    
    def __repr__(self) -> str:
        status = "initialized" if self._initialized else "not_initialized"
        return (
            f"ReflectionSystem(status={status}, "
            f"mode={self._mode.value}, "
            f"state={self._state.value}, "
            f"steps={self._online_step_count})"
        )


def create_reflection_system_from_config(
    config: ChronosConfig,
    integration_engine: Optional[IntegrationEngine] = None,
    device: Optional[str] = None
) -> ReflectionSystem:
    """
    从全局配置创建反思系统
    
    Args:
        config: 全局配置
        integration_engine: 积分引擎
        device: 计算设备
        
    Returns:
        ReflectionSystem 实例
    """
    reflection_config = ReflectionSystemConfig(
        enable_realtime_reflection=True,
        enable_sleep_replay=True,
        auto_sleep_trigger=config.memory_temporal.sleep_replay_interval_hours > 0,
        realtime_reflection_interval=config.memory_temporal.reflection_window // 10,
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
    )
    
    system = ReflectionSystem(
        config=reflection_config,
        global_config=config,
        integration_engine=integration_engine,
        device=device,
    )
    
    system.initialize()
    return system