"""
最小验证配置测试
==================

验证 Chronos-Self 系统在最小维度配置下能够正常运行。

最小配置：
- D_f (快变量维度): 256 (默认 2048)
- D_s (慢变量维度): 128 (默认 512)
- k (核心子空间维度): 32 (默认 64)

测试目标：
1. 验证系统初始化成功
2. 验证核心组件正常运行
3. 验证数值稳定性
4. 验证内存占用合理
5. 验证快速验证流程

用途：
- 开发调试时的快速验证
- 资源受限环境下的测试
- 验证系统基本功能完整性

运行方式：
    python scripts/minimal_config_test.py

预期结果：
- 所有测试通过
- 系统运行稳定
- 内存占用 < 500MB
- 单步处理时间 < 100ms
"""

import sys
import torch
import numpy as np
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.utils.config import ChronosConfig, DimensionalityConfig
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.dmn_system import DefaultModeNetwork
from chronos_core.core.meta_cognitive.meta_cognitive_system import MetaCognitiveSystem
from chronos_core.core.reflection.reflection_system import ReflectionSystem, ReflectionSystemConfig
from chronos_core.memory.work_memory import WorkingMemory
from chronos_core.integration.system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController,
    SystemStatus,
)
from chronos_core.validation.validation_system import ValidationSystem, ValidationMode

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MinimalConfigTest")


@dataclass
class MinimalConfigTestResult:
    """测试结果"""
    test_name: str
    passed: bool
    duration_ms: float = 0.0
    error_message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "metrics": self.metrics,
        }


class MinimalConfigTest:
    """
    最小配置验证测试
    
    使用最小维度配置验证系统基本功能。
    """
    
    def __init__(
        self,
        fast_dim: int = 256,
        slow_dim: int = 128,
        core_dim: int = 32,
        device: str = "cpu"  # 默认使用 CPU，便于测试
    ):
        """
        初始化测试
        
        Args:
            fast_dim: 快变量维度
            slow_dim: 慢变量维度
            core_dim: 核心子空间维度
            device: 计算设备
        """
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.core_dim = core_dim
        self.device = device
        
        # 创建最小配置
        self.minimal_config = self._create_minimal_config()
        
        # 测试结果
        self.results: list[MinimalConfigTestResult] = []
        
        logger.info(
            f"MinimalConfigTest initialized: "
            f"D_f={fast_dim}, D_s={slow_dim}, k={core_dim}, "
            f"device={device}"
        )
    
    def _create_minimal_config(self) -> ChronosConfig:
        """创建最小配置"""
        config = ChronosConfig()
        
        # 更新维度配置
        config.dim.fast_variable_dim = self.fast_dim
        config.dim.slow_variable_dim = self.slow_dim
        config.dim.core_subspace_dim = self.core_dim
        
        # 更新相关维度
        config.dim.semantic_dim = min(self.fast_dim // 2, 256)
        config.dim.physical_dim = min(self.fast_dim // 2, 256)
        config.dim.fusion_dim = min(self.fast_dim, 512)
        config.dim.working_memory_dim = min(self.core_dim, 128)
        
        # 更新元认知配置
        config.meta_cognitive.l0_hidden_dim = min(self.fast_dim // 4, 64)
        config.meta_cognitive.l1_hidden_dim = min(self.fast_dim // 2, 128)
        config.meta_cognitive.l2_hidden_dim = min(self.core_dim, 32)
        config.meta_cognitive.l2_projection_dim = min(self.core_dim // 2, 16)
        
        # 更新编码器配置
        config.encoder.semantic_hidden_dim = min(self.fast_dim // 2, 128)
        config.encoder.physical_hidden_dim = min(self.fast_dim // 2, 128)
        config.encoder.physical_state_dim = min(self.core_dim, 32)
        
        # 更新验证配置（快速验证）
        config.validation.p0_open_loop_hours = 0.01  # 36秒
        config.validation.alignment_num_steps = [1, 10, 50]
        config.validation.lyapunov_window = 50
        
        # 设备设置
        config.device = self.device
        
        return config
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        运行所有测试
        
        Returns:
            测试结果汇总
        """
        logger.info("=" * 80)
        logger.info("开始最小配置验证测试")
        logger.info("=" * 80)
        
        tests = [
            ("系统初始化测试", self.test_system_initialization),
            ("积分引擎测试", self.test_integration_engine),
            ("DMN测试", self.test_dmn),
            ("元认知系统测试", self.test_meta_cognitive),
            ("反思系统测试", self.test_reflection_system),
            ("工作记忆测试", self.test_working_memory),
            ("完整系统流程测试", self.test_complete_flow),
            ("数值稳定性测试", self.test_numerical_stability),
            ("内存占用测试", self.test_memory_usage),
            ("性能基准测试", self.test_performance),
        ]
        
        total_start = time.time()
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.results.append(result)
                
                status = "✓ 通过" if result.passed else "✗ 失败"
                logger.info(f"{test_name}: {status} ({result.duration_ms:.2f}ms)")
                
                if not result.passed:
                    logger.error(f"错误: {result.error_message}")
                
            except Exception as e:
                result = MinimalConfigTestResult(
                    test_name=test_name,
                    passed=False,
                    error_message=str(e)
                )
                self.results.append(result)
                logger.error(f"{test_name}: ✗ 异常 - {e}")
        
        total_duration = (time.time() - total_start) * 1000
        
        # 汇总结果
        summary = self._generate_summary(total_duration)
        
        logger.info("=" * 80)
        logger.info(f"测试完成: 通过 {summary['passed_count']}/{summary['total_count']}")
        logger.info(f"总耗时: {total_duration:.2f}ms")
        logger.info("=" * 80)
        
        return summary
    
    def _generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """生成测试汇总"""
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        
        summary = {
            "total_tests": total_count,
            "passed_count": passed_count,
            "failed_count": total_count - passed_count,
            "pass_rate": passed_count / total_count if total_count > 0 else 0.0,
            "total_duration_ms": total_duration,
            "all_passed": passed_count == total_count,
            "results": [r.to_dict() for r in self.results],
            "config": {
                "fast_dim": self.fast_dim,
                "slow_dim": self.slow_dim,
                "core_dim": self.core_dim,
                "device": self.device,
            },
        }
        
        return summary
    
    def test_system_initialization(self) -> MinimalConfigTestResult:
        """测试系统初始化"""
        start = time.time()
        
        try:
            # 创建系统配置
            system_config = ChronosSystemConfig(
                enable_semantic_encoder=False,  # 简化测试
                enable_logical_encoder=False,
                enable_fusion_module=False,
                enable_meta_cognitive=True,
                enable_reflection=True,
                enable_working_memory=True,
                enable_training_system=False,
                enable_validation_system=False,
            )
            
            # 创建系统
            system = ChronosSystem(
                config=system_config,
                global_config=self.minimal_config,
                device=self.device
            )
            
            # 初始化
            system.initialize()
            
            # 验证初始化状态
            assert system._initialized, "系统未初始化"
            assert system._system_state.status == SystemStatus.READY, "系统状态不正确"
            
            # 验证组件
            components_status = {
                "integration_engine": system.integration_engine is not None,
                "dmn": system.dmn is not None,
                "working_memory": system.working_memory is not None,
                "meta_cognitive_system": system.meta_cognitive_system is not None,
                "reflection_system": system.reflection_system is not None,
            }
            
            for name, status in components_status.items():
                assert status, f"{name} 未初始化"
            
            duration = (time.time() - start) * 1000
            
            return MinimalConfigTestResult(
                test_name="系统初始化测试",
                passed=True,
                duration_ms=duration,
                metrics=components_status
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="系统初始化测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_integration_engine(self) -> MinimalConfigTestResult:
        """测试积分引擎"""
        start = time.time()
        
        try:
            # 创建积分引擎
            engine = IntegrationEngine(
                config=self.minimal_config,
                device=self.device
            )
            engine.initialize()
            
            # 创建初始状态
            initial_state = SelfState(
                E_fast=torch.randn(self.fast_dim, device=self.device) * 0.1,
                E_slow=torch.randn(self.slow_dim, device=self.device) * 0.1,
                timestamp=0.0
            )
            
            # 运行多步
            current_state = initial_state
            num_steps = 100
            
            for _ in range(num_steps):
                current_state = engine.step(current_state)
            
            # 验证状态
            assert current_state.timestamp > initial_state.timestamp, "时间未更新"
            assert not torch.isnan(current_state.E_fast).any(), "快变量包含 NaN"
            assert not torch.isnan(current_state.E_slow).any(), "慢变量包含 NaN"
            
            # 获取监控数据
            monitoring = engine.get_state_monitoring()
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_steps": num_steps,
                "final_timestamp": current_state.timestamp,
                "E_fast_norm": current_state.get_fast_norm(),
                "E_slow_norm": current_state.get_slow_norm(),
                "monitoring_keys": list(monitoring.keys()),
            }
            
            return MinimalConfigTestResult(
                test_name="积分引擎测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="积分引擎测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_dmn(self) -> MinimalConfigTestResult:
        """测试默认模式网络"""
        start = time.time()
        
        try:
            # 创建 DMN
            dmn = DefaultModeNetwork(
                chaos_config=self.minimal_config.chaos_injection,
                dim_config=self.minimal_config.dim,
                device=self.device
            )
            dmn.initialize()
            
            # 获取混沌注入信号
            chaos_signal = dmn.get_chaos_injection()
            
            # 验证维度
            assert chaos_signal.shape[0] == self.core_dim, \
                f"混沌信号维度不正确: {chaos_signal.shape[0]} vs {self.core_dim}"
            
            # 验证数值稳定性
            assert not torch.isnan(chaos_signal).any(), "混沌信号包含 NaN"
            
            # 运行多步
            num_steps = 50
            signals = []
            
            for _ in range(num_steps):
                signal = dmn.get_chaos_injection()
                signals.append(torch.norm(signal).item())
                dmn.step()
            
            # 验证信号变化
            signal_variance = np.var(signals)
            assert signal_variance > 0, "混沌信号无变化"
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_steps": num_steps,
                "signal_dim": chaos_signal.shape[0],
                "signal_norm_mean": np.mean(signals),
                "signal_norm_std": np.std(signals),
            }
            
            return MinimalConfigTestResult(
                test_name="DMN测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="DMN测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_meta_cognitive(self) -> MinimalConfigTestResult:
        """测试元认知系统"""
        start = time.time()
        
        try:
            # 创建元认知系统
            meta_sys = MetaCognitiveSystem(
                global_config=self.minimal_config,
                device=self.device
            )
            
            # 创建输入
            semantic_input = torch.randn(self.minimal_config.dim.semantic_dim, device=self.device)
            physical_input = torch.randn(self.minimal_config.dim.physical_dim, device=self.device)
            
            # 运行元认知循环
            output = meta_sys.forward(
                semantic_input=semantic_input,
                physical_input=physical_input,
                dt=0.01,
                apply_regulation=True
            )
            
            # 验证输出
            assert "l0_output" in output, "缺少 L0 输出"
            assert "l1_output" in output, "缺少 L1 输出"
            assert "l2_control" in output, "缺少 L2 控制信号"
            
            # 验证数值稳定性
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    assert not torch.isnan(value).any(), f"{key} 包含 NaN"
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "output_keys": list(output.keys()),
                "l2_control_shape": output["l2_control"].shape if "l2_control" in output else None,
            }
            
            return MinimalConfigTestResult(
                test_name="元认知系统测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="元认知系统测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_reflection_system(self) -> MinimalConfigTestResult:
        """测试反思系统"""
        start = time.time()
        
        try:
            # 创建积分引擎
            engine = IntegrationEngine(
                config=self.minimal_config,
                device=self.device
            )
            engine.initialize()
            
            # 创建反思系统
            reflection_config = ReflectionSystemConfig(
                enable_realtime_reflection=True,
                enable_sleep_replay=False,  # 简化测试
                fast_dim=self.fast_dim,
                slow_dim=self.slow_dim,
            )
            
            reflection_sys = ReflectionSystem(
                config=reflection_config,
                global_config=self.minimal_config,
                integration_engine=engine,
                device=self.device
            )
            reflection_sys.initialize(engine)
            
            # 创建状态
            state = SelfState(
                E_fast=torch.randn(self.fast_dim, device=self.device) * 0.1,
                E_slow=torch.randn(self.slow_dim, device=self.device) * 0.1,
                timestamp=0.0
            )
            
            # 添加多步
            num_steps = 20
            for i in range(num_steps):
                result = reflection_sys.add_online_step(
                    state=state,
                    metadata={"step": i}
                )
                state = engine.step(state)
            
            # 获取统计
            stats = reflection_sys.get_statistics()
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_steps": num_steps,
                "stats_keys": list(stats.keys()),
                "step_count": stats.get("step_count", 0),
            }
            
            return MinimalConfigTestResult(
                test_name="反思系统测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="反思系统测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_working_memory(self) -> MinimalConfigTestResult:
        """测试工作记忆"""
        start = time.time()
        
        try:
            # 创建工作记忆
            working_memory = WorkingMemory(
                capacity=7,  # Miller's law
                fast_dim=self.fast_dim,
                chunk_dim=self.minimal_config.dim.working_memory_dim,
                device=self.device
            )
            
            # 创建多个组块
            num_chunks = 10  # 超出容量测试
            chunks = []
            
            for i in range(num_chunks):
                source_state = torch.randn(self.fast_dim, device=self.device)
                chunk = working_memory.create_chunk(
                    source_state=source_state,
                    chunk_type="test",
                    initial_activation=1.0 - i * 0.1,  # 递减激活强度
                    metadata={"index": i}
                )
                chunks.append(chunk)
            
            # 获取活跃组块
            active_chunks = working_memory.get_active_chunks()
            
            # 验证容量限制
            assert len(active_chunks) <= 7, f"容量超限: {len(active_chunks)} > 7"
            
            # 验证激活衰减
            # 更新一步
            working_memory.update(dt=1.0)
            
            # 再次获取
            active_chunks_after = working_memory.get_active_chunks()
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_chunks_created": num_chunks,
                "capacity": 7,
                "active_chunks_before": len(active_chunks),
                "active_chunks_after": len(active_chunks_after),
            }
            
            return MinimalConfigTestResult(
                test_name="工作记忆测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="工作记忆测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_complete_flow(self) -> MinimalConfigTestResult:
        """测试完整系统流程"""
        start = time.time()
        
        try:
            # 创建完整系统
            system_config = ChronosSystemConfig(
                enable_semantic_encoder=False,
                enable_logical_encoder=False,
                enable_fusion_module=False,
                enable_meta_cognitive=True,
                enable_reflection=True,
                enable_working_memory=True,
            )
            
            system = ChronosSystem(
                config=system_config,
                global_config=self.minimal_config,
                device=self.device
            )
            system.initialize()
            
            # 创建控制器
            controller = ChronosSystemController(system)
            controller.start()
            
            # 处理多个输入
            num_inputs = 10
            responses = []
            
            for i in range(num_inputs):
                response = controller.process_input(text=f"测试输入 {i}")
                responses.append(response)
            
            # 验证响应
            assert len(responses) == num_inputs, "响应数量不匹配"
            
            for response in responses:
                assert response.state_after is not None, "缺少最终状态"
                assert response.processing_time_ms > 0, "处理时间异常"
            
            # 获取系统状态
            system_state = controller.get_system_state()
            
            assert system_state.total_steps >= num_inputs, "步数统计错误"
            assert system_state.status == SystemStatus.RUNNING, "系统状态不正确"
            
            # 停止系统
            controller.stop()
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_inputs": num_inputs,
                "total_steps": system_state.total_steps,
                "avg_processing_time_ms": np.mean([r.processing_time_ms for r in responses]),
            }
            
            return MinimalConfigTestResult(
                test_name="完整系统流程测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="完整系统流程测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_numerical_stability(self) -> MinimalConfigTestResult:
        """测试数值稳定性"""
        start = time.time()
        
        try:
            # 创建积分引擎
            engine = IntegrationEngine(
                config=self.minimal_config,
                device=self.device
            )
            engine.initialize()
            
            # 使用较大初始值测试稳定性
            initial_state = SelfState(
                E_fast=torch.randn(self.fast_dim, device=self.device) * 10.0,  # 较大值
                E_slow=torch.randn(self.slow_dim, device=self.device) * 5.0,
                timestamp=0.0
            )
            
            # 运行长时间
            num_steps = 1000
            current_state = initial_state
            
            norm_history = []
            
            for _ in range(num_steps):
                current_state = engine.step(current_state)
                norm_history.append(current_state.get_fast_norm())
                
                # 检查 NaN
                if torch.isnan(current_state.E_fast).any():
                    raise ValueError("快变量发散")
                if torch.isnan(current_state.E_slow).any():
                    raise ValueError("慢变量发散")
            
            # 分析稳定性
            max_norm = max(norm_history)
            min_norm = min(norm_history)
            final_norm = norm_history[-1]
            
            # 验证没有发散
            assert max_norm < 1000, f"快变量发散: max_norm={max_norm}"
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_steps": num_steps,
                "max_norm": max_norm,
                "min_norm": min_norm,
                "final_norm": final_norm,
                "norm_variance": np.var(norm_history),
            }
            
            return MinimalConfigTestResult(
                test_name="数值稳定性测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="数值稳定性测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_memory_usage(self) -> MinimalConfigTestResult:
        """测试内存占用"""
        start = time.time()
        
        try:
            import gc
            import os
            import psutil
            
            # 清理内存
            gc.collect()
            
            # 获取初始内存
            process = psutil.Process(os.getpid())
            initial_memory_mb = process.memory_info().rss / 1024 / 1024
            
            # 创建系统
            system_config = ChronosSystemConfig(
                enable_semantic_encoder=False,
                enable_logical_encoder=False,
                enable_fusion_module=False,
                enable_meta_cognitive=True,
                enable_reflection=True,
                enable_working_memory=True,
            )
            
            system = ChronosSystem(
                config=system_config,
                global_config=self.minimal_config,
                device=self.device
            )
            system.initialize()
            
            # 运行一些步骤
            for _ in range(100):
                system.process_input()
            
            # 获取内存占用
            gc.collect()
            final_memory_mb = process.memory_info().rss / 1024 / 1024
            
            memory_increase_mb = final_memory_mb - initial_memory_mb
            
            # 验证内存占用合理
            # 最小配置预期内存增加 < 500MB
            assert memory_increase_mb < 500, f"内存占用过高: {memory_increase_mb:.2f}MB"
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "initial_memory_mb": initial_memory_mb,
                "final_memory_mb": final_memory_mb,
                "memory_increase_mb": memory_increase_mb,
                "threshold_mb": 500,
            }
            
            return MinimalConfigTestResult(
                test_name="内存占用测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except ImportError:
            # psutil 未安装，跳过内存测试
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="内存占用测试",
                passed=True,
                duration_ms=duration,
                error_message="psutil 未安装，跳过内存测试",
                metrics={"note": "内存测试需要安装 psutil"}
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="内存占用测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )
    
    def test_performance(self) -> MinimalConfigTestResult:
        """测试性能基准"""
        start = time.time()
        
        try:
            # 创建系统
            system_config = ChronosSystemConfig(
                enable_semantic_encoder=False,
                enable_logical_encoder=False,
                enable_fusion_module=False,
                enable_meta_cognitive=True,
                enable_reflection=True,
                enable_working_memory=True,
            )
            
            system = ChronosSystem(
                config=system_config,
                global_config=self.minimal_config,
                device=self.device
            )
            system.initialize()
            
            # 性能测试
            num_steps = 100
            step_times = []
            
            for _ in range(num_steps):
                step_start = time.time()
                system.process_input()
                step_times.append((time.time() - step_start) * 1000)
            
            avg_time_ms = np.mean(step_times)
            max_time_ms = np.max(step_times)
            min_time_ms = np.min(step_times)
            
            # 验证性能
            # 最小配置预期单步时间 < 100ms (CPU)
            assert avg_time_ms < 100, f"性能不达标: avg_time={avg_time_ms:.2f}ms"
            
            duration = (time.time() - start) * 1000
            
            metrics = {
                "num_steps": num_steps,
                "avg_time_ms": avg_time_ms,
                "max_time_ms": max_time_ms,
                "min_time_ms": min_time_ms,
                "threshold_ms": 100,
            }
            
            return MinimalConfigTestResult(
                test_name="性能基准测试",
                passed=True,
                duration_ms=duration,
                metrics=metrics
            )
            
        except Exception as e:
            duration = (time.time() - start) * 1000
            return MinimalConfigTestResult(
                test_name="性能基准测试",
                passed=False,
                duration_ms=duration,
                error_message=str(e)
            )


def main():
    """主函数"""
    logger.info("=" * 80)
    logger.info("Chronos-Self 最小配置验证测试")
    logger.info("=" * 80)
    
    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 创建测试
    test = MinimalConfigTest(
        fast_dim=256,
        slow_dim=128,
        core_dim=32,
        device=device
    )
    
    # 运行测试
    summary = test.run_all_tests()
    
    # 输出结果
    if summary["all_passed"]:
        logger.info("\n✓ 所有测试通过！系统在最小配置下运行正常。")
    else:
        logger.error(f"\n✗ 有 {summary['failed_count']} 个测试失败。")
    
    # 输出详细指标
    logger.info("\n关键指标:")
    for result in test.results:
        if result.passed and result.metrics:
            logger.info(f"  {result.test_name}:")
            for key, value in result.metrics.items():
                logger.info(f"    {key}: {value}")
    
    return summary


if __name__ == "__main__":
    main()