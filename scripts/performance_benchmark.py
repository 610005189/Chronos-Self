"""
性能基准测试脚本
================

单GPU验证性能测试：
- 测试GPU利用率
- 测试内存使用
- 测试计算速度
- 测试训练效率

性能指标：
- 积分速度（步/秒）
- 训练速度（epoch/小时）
- 内存占用（MB）
- GPU利用率（%）
- 验证时间（分钟）

使用方式：
    python scripts/performance_benchmark.py
"""

import torch
import numpy as np
import time
import sys
import json
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.utils.config import ChronosConfig, DimensionalityConfig
from chronos_core.integration.system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController,
)
from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput


@dataclass
class PerformanceMetrics:
    """性能指标数据"""
    
    # 积分性能
    integration_steps_per_second: float = 0.0
    avg_step_time_ms: float = 0.0
    
    # 训练性能
    epochs_per_hour: float = 0.0
    avg_epoch_time_s: float = 0.0
    
    # 内存性能
    memory_usage_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    
    # GPU性能
    gpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    
    # 验证性能
    validation_time_minutes: float = 0.0
    
    # 稳定性
    numerical_stability_score: float = 1.0
    stability_issues: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "integration_steps_per_second": self.integration_steps_per_second,
            "avg_step_time_ms": self.avg_step_time_ms,
            "epochs_per_hour": self.epochs_per_hour,
            "avg_epoch_time_s": self.avg_epoch_time_s,
            "memory_usage_mb": self.memory_usage_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_utilization": self.gpu_utilization,
            "gpu_temperature": self.gpu_temperature,
            "validation_time_minutes": self.validation_time_minutes,
            "numerical_stability_score": self.numerical_stability_score,
            "stability_issues": self.stability_issues,
        }


class PerformanceBenchmark:
    """
    性能基准测试
    
    测试系统性能和资源使用情况
    """
    
    def __init__(
        self,
        global_config: Optional[ChronosConfig] = None,
        system_config: Optional[ChronosSystemConfig] = None,
        device: Optional[str] = None
    ):
        """
        初始化基准测试
        
        Args:
            global_config: 全局配置
            system_config: 系统配置
            device: 计算设备
        """
        self.global_config = global_config or ChronosConfig()
        self.system_config = system_config or ChronosSystemConfig()
        self.device = device or self.global_config.device
        
        # 性能指标
        self.metrics = PerformanceMetrics()
        
        # 测试参数
        self.test_steps = 1000  # 测试步数
        self.test_epochs = 10   # 测试epoch数
        
        print(f"性能基准测试初始化: device={self.device}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        获取系统信息
        
        Returns:
            系统信息字典
        """
        info = {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "memory_available_gb": psutil.virtual_memory().available / (1024**3),
        }
        
        # GPU信息
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    info["gpu_name"] = gpu.name
                    info["gpu_memory_total_mb"] = gpu.memoryTotal
                    info["gpu_memory_free_mb"] = gpu.memoryFree
                    info["gpu_driver"] = gpu.driver
            except:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        
        return info
    
    def test_integration_speed(self) -> Dict[str, Any]:
        """
        测试积分速度
        
        Returns:
            积分速度测试结果
        """
        print("\n=== 测试积分速度 ===")
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 准备测试
        step_times = []
        stability_issues = 0
        
        # 运行测试
        start_time = time.time()
        
        for step_idx in range(self.test_steps):
            step_start = time.time()
            
            # 创建随机输入
            external_input = ExternalInput(
                X_sem=torch.randn(self.global_config.dim.semantic_dim) * 0.1,
                X_log=torch.randn(self.global_config.dim.physical_dim) * 0.1
            )
            
            # 处理输入
            response = system.process_input(external_input=external_input)
            
            step_time = (time.time() - step_start) * 1000
            step_times.append(step_time)
            
            # 检查稳定性
            if response.state_after.get_fast_norm() > 1000:
                stability_issues += 1
            
            # 进度输出
            if step_idx % 100 == 0:
                print(f"  Step {step_idx}/{self.test_steps}: avg_time={np.mean(step_times[-100:]):.2f}ms")
        
        elapsed_time = time.time() - start_time
        
        # 计算指标
        avg_step_time = np.mean(step_times)
        steps_per_second = self.test_steps / elapsed_time
        
        # 更新指标
        self.metrics.integration_steps_per_second = steps_per_second
        self.metrics.avg_step_time_ms = avg_step_time
        self.metrics.stability_issues = stability_issues
        self.metrics.numerical_stability_score = max(0, 1 - stability_issues / self.test_steps)
        
        # 清理
        system.shutdown()
        
        result = {
            "total_steps": self.test_steps,
            "elapsed_time": elapsed_time,
            "steps_per_second": steps_per_second,
            "avg_step_time_ms": avg_step_time,
            "stability_issues": stability_issues,
        }
        
        print(f"\n积分速度测试结果:")
        print(f"  总步数: {self.test_steps}")
        print(f"  总耗时: {elapsed_time:.2f}秒")
        print(f"  积分速度: {steps_per_second:.2f} 步/秒")
        print(f"  平均步耗时: {avg_step_time:.2f}ms")
        print(f"  稳定性问题: {stability_issues}")
        
        return result
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """
        测试内存使用
        
        Returns:
            内存使用测试结果
        """
        print("\n=== 测试内存使用 ===")
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        
        # 初始化前的内存
        process = psutil.Process()
        memory_before = process.memory_info().rss / (1024**2)
        
        # 初始化系统
        system.initialize()
        
        # 初始化后的内存
        memory_after_init = process.memory_info().rss / (1024**2)
        
        # 运行一些步骤
        for _ in range(100):
            external_input = ExternalInput(
                X_sem=torch.randn(self.global_config.dim.semantic_dim),
                X_log=torch.randn(self.global_config.dim.physical_dim)
            )
            system.process_input(external_input=external_input)
        
        # 运行后的内存
        memory_after_run = process.memory_info().rss / (1024**2)
        
        # GPU内存
        gpu_memory = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)
        
        # 更新指标
        self.metrics.memory_usage_mb = memory_after_run
        self.metrics.gpu_memory_mb = gpu_memory
        
        # 清理
        system.shutdown()
        
        result = {
            "memory_before_mb": memory_before,
            "memory_after_init_mb": memory_after_init,
            "memory_after_run_mb": memory_after_run,
            "memory_increase_mb": memory_after_run - memory_before,
            "gpu_memory_mb": gpu_memory,
        }
        
        print(f"\n内存使用测试结果:")
        print(f"  初始化前内存: {memory_before:.2f}MB")
        print(f"  初始化后内存: {memory_after_init:.2f}MB")
        print(f"  运行后内存: {memory_after_run:.2f}MB")
        print(f"  内存增量: {memory_after_run - memory_before:.2f}MB")
        print(f"  GPU内存: {gpu_memory:.2f}MB")
        
        return result
    
    def test_gpu_utilization(self) -> Dict[str, Any]:
        """
        测试GPU利用率
        
        Returns:
            GPU利用率测试结果
        """
        print("\n=== 测试GPU利用率 ===")
        
        if self.device != "cuda" or not torch.cuda.is_available():
            print("  GPU不可用，跳过测试")
            return {"gpu_available": False}
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # GPU利用率记录
        gpu_utilizations = []
        gpu_temperatures = []
        
        # 运行测试
        for step_idx in range(200):
            external_input = ExternalInput(
                X_sem=torch.randn(self.global_config.dim.semantic_dim),
                X_log=torch.randn(self.global_config.dim.physical_dim)
            )
            system.process_input(external_input=external_input)
            
            # 记录GPU状态
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_utilizations.append(gpu.load * 100)
                    gpu_temperatures.append(gpu.temperature)
            except:
                pass
            
            if step_idx % 50 == 0:
                print(f"  Step {step_idx}: GPU utilization sampling...")
        
        # 计算指标
        avg_utilization = np.mean(gpu_utilizations) if gpu_utilizations else 0
        avg_temperature = np.mean(gpu_temperatures) if gpu_temperatures else 0
        
        # 更新指标
        self.metrics.gpu_utilization = avg_utilization
        self.metrics.gpu_temperature = avg_temperature
        
        # 清理
        system.shutdown()
        
        result = {
            "avg_gpu_utilization": avg_utilization,
            "avg_gpu_temperature": avg_temperature,
            "max_gpu_utilization": np.max(gpu_utilizations) if gpu_utilizations else 0,
            "gpu_available": True,
        }
        
        print(f"\nGPU利用率测试结果:")
        print(f"  平均GPU利用率: {avg_utilization:.2f}%")
        print(f"  最大GPU利用率: {np.max(gpu_utilizations) if gpu_utilizations else 0:.2f}%")
        print(f"  平均GPU温度: {avg_temperature:.2f}°C")
        
        return result
    
    def test_validation_time(self) -> Dict[str, Any]:
        """
        测试验证时间
        
        Returns:
            验证时间测试结果
        """
        print("\n=== 测试验证时间 ===")
        
        # 使用简化的验证测试
        # 实际验证时间需要运行完整的验证流程
        
        start_time = time.time()
        
        # 创建系统
        system = ChronosSystem(
            config=self.system_config,
            global_config=self.global_config
        )
        system.initialize()
        
        # 模拟快速验证（1000步）
        validation_steps = 1000
        
        for step_idx in range(validation_steps):
            external_input = ExternalInput(
                X_sem=torch.randn(self.global_config.dim.semantic_dim) * 0.1,
                X_log=torch.randn(self.global_config.dim.physical_dim) * 0.1
            )
            system.process_input(external_input=external_input)
            
            if step_idx % 200 == 0:
                print(f"  Validation step {step_idx}/{validation_steps}")
        
        elapsed_time = time.time() - start_time
        
        # 更新指标
        self.metrics.validation_time_minutes = elapsed_time / 60
        
        # 清理
        system.shutdown()
        
        # 估算完整验证时间（72小时模拟）
        estimated_full_validation = (elapsed_time / validation_steps) * (72 * 3600 / 0.01) / 3600
        
        result = {
            "quick_validation_time_seconds": elapsed_time,
            "validation_steps": validation_steps,
            "estimated_full_validation_hours": estimated_full_validation,
        }
        
        print(f"\n验证时间测试结果:")
        print(f"  快速验证耗时: {elapsed_time:.2f}秒")
        print(f"  快速验证步数: {validation_steps}")
        print(f"  估算完整验证时间: {estimated_full_validation:.2f}小时")
        
        return result
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        运行完整基准测试
        
        Returns:
            完整基准测试结果
        """
        print("\n" + "=" * 80)
        print("Chronos-Self 性能基准测试")
        print("=" * 80)
        
        # 系统信息
        system_info = self.get_system_info()
        print(f"\n系统信息:")
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        # 运行各项测试
        results = {
            "system_info": system_info,
            "integration_speed": self.test_integration_speed(),
            "memory_usage": self.test_memory_usage(),
            "gpu_utilization": self.test_gpu_utilization(),
            "validation_time": self.test_validation_time(),
            "metrics": self.metrics.to_dict(),
        }
        
        # 总结
        print("\n" + "=" * 80)
        print("性能基准测试总结")
        print("=" * 80)
        
        print(f"\n关键性能指标:")
        print(f"  积分速度: {self.metrics.integration_steps_per_second:.2f} 步/秒")
        print(f"  平均步耗时: {self.metrics.avg_step_time_ms:.2f}ms")
        print(f"  内存占用: {self.metrics.memory_usage_mb:.2f}MB")
        print(f"  GPU利用率: {self.metrics.gpu_utilization:.2f}%")
        print(f"  数值稳定性: {self.metrics.numerical_stability_score:.4f}")
        
        # 保存结果
        output_path = Path("results") / "performance_benchmark.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存至: {output_path}")
        
        return results
    
    def compare_with_targets(self) -> Dict[str, Any]:
        """
        与目标性能对比
        
        Returns:
            对比结果
        """
        # 目标性能指标
        targets = {
            "integration_steps_per_second": 100,  # 目标100步/秒
            "avg_step_time_ms": 10,  # 目标10ms
            "memory_usage_mb": 2000,  # 目标2000MB
            "gpu_utilization": 50,  # 目标50%利用率
            "numerical_stability_score": 0.99,  # 目标99%稳定性
        }
        
        comparison = {}
        
        for metric_name, target_value in targets.items():
            actual_value = getattr(self.metrics, metric_name, 0)
            
            if target_value > 0:
                ratio = actual_value / target_value
                status = "PASS" if metric_name == "avg_step_time_ms" and actual_value <= target_value else \
                         "PASS" if metric_name != "avg_step_time_ms" and actual_value >= target_value else "FAIL"
            else:
                ratio = 0
                status = "N/A"
            
            comparison[metric_name] = {
                "target": target_value,
                "actual": actual_value,
                "ratio": ratio,
                "status": status,
            }
        
        print(f"\n性能对比结果:")
        for metric_name, comp in comparison.items():
            print(f"  {metric_name}: target={comp['target']}, actual={comp['actual']:.2f}, status={comp['status']}")
        
        return comparison


def run_benchmark_with_minimal_config():
    """
    使用最小配置运行基准测试
    
    测试配置：D_f=256, D_s=128, k=32
    """
    print("\n最小配置基准测试")
    
    # 创建最小配置
    dim_config = DimensionalityConfig()
    dim_config.fast_variable_dim = 256
    dim_config.slow_variable_dim = 128
    dim_config.core_subspace_dim = 32
    dim_config.semantic_dim = 256
    dim_config.physical_dim = 256
    
    global_config = ChronosConfig(dim=dim_config)
    global_config.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    system_config = ChronosSystemConfig()
    system_config.device = global_config.device
    system_config.enable_semantic_encoder = False
    system_config.enable_logical_encoder = False
    system_config.enable_fusion_module = False
    
    # 运行基准测试
    benchmark = PerformanceBenchmark(
        global_config=global_config,
        system_config=system_config,
        device=global_config.device
    )
    
    results = benchmark.run_full_benchmark()
    
    # 性能对比
    comparison = benchmark.compare_with_targets()
    
    return results, comparison


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("Chronos-Self 性能基准测试")
    print("=" * 80)
    
    # 运行最小配置基准测试
    results, comparison = run_benchmark_with_minimal_config()
    
    print("\n基准测试完成！")
    
    return results


if __name__ == "__main__":
    main()