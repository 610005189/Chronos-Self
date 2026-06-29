"""
单参数验证脚本
==============

使用真实 Chronos 系统（低维快速模式）验证指定参数的 P0 验证结果。

使用方法：
    python scripts/validate_single_params.py
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict
from dataclasses import dataclass

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    ChaosInjectionConfig,
    CouplingStabilityConfig,
    NeuralODEConfig,
    NumericsConfig,
    ValidationConfig,
)
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine, IntegrationEngineConfig
from chronos_core.validation.p0_validation import P0Validation, P0ValidationConfig

logger = logging.getLogger(__name__)


@dataclass
class ValidationParams:
    """验证参数"""
    base_gain: float = 0.4
    min_gain: float = 0.08
    slow_coupling_limit: float = 0.5
    stability_threshold: float = 1000.0


def create_validation_config(
    fast_dim: int = 256,
    slow_dim: int = 64,
    params: ValidationParams = None,
    lyapunov_window: int = 300,
) -> ChronosConfig:
    """
    创建验证配置
    
    Args:
        fast_dim: 快变量维度
        slow_dim: 慢变量维度
        params: 验证参数
        lyapunov_window: Lyapunov 计算窗口
    
    Returns:
        ChronosConfig
    """
    params = params or ValidationParams()
    
    config = ChronosConfig()
    
    # 1. 降低维度
    config.dim = DimensionalityConfig(
        fast_variable_dim=fast_dim,
        slow_variable_dim=slow_dim,
        core_subspace_dim=min(64, fast_dim // 4),
        semantic_dim=min(256, fast_dim // 2),
        physical_dim=min(256, fast_dim // 2),
        fusion_dim=min(512, fast_dim),
        working_memory_dim=min(128, slow_dim * 2),
    )
    
    # 2. 改用 Euler 法（固定步长，1次函数评估/步）
    config.neural_ode = NeuralODEConfig(
        integration_method="rk4",
        atol=1e-4,
        rtol=1e-3,
        max_steps=100,
        dt=0.01,
    )
    
    # 3. 数值求解器配置
    config.numerics = NumericsConfig(
        solver_type="euler",
        spectral_norm_enabled=True,
        attention_mode="linear",
        checkpointing_enabled=False,
        fourier_enabled=False,
        imex_update_interval=100,
        imex_dt_safety_factor=0.9,
    )
    
    # 4. 混沌注入参数
    config.chaos_injection = ChaosInjectionConfig(
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        chaos_injection_gain=params.base_gain,
        attractor_switch_interval=2000,
        attractor_noise_scale=0.01,
        gain_smoothing=0.95,
    )
    
    # 5. 耦合与稳定性参数
    config.coupling_stability = CouplingStabilityConfig(
        coupling_adaptation_coeff=params.slow_coupling_limit,
        elastic_restoration_coeff=0.05,
        l2_perturbation_noise=0.05,
        anti_quietus_weight=0.1,
        inertia_weight=0.05,
        coupling_upper_bound=10.0,
        stability_threshold=params.stability_threshold,
        lyapunov_threshold=0.1,
    )
    
    # 6. 设备
    config.device = "cpu"
    config.use_amp = False
    
    # 7. 验证配置（大幅减少步数以加快验证）
    config.validation.p0_open_loop_hours = 0.005  # 18秒模拟时间
    config.validation.lyapunov_window = lyapunov_window
    config.validation.alignment_num_steps = [10, 30]
    
    return config


def run_p0_validation(
    config: ChronosConfig,
    fast_dim: int,
    seed: int = 42,
) -> Dict:
    """
    运行 P0 级验证
    
    Returns:
        验证结果字典
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建引擎
    engine_config = IntegrationEngineConfig(
        fast_dim=config.dim.fast_variable_dim,
        slow_dim=config.dim.slow_variable_dim,
        slow_update_frequency=config.memory_temporal.slow_update_frequency,
        solver_method=config.neural_ode.integration_method,
        solver_atol=config.neural_ode.atol,
        solver_rtol=config.neural_ode.rtol,
        solver_type=config.numerics.solver_type,
        imex_update_interval=config.numerics.imex_update_interval,
    )
    
    engine = IntegrationEngine(
        config=config,
        engine_config=engine_config,
        device=config.device,
        seed=seed,
    )
    engine.initialize()
    
    # 创建验证器
    p0_config = P0ValidationConfig(
        open_loop_hours=config.validation.p0_open_loop_hours,
        open_loop_dt=0.01,
        stability_check_interval=500,
        max_stability_warnings=20,
        drift_calculation_window=100,
        max_baseline_drift_rate=config.validation.p0_max_baseline_drift,
        drift_monitoring_interval=200,
        lyapunov_calculation_steps=config.validation.lyapunov_window,
        lyapunov_min_threshold=0.0,
        lyapunov_max_threshold=0.1,
        perturbation_magnitude=1e-6,
        lyapunov_recalculation_interval=2000,
        alignment_test_steps=config.validation.alignment_num_steps,
        alignment_max_error_threshold=config.validation.alignment_max_error,
        alignment_num_tests=3,
    )
    
    validator = P0Validation(
        engine=engine,
        config=config,
        p0_config=p0_config,
        device=config.device,
    )
    
    # 初始化状态
    initial_state = SelfState(
        E_fast=torch.randn(fast_dim) * 0.1,
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
        timestamp=0.0,
    )
    
    # 运行验证
    start = time.time()
    result = validator.run_full_validation(initial_state, verbose=False)
    elapsed = time.time() - start
    
    return {
        'passed': result.is_passed,
        'score': result.overall_score,
        'lyapunov_mean': result.lyapunov_mean,
        'lyapunov_max': result.lyapunov_max,
        'lyapunov_min': result.lyapunov_min,
        'lyapunov_std': result.lyapunov_std,
        'drift_rate': result.drift_rate,
        'drift_passed': result.drift_passed,
        'alignment_max_error': result.alignment_max_error,
        'alignment_avg_error': result.alignment_avg_error,
        'alignment_passed': result.alignment_passed,
        'open_loop_stable': result.open_loop_stable,
        'open_loop_passed': result.open_loop_passed,
        'stability_warnings': result.open_loop_stability_warnings,
        'validation_time': elapsed,
    }


def main():
    import logging
    logging.basicConfig(level=logging.WARNING)  # 只显示警告和错误
    
    print("=" * 80, flush=True)
    print("真实 Chronos 系统 P0 验证", flush=True)
    print("=" * 80, flush=True)
    
    # 待验证参数（纯数学模型找到的最佳参数）
    params = ValidationParams(
        base_gain=0.4,
        min_gain=0.08,
        slow_coupling_limit=0.5,
    )
    
    fast_dim = 256
    slow_dim = 64
    seed = 42
    solver = "euler"
    lyapunov_window = 300
    
    print(f"参数来源: 纯数学模型最佳参数", flush=True)
    print(f"base_gain = {params.base_gain}", flush=True)
    print(f"min_gain = {params.min_gain}", flush=True)
    print(f"slow_coupling_limit = {params.slow_coupling_limit}", flush=True)
    print(f"fast_dim = {fast_dim}", flush=True)
    print(f"slow_dim = {slow_dim}", flush=True)
    print(f"solver = {solver}", flush=True)
    print(f"seed = {seed}", flush=True)
    print(f"lyapunov_window = {lyapunov_window}", flush=True)
    print("", flush=True)
    
    # 创建配置
    print("创建验证配置...", flush=True)
    config = create_validation_config(
        fast_dim=fast_dim,
        slow_dim=slow_dim,
        params=params,
        lyapunov_window=lyapunov_window,
    )
    print("配置创建完成", flush=True)
    
    print("开始运行 P0 验证...", flush=True)
    
    try:
        result = run_p0_validation(config, fast_dim=fast_dim, seed=seed)
        
        print("", flush=True)
        print("=" * 80, flush=True)
        print("验证结果", flush=True)
        print("=" * 80, flush=True)
        print(f"P0 验证通过: {'是 ✓' if result['passed'] else '否 ✗'}", flush=True)
        print(f"综合得分: {result['score']:.4f}", flush=True)
        print("", flush=True)
        print(f"Lyapunov 指数（均值）: {result['lyapunov_mean']:.6f}", flush=True)
        print(f"Lyapunov 指数（最大）: {result['lyapunov_max']:.6f}", flush=True)
        print(f"Lyapunov 指数（最小）: {result['lyapunov_min']:.6f}", flush=True)
        print(f"Lyapunov 目标区间: (0, 0.1)", flush=True)
        print("", flush=True)
        print(f"漂移率: {result['drift_rate']:.6f}", flush=True)
        print(f"漂移率达标: {'是 ✓' if result['drift_passed'] else '否 ✗'}", flush=True)
        print("", flush=True)
        print(f"对齐误差（最大）: {result['alignment_max_error']:.6f}", flush=True)
        print(f"对齐误差（平均）: {result['alignment_avg_error']:.6f}", flush=True)
        print(f"对齐验证通过: {'是 ✓' if result['alignment_passed'] else '否 ✗'}", flush=True)
        print("", flush=True)
        print(f"开环稳定: {'是 ✓' if result['open_loop_stable'] else '否 ✗'}", flush=True)
        print(f"稳定性警告次数: {result['stability_warnings']}", flush=True)
        print("", flush=True)
        print(f"运行时间: {result['validation_time']:.2f} 秒", flush=True)
        print("=" * 80, flush=True)
        
        # 保存结果
        output_dir = Path("real_system_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_data = {
            "params": {
                "base_gain": params.base_gain,
                "min_gain": params.min_gain,
                "slow_coupling_limit": params.slow_coupling_limit,
                "fast_dim": fast_dim,
                "slow_dim": slow_dim,
                "solver": solver,
                "seed": seed,
                "lyapunov_window": lyapunov_window,
            },
            "validation_result": result,
            "summary": {
                "p0_passed": result['passed'],
                "overall_score": result['score'],
                "lyapunov_mean": result['lyapunov_mean'],
                "drift_rate": result['drift_rate'],
                "alignment_max_error": result['alignment_max_error'],
                "open_loop_stable": result['open_loop_stable'],
                "validation_time_seconds": result['validation_time'],
            }
        }
        
        output_file = output_dir / "validation_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, default=str)
        
        print(f"结果已保存到: {output_file}", flush=True)
        
    except Exception as e:
        print(f"验证失败: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # 保存错误信息
        output_dir = Path("real_system_validation")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        error_data = {
            "params": {
                "base_gain": params.base_gain,
                "min_gain": params.min_gain,
                "slow_coupling_limit": params.slow_coupling_limit,
                "fast_dim": fast_dim,
                "slow_dim": slow_dim,
                "solver": solver,
                "seed": seed,
            },
            "error": str(e),
            "passed": False,
        }
        
        output_file = output_dir / "validation_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, indent=2)


if __name__ == '__main__':
    main()
