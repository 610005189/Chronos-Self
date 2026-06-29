"""
直接调优测试 - 绕过谱归一化限制

直接修改动力学函数参数，验证方案A + 方案B的真实效果。
"""

import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.utils.config import (
    ChronosConfig,
    DimensionalityConfig,
    ChaosInjectionConfig,
    CouplingStabilityConfig,
    NeuralODEConfig,
    NumericsConfig,
)
from chronos_core.core.state import SelfState
from chronos_core.core.integration_engine import IntegrationEngine, IntegrationEngineConfig
from chronos_core.validation.p0_validation import P0Validation, P0ValidationConfig


def create_config(fast_dim, slow_dim):
    config = ChronosConfig()
    
    config.dim = DimensionalityConfig(
        fast_variable_dim=fast_dim,
        slow_variable_dim=slow_dim,
        core_subspace_dim=min(64, fast_dim // 4),
        semantic_dim=min(256, fast_dim // 2),
        physical_dim=min(256, fast_dim // 2),
        fusion_dim=min(512, fast_dim),
        working_memory_dim=min(128, slow_dim * 2),
    )
    
    config.neural_ode = NeuralODEConfig(
        integration_method="euler",
        atol=1e-4,
        rtol=1e-3,
        max_steps=100,
        dt=0.01,
    )
    
    config.numerics = NumericsConfig(
        solver_type="euler",
        spectral_norm_enabled=False,
        attention_mode="linear",
        checkpointing_enabled=False,
        fourier_enabled=False,
        imex_update_interval=100,
        imex_dt_safety_factor=0.9,
    )
    
    config.chaos_injection = ChaosInjectionConfig(
        base_gain=0.01,
        min_gain=0.005,
        chaos_injection_gain=0.01,
        attractor_switch_interval=2000,
        attractor_noise_scale=0.001,
        gain_smoothing=0.95,
    )
    
    config.coupling_stability = CouplingStabilityConfig(
        coupling_adaptation_coeff=0.1,
        elastic_restoration_coeff=0.05,
        l2_perturbation_noise=0.01,
        anti_quietus_weight=0.01,
        inertia_weight=0.01,
        coupling_upper_bound=10.0,
        stability_threshold=1000.0,
        lyapunov_threshold=0.1,
    )
    
    config.device = "cpu"
    config.use_amp = False
    
    config.validation.p0_open_loop_hours = 0.005
    config.validation.lyapunov_window = 200
    config.validation.alignment_num_steps = [10, 30]
    
    return config


def apply_params(engine, decay_rate, max_grad_norm, damping_coeff):
    if hasattr(engine, 'fast_dynamics') and engine.fast_dynamics:
        fast_sys = engine.fast_dynamics
        if hasattr(fast_sys, 'dynamics_fn') and fast_sys.dynamics_fn:
            dyn_fn = fast_sys.dynamics_fn
            
            dyn_fn.config.decay_rate = decay_rate
            dyn_fn.config.max_gradient_norm = max_grad_norm
            dyn_fn.config.damping_coeff = damping_coeff
            
            if hasattr(dyn_fn, 'decay_layer'):
                dyn_fn.decay_layer = nn.utils.remove_spectral_norm(dyn_fn.decay_layer)
                nn.init.constant_(dyn_fn.decay_layer.weight, -decay_rate)
                if dyn_fn.decay_layer.bias is not None:
                    dyn_fn.decay_layer.bias = None
            
            for name, module in dyn_fn.named_modules():
                if isinstance(module, nn.Linear) and hasattr(module, 'weight_orig'):
                    try:
                        module = nn.utils.remove_spectral_norm(module)
                    except:
                        pass


def run_test(fast_dim, slow_dim, decay_rate, max_grad_norm, damping_coeff, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = create_config(fast_dim, slow_dim)
    
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
    
    apply_params(engine, decay_rate, max_grad_norm, damping_coeff)
    
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
    
    initial_state = SelfState(
        E_fast=torch.randn(fast_dim) * 0.01,
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.01,
        timestamp=0.0,
    )
    
    start = time.time()
    try:
        result = validator.run_full_validation(initial_state, verbose=False)
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return None
    elapsed = time.time() - start
    
    return {
        'decay_rate': decay_rate,
        'max_grad_norm': max_grad_norm,
        'damping_coeff': damping_coeff,
        'fast_dim': fast_dim,
        'lyapunov_mean': result.lyapunov_mean,
        'lyapunov_max': result.lyapunov_max,
        'lyapunov_min': result.lyapunov_min,
        'drift_rate': result.drift_rate,
        'alignment_max_error': result.alignment_max_error,
        'passed': result.is_passed,
        'time_elapsed': elapsed,
        'target_reached': 0 < result.lyapunov_mean < 0.1,
    }


def main():
    print(f"\n{'='*70}")
    print("直接调优测试 - 方案A + 方案B")
    print("目标: 将 Lyapunov 指数降到 (0, 0.1)")
    print(f"{'='*70}")
    
    print("\n策略说明:")
    print("1. 关闭谱归一化 (spectral_norm_enabled=False)")
    print("2. 降低混沌注入强度 (base_gain=0.01)")
    print("3. 直接修改 decay_layer 权重")
    print("4. 添加显式阻尼项")
    
    test_cases = [
        {'decay_rate': 0.1, 'max_grad_norm': 10.0, 'damping_coeff': 0.0},
        {'decay_rate': 0.5, 'max_grad_norm': 10.0, 'damping_coeff': 0.0},
        {'decay_rate': 1.0, 'max_grad_norm': 5.0, 'damping_coeff': 0.0},
        {'decay_rate': 2.0, 'max_grad_norm': 5.0, 'damping_coeff': 0.0},
        {'decay_rate': 3.0, 'max_grad_norm': 3.0, 'damping_coeff': 0.0},
        {'decay_rate': 4.0, 'max_grad_norm': 2.0, 'damping_coeff': 0.0},
        {'decay_rate': 2.0, 'max_grad_norm': 5.0, 'damping_coeff': 0.5},
        {'decay_rate': 2.0, 'max_grad_norm': 5.0, 'damping_coeff': 1.0},
        {'decay_rate': 2.0, 'max_grad_norm': 5.0, 'damping_coeff': 2.0},
        {'decay_rate': 3.0, 'max_grad_norm': 3.0, 'damping_coeff': 1.0},
        {'decay_rate': 3.0, 'max_grad_norm': 3.0, 'damping_coeff': 1.5},
        {'decay_rate': 3.0, 'max_grad_norm': 3.0, 'damping_coeff': 2.0},
        {'decay_rate': 4.0, 'max_grad_norm': 2.0, 'damping_coeff': 2.0},
        {'decay_rate': 5.0, 'max_grad_norm': 1.0, 'damping_coeff': 2.0},
        {'decay_rate': 6.0, 'max_grad_norm': 1.0, 'damping_coeff': 3.0},
    ]
    
    results = []
    best_result = None
    best_lyapunov = float('inf')
    
    print(f"\n阶段1: 粗搜 (128D) - 共 {len(test_cases)} 个参数组合")
    print("-" * 70)
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] 测试: decay_rate={params['decay_rate']:.2f}, "
              f"max_grad_norm={params['max_grad_norm']:.2f}, damping={params['damping_coeff']:.2f}")
        
        result = run_test(128, 32, **params)
        
        if result is None:
            continue
        
        print(f"  Lyapunov={result['lyapunov_mean']:.4f}, drift={result['drift_rate']:.4f}, "
              f"time={result['time_elapsed']:.2f}s")
        
        if result['lyapunov_mean'] < best_lyapunov:
            best_lyapunov = result['lyapunov_mean']
            best_result = result
            print(f"  ✨ 新最佳!")
        
        results.append(result)
        
        if result['target_reached']:
            print(f"  🎉 Lyapunov进入目标区间 (0, 0.1)!")
    
    print(f"\n{'='*70}")
    print("阶段1 粗搜完成")
    print(f"最佳 Lyapunov = {best_lyapunov:.4f}")
    if best_result:
        print(f"最佳参数: decay_rate={best_result['decay_rate']:.2f}, "
              f"max_grad_norm={best_result['max_grad_norm']:.2f}, "
              f"damping={best_result['damping_coeff']:.2f}")
    
    if best_lyapunov < 2.0 and best_result:
        print(f"\n{'='*70}")
        print("阶段2: 精搜 (256D)")
        print("-" * 70)
        
        dr = best_result['decay_rate']
        gn = best_result['max_grad_norm']
        dm = best_result['damping_coeff']
        
        fine_cases = [
            {'decay_rate': dr * 0.8, 'max_grad_norm': gn, 'damping_coeff': dm},
            {'decay_rate': dr * 0.9, 'max_grad_norm': gn, 'damping_coeff': dm},
            {'decay_rate': dr * 1.1, 'max_grad_norm': gn, 'damping_coeff': dm},
            {'decay_rate': dr * 1.2, 'max_grad_norm': gn, 'damping_coeff': dm},
            {'decay_rate': dr, 'max_grad_norm': gn * 0.8, 'damping_coeff': dm},
            {'decay_rate': dr, 'max_grad_norm': gn * 0.9, 'damping_coeff': dm},
            {'decay_rate': dr, 'max_grad_norm': gn * 1.1, 'damping_coeff': dm},
            {'decay_rate': dr, 'max_grad_norm': gn, 'damping_coeff': dm * 0.8},
            {'decay_rate': dr, 'max_grad_norm': gn, 'damping_coeff': dm * 0.9},
            {'decay_rate': dr, 'max_grad_norm': gn, 'damping_coeff': dm * 1.1},
            {'decay_rate': dr, 'max_grad_norm': gn, 'damping_coeff': dm * 1.2},
            {'decay_rate': dr * 1.1, 'max_grad_norm': gn * 0.9, 'damping_coeff': dm * 1.1},
        ]
        
        for i, params in enumerate(fine_cases, 1):
            print(f"\n[{i}/{len(fine_cases)}] 精搜测试: decay_rate={params['decay_rate']:.2f}, "
                  f"max_grad_norm={params['max_grad_norm']:.2f}, damping={params['damping_coeff']:.2f}")
            
            result = run_test(256, 64, **params)
            
            if result is None:
                continue
            
            print(f"  Lyapunov={result['lyapunov_mean']:.4f}, time={result['time_elapsed']:.2f}s")
            
            if result['lyapunov_mean'] < best_lyapunov:
                best_lyapunov = result['lyapunov_mean']
                best_result = result
                print(f"  ✨ 新最佳!")
            
            results.append(result)
            
            if result['target_reached']:
                print(f"  🎉 Lyapunov进入目标区间 (0, 0.1)!")
    
    print(f"\n{'='*70}")
    print("测试完成")
    print(f"{'='*70}")
    
    print(f"\n最终最佳结果:")
    if best_result:
        print(f"  decay_rate = {best_result['decay_rate']:.6f}")
        print(f"  max_grad_norm = {best_result['max_grad_norm']:.6f}")
        print(f"  damping_coeff = {best_result['damping_coeff']:.6f}")
        print(f"  Lyapunov λ (mean) = {best_result['lyapunov_mean']:.6f}")
        print(f"  Lyapunov λ (max) = {best_result['lyapunov_max']:.6f}")
        print(f"  Lyapunov λ (min) = {best_result['lyapunov_min']:.6f}")
        print(f"  Drift rate = {best_result['drift_rate']:.6f}")
        print(f"  Alignment max error = {best_result['alignment_max_error']:.6f}")
        print(f"  目标达成 = {'✅ YES' if best_result['target_reached'] else '❌ NO'}")
    else:
        print("  ❌ 所有测试均失败")
    
    print(f"\n{'='*70}")
    print("参数影响分析:")
    print("-" * 70)
    
    if len(results) >= 3:
        decay_results = [r for r in results if r['damping_coeff'] == 0.0]
        if decay_results:
            decay_effect = max(r['lyapunov_mean'] for r in decay_results) - min(r['lyapunov_mean'] for r in decay_results)
            print(f"方案A (decay_rate): 影响范围 = {decay_effect:.4f}")
        
        damping_results = [r for r in results if r['decay_rate'] == 2.0]
        if damping_results:
            damping_effect = max(r['lyapunov_mean'] for r in damping_results) - min(r['lyapunov_mean'] for r in damping_results)
            print(f"方案B (damping_coeff): 影响范围 = {damping_effect:.4f}")
    
    output = {
        'test_cases': len(test_cases),
        'total_results': len(results),
        'best_result': best_result,
        'all_results': results,
        'target_range': '(0, 0.1)',
    }
    
    output_dir = Path('direct_tuning_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == "__main__":
    main()
