"""
真实系统快速调优脚本
======================

使用真实 Chronos 系统，但配置为：
- 低维度（fast_dim=256, slow_dim=64）
- Euler 法（替代 dopri5，每步只需1次函数评估）
- 减少 Lyapunov 计算步数

预计加速：~200-500倍（维度缩小64倍 + 求解器快6倍 + 步数减少）

使用方法：
    python scripts/real_system_fast_tuner.py
    python scripts/real_system_fast_tuner.py --fast-dim 256 --max-iter 15
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

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
class TuningParams:
    """调优参数"""
    base_gain: float = 0.1
    min_gain: float = 0.1
    slow_coupling_limit: float = 0.5
    stability_threshold: float = 1000.0


def create_fast_config(
    fast_dim: int = 256,
    slow_dim: int = 64,
    params: Optional[TuningParams] = None,
) -> ChronosConfig:
    """
    创建快速调优配置
    
    Args:
        fast_dim: 快变量维度
        slow_dim: 慢变量维度
        params: 调优参数
    
    Returns:
        ChronosConfig
    """
    params = params or TuningParams()
    
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
        coupling_adaptation_coeff=0.5,
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
    
    # 7. 验证配置（减少步数）
    # ValidationConfig 是扁平结构，p0 配置在 P0ValidationConfig 中单独管理
    config.validation.p0_open_loop_hours = 0.05  # 3分钟
    config.validation.lyapunov_window = 300
    config.validation.alignment_num_steps = [10, 50]
    
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
        'drift_rate': result.drift_rate,
        'alignment_max_error': result.alignment_max_error,
        'open_loop_stable': result.open_loop_stable,
        'stability_warnings': result.open_loop_stability_warnings,
        'validation_time': elapsed,
    }


def compute_score(result: Dict) -> float:
    """
    计算综合得分
    
    权重：
    - Lyapunov 指数在 (0, 0.1) 区间：0.5
    - 漂移率 < 0.1：0.25
    - 对齐误差 < 0.1：0.25
    """
    lyapunov = result.get('lyapunov_mean', 0)
    
    # Lyapunov 得分
    if 0 < lyapunov < 0.1:
        lyap_score = 1.0
    elif lyapunov <= 0:
        lyap_score = max(0.0, 1.0 + lyapunov * 10)
    else:
        lyap_score = max(0.0, 1.0 - (lyapunov - 0.1) * 5)
    
    # 漂移率得分
    drift = result.get('drift_rate', 1.0)
    drift_score = max(0.0, 1.0 - drift * 5)
    drift_score = min(1.0, drift_score)
    
    # 对齐误差得分
    align_err = result.get('alignment_max_error', 1.0)
    align_score = max(0.0, 1.0 - align_err * 10)
    align_score = min(1.0, align_score)
    
    return 0.5 * lyap_score + 0.25 * drift_score + 0.25 * align_score


def grid_search_tuning(
    fast_dim: int = 256,
    output_dir: str = "real_tuning_results",
) -> Dict:
    """
    网格搜索调优
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 参数搜索空间
    base_gain_values = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3]
    min_gain_frac = [0.2, 0.5, 0.8, 1.0]  # min_gain = base_gain * frac
    coupling_limits = [0.3, 0.5, 0.7, 1.0]
    
    total = len(base_gain_values) * len(min_gain_frac) * len(coupling_limits)
    logger.info(f"开始网格搜索，共 {total} 组参数...")
    logger.info(f"快变量维度: {fast_dim}")
    
    all_results = []
    best_result = None
    best_params = None
    
    start_time = time.time()
    count = 0
    
    for base_gain in base_gain_values:
        for min_frac in min_gain_frac:
            min_gain = base_gain * min_frac
            
            for coupling_limit in coupling_limits:
                count += 1
                
                params = TuningParams(
                    base_gain=base_gain,
                    min_gain=min_gain,
                    slow_coupling_limit=coupling_limit,
                )
                
                config = create_fast_config(fast_dim=fast_dim, params=params)
                
                try:
                    result = run_p0_validation(config, fast_dim=fast_dim)
                    score = compute_score(result)
                    result['score'] = score
                    result['params'] = {
                        'base_gain': base_gain,
                        'min_gain': min_gain,
                        'coupling_limit': coupling_limit,
                    }
                    
                    all_results.append(result)
                    
                    if best_result is None or score > best_result['score']:
                        best_result = result
                        best_params = params
                        
                        logger.info(
                            f"[{count}/{total}] 新最佳! "
                            f"score={score:.4f}, "
                            f"λ={result['lyapunov_mean']:.6f}, "
                            f"drift={result['drift_rate']:.4f}, "
                            f"align={result['alignment_max_error']:.4f}, "
                            f"time={result['validation_time']:.1f}s, "
                            f"params: base_gain={base_gain:.3f}, "
                            f"min_gain={min_gain:.3f}, "
                            f"coupling={coupling_limit:.2f}"
                        )
                    
                except Exception as e:
                    logger.error(f"[{count}/{total}] 验证失败: {e}")
                    result = {
                        'passed': False,
                        'score': 0.0,
                        'lyapunov_mean': -999,
                        'drift_rate': 999,
                        'alignment_max_error': 999,
                        'validation_time': 0,
                        'error': str(e),
                        'params': {
                            'base_gain': base_gain,
                            'min_gain': min_gain,
                            'coupling_limit': coupling_limit,
                        }
                    }
                    all_results.append(result)
                
                if count % 5 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed
                    remaining = (total - count) / speed
                    logger.info(
                        f"进度: {count}/{total} ({100*count/total:.1f}%), "
                        f"速度: {speed:.2f}组/秒(?), 剩余: {remaining/60:.1f}分钟"
                    )
    
    total_time = time.time() - start_time
    
    # 保存结果
    results_data = {
        'fast_dim': fast_dim,
        'solver_type': 'euler',
        'total_time_seconds': total_time,
        'total_combos': total,
        'best_params': {
            'base_gain': best_params.base_gain if best_params else 0,
            'min_gain': best_params.min_gain if best_params else 0,
            'slow_coupling_limit': best_params.slow_coupling_limit if best_params else 0,
        },
        'best_result': best_result,
        'top_10': sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:10],
    }
    
    with open(output_path / 'tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # 生成报告
    report = f"""# 真实系统快速调优报告

## 概述

- **调优方法**: 网格搜索（真实系统，低维+Euler法）
- **快变量维度**: {fast_dim}
- **求解器**: Euler（固定步长）
- **总参数组合数**: {total}
- **总耗时**: {total_time/60:.2f}分钟
- **平均每组耗时**: {total_time/total:.2f}秒

## 最佳参数

| 参数 | 值 |
|------|-----|
| base_gain | {best_params.base_gain if best_params else 'N/A'} |
| min_gain | {best_params.min_gain if best_params else 'N/A'} |
| slow_coupling_limit | {best_params.slow_coupling_limit if best_params else 'N/A'} |

## 最佳指标

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| 综合得分 | {best_result['score']:.4f} | - | - |
| Lyapunov λ | {best_result['lyapunov_mean']:.6f} | (0, 0.1) | {'✓' if 0 < best_result['lyapunov_mean'] < 0.1 else '✗'} |
| 漂移率 | {best_result['drift_rate']:.6f} | < 0.1 | {'✓' if best_result['drift_rate'] < 0.1 else '✗'} |
| 对齐误差 | {best_result['alignment_max_error']:.6f} | < 0.05 | {'✓' if best_result['alignment_max_error'] < 0.05 else '✗'} |
| 验证时间 | {best_result['validation_time']:.1f}秒 | - | - |

## 是否通过 P0 验证

{'**是** ✓' if best_result.get('passed') else '**否** ✗'}

## Top 10 参数组合

| 排名 | base_gain | min_gain | coupling | λ | drift | align | 得分 |
|------|-----------|----------|----------|---|-------|-------|------|
"""
    
    top10 = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:10]
    for i, r in enumerate(top10, 1):
        p = r.get('params', {})
        status = '✓' if r.get('passed') else '✗'
        report += (
            f"| {i} | {p.get('base_gain', 0):.3f} | {p.get('min_gain', 0):.3f} | "
            f"{p.get('coupling_limit', 0):.2f} | {r.get('lyapunov_mean', 0):.6f} | "
            f"{r.get('drift_rate', 0):.4f} | {r.get('alignment_max_error', 0):.4f} | "
            f"{r.get('score', 0):.4f} {status} |\n"
        )
    
    report += f"""
## 速度分析

- 2048维+dopri5: 预计 > 30分钟/组（从之前观测估计）
- {fast_dim}维+Euler: {total_time/total:.1f}秒/组
- 加速比: 约 {(30*60)/(total_time/total):.0f} 倍

## 下一步建议

1. **确认结果有效性**：将最佳参数应用到 2048 维完整系统，验证是否有类似效果
2. **精细调优**：在最佳参数附近进行更精细的搜索（步长更小）
3. **P1/P2 验证**：P0 通过后，用同样的低维模式进行 P1/P2 调优
4. **维度缩放**：验证低维找到的参数在高维系统上是否仍然有效

## 注意事项

1. 低维系统的动力学可能与高维系统有差异（维度灾难/涌现效应）
2. Euler 法的数值精度低于 dopri5，结果可能有偏差
3. 建议先用低维找到参数范围，再用高维+高精度验证
"""
    
    with open(output_path / 'tuning_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"结果已保存到 {output_path}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description='真实系统快速参数调优')
    parser.add_argument(
        '--fast-dim', type=int, default=256,
        help='快变量维度（默认256）'
    )
    parser.add_argument(
        '--slow-dim', type=int, default=64,
        help='慢变量维度（默认64）'
    )
    parser.add_argument(
        '--output', type=str, default='real_tuning_results',
        help='输出目录'
    )
    parser.add_argument(
        '--max-iter', type=int, default=0,
        help='最大迭代次数（0=全部）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("真实系统快速参数调优")
    logger.info("=" * 80)
    logger.info(f"快变量维度: {args.fast_dim}")
    logger.info(f"慢变量维度: {args.slow_dim}")
    logger.info(f"输出目录: {args.output}")
    logger.info("")
    
    results = grid_search_tuning(
        fast_dim=args.fast_dim,
        output_dir=args.output,
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("调优完成！最佳结果：")
    logger.info("=" * 80)
    bp = results['best_params']
    br = results['best_result']
    logger.info(f"  base_gain = {bp['base_gain']}")
    logger.info(f"  min_gain = {bp['min_gain']}")
    logger.info(f"  slow_coupling_limit = {bp['slow_coupling_limit']}")
    logger.info(f"  综合得分 = {br['score']:.4f}")
    logger.info(f"  Lyapunov λ = {br['lyapunov_mean']:.6f}")
    logger.info(f"  漂移率 = {br['drift_rate']:.6f}")
    logger.info(f"  对齐误差 = {br['alignment_max_error']:.6f}")
    logger.info(f"  P0通过: {'是' if br.get('passed') else '否'}")
    logger.info(f"  总耗时: {results['total_time_seconds']/60:.2f}分钟")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
