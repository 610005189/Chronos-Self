"""
精细调优 - 搜索边缘混沌区域
==============================

在初步网格搜索的基础上，精细搜索Lyapunov指数从负变正的分岔点附近，
寻找 λ ∈ (0, 0.1) 的边缘混沌区域。
"""

import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from fast_param_tuner import (
    SimplifiedDynamics,
    TuningParams,
    TuningResult,
    calculate_lyapunov_exponent,
    calculate_drift_rate,
    calculate_autocorrelation,
    evaluate_params
)

logger = logging.getLogger(__name__)


def fine_tune_around_transition(
    fast_dim: int = 64,
    output_dir: str = "fast_tuning_results"
) -> Dict:
    """
    精细调优 - 在分岔点附近搜索
    
    策略：
    1. 先用二分法找到Lyapunov=0的临界增益
    2. 在临界增益附近精细搜索
    3. 同时调整衰减率和非线性强度
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    def test_params(base_gain, min_gain, coupling, decay_rate, noise_scale):
        params = TuningParams(
            base_gain=base_gain,
            min_gain=min_gain,
            coupling_strength=coupling,
            decay_rate=decay_rate,
            noise_scale=noise_scale
        )
        result = evaluate_params(params, fast_dim=fast_dim)
        all_results.append(result)
        return result
    
    # Phase 1: 二分法找临界增益（Lyapunov从负变正的点）
    logger.info("=" * 80)
    logger.info("Phase 1: 二分法搜索临界增益")
    logger.info("=" * 80)
    
    low_gain = 0.01
    high_gain = 1.0
    target_lambda = 0.02  # 目标：小的正Lyapunov
    
    # 先确认范围
    r_low = test_params(low_gain, low_gain, 0.5, 0.1, 0.01)
    r_high = test_params(high_gain, 0.05, 0.5, 0.1, 0.01)
    
    logger.info(f"低增益 ({low_gain}): λ={r_low.lyapunov_lambda:.6f}")
    logger.info(f"高增益 ({high_gain}): λ={r_high.lyapunov_lambda:.6f}")
    
    # 二分搜索
    for iteration in range(15):
        mid_gain = (low_gain + high_gain) / 2
        r_mid = test_params(mid_gain, 0.02, 0.5, 0.1, 0.01)
        
        logger.info(
            f"  迭代 {iteration+1}: gain={mid_gain:.6f}, "
            f"λ={r_mid.lyapunov_lambda:.6f}, "
            f"score={r_mid.overall_score:.4f}"
        )
        
        if r_mid.lyapunov_lambda < target_lambda:
            low_gain = mid_gain
        else:
            high_gain = mid_gain
    
    critical_gain = (low_gain + high_gain) / 2
    logger.info(f"临界增益约为: {critical_gain:.6f}")
    
    # Phase 2: 在临界增益附近精细搜索（调整多个参数）
    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 2: 临界区域精细网格搜索")
    logger.info("=" * 80)
    
    # 围绕临界增益构建搜索空间
    gain_center = critical_gain
    gain_range = np.linspace(gain_center * 0.8, gain_center * 1.5, 12)
    
    decay_rates = [0.02, 0.05, 0.08, 0.1, 0.15]
    noise_scales = [0.005, 0.01, 0.02, 0.05]
    couplings = [0.3, 0.5, 0.7]
    
    total = len(gain_range) * len(decay_rates) * len(noise_scales) * len(couplings)
    logger.info(f"总参数组合数: {total}")
    
    best_result = None
    count = 0
    start_time = time.time()
    
    for base_gain in gain_range:
        min_gain = min(0.02, base_gain * 0.5)
        for decay in decay_rates:
            for noise in noise_scales:
                for coupling in couplings:
                    count += 1
                    
                    params = TuningParams(
                        base_gain=base_gain,
                        min_gain=min_gain,
                        coupling_strength=coupling,
                        decay_rate=decay,
                        noise_scale=noise
                    )
                    result = evaluate_params(params, fast_dim=fast_dim)
                    all_results.append(result)
                    
                    if best_result is None or result.overall_score > best_result.overall_score:
                        best_result = result
                        logger.info(
                            f"[{count}/{total}] 新最佳! "
                            f"score={result.overall_score:.4f}, "
                            f"λ={result.lyapunov_lambda:.6f}, "
                            f"drift={result.drift_rate:.6f}, "
                            f"autocorr={result.autocorrelation:.4f}, "
                            f"gain={base_gain:.4f}, decay={decay:.3f}, "
                            f"noise={noise:.3f}, coupling={coupling:.2f}"
                        )
                    
                    if count % 50 == 0:
                        elapsed = time.time() - start_time
                        speed = count / elapsed
                        remaining = (total - count) / speed
                        logger.info(
                            f"进度: {count}/{total} ({100*count/total:.1f}%), "
                            f"速度: {speed:.2f}组/秒, 剩余: {remaining:.1f}秒"
                        )
    
    total_time = time.time() - start_time
    logger.info(f"精细搜索完成! 总耗时: {total_time:.2f}秒")
    
    # Phase 3: 输出最佳结果
    logger.info("")
    logger.info("=" * 80)
    logger.info("Phase 3: 最佳结果总结")
    logger.info("=" * 80)
    
    if best_result:
        bp = best_result.params
        logger.info(f"最佳参数:")
        logger.info(f"  base_gain = {bp.base_gain:.6f}")
        logger.info(f"  min_gain = {bp.min_gain:.6f}")
        logger.info(f"  coupling_strength = {bp.coupling_strength:.4f}")
        logger.info(f"  decay_rate = {bp.decay_rate:.4f}")
        logger.info(f"  noise_scale = {bp.noise_scale:.4f}")
        logger.info(f"最佳指标:")
        logger.info(f"  Lyapunov λ = {best_result.lyapunov_lambda:.6f}")
        logger.info(f"  漂移率 = {best_result.drift_rate:.6f}")
        logger.info(f"  自相关 = {best_result.autocorrelation:.4f}")
        logger.info(f"  总体得分 = {best_result.overall_score:.4f}")
        logger.info(f"  是否通过: {'是' if best_result.passed else '否'}")
    
    # 保存结果
    results_data = {
        'best_params': {
            'base_gain': best_result.params.base_gain if best_result else 0,
            'min_gain': best_result.params.min_gain if best_result else 0,
            'coupling_strength': best_result.params.coupling_strength if best_result else 0,
            'decay_rate': best_result.params.decay_rate if best_result else 0,
            'noise_scale': best_result.params.noise_scale if best_result else 0,
        },
        'best_metrics': {
            'lyapunov_lambda': best_result.lyapunov_lambda if best_result else 0,
            'drift_rate': best_result.drift_rate if best_result else 0,
            'autocorrelation': best_result.autocorrelation if best_result else 0,
            'overall_score': best_result.overall_score if best_result else 0,
            'passed': best_result.passed if best_result else False
        },
        'critical_gain_estimate': float(critical_gain),
        'fast_dim': fast_dim,
        'total_time_seconds': total_time,
        'total_combos_tested': len(all_results),
        'top_20': [
            {
                'params': {
                    'base_gain': r.params.base_gain,
                    'min_gain': r.params.min_gain,
                    'coupling_strength': r.params.coupling_strength,
                    'decay_rate': r.params.decay_rate,
                    'noise_scale': r.params.noise_scale
                },
                'lyapunov_lambda': r.lyapunov_lambda,
                'drift_rate': r.drift_rate,
                'autocorrelation': r.autocorrelation,
                'overall_score': r.overall_score,
                'passed': r.passed
            }
            for r in sorted(all_results, key=lambda x: -x.overall_score)[:20]
        ]
    }
    
    with open(output_path / 'fine_tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=float)
    
    # 生成报告
    top10 = sorted(all_results, key=lambda x: -x.overall_score)[:10]
    
    report = f"""# 精细调优报告 - 边缘混沌区域搜索

## 概述

- **调优方法**: 二分法 + 精细网格搜索（简化动力学模型）
- **快变量维度**: {fast_dim}
- **总测试组合数**: {len(all_results)}
- **总耗时**: {total_time:.2f}秒
- **临界增益估计**: {critical_gain:.6f}

## 关键发现

系统存在**分岔跃迁**现象：
- 增益 < 临界值: Lyapunov 指数为负（稳定）
- 增益 > 临界值: Lyapunov 指数可能跳变到正值（混沌）
- 边缘混沌区域可能非常窄，需要精确控制增益

## 最佳参数

| 参数 | 值 |
|------|-----|
| base_gain | {best_result.params.base_gain:.6f} |
| min_gain | {best_result.params.min_gain:.6f} |
| coupling_strength | {best_result.params.coupling_strength:.4f} |
| decay_rate | {best_result.params.decay_rate:.4f} |
| noise_scale | {best_result.params.noise_scale:.4f} |

## 最佳指标

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| Lyapunov λ | {best_result.lyapunov_lambda:.6f} | (0, 0.1) | {'✓' if 0 < best_result.lyapunov_lambda < 0.1 else '✗'} |
| 漂移率 | {best_result.drift_rate:.6f} | < 0.05 | {'✓' if best_result.drift_rate < 0.05 else '✗'} |
| 自相关系数 | {best_result.autocorrelation:.4f} | > 0.3 | {'✓' if best_result.autocorrelation > 0.3 else '✗'} |
| 总体得分 | {best_result.overall_score:.4f} | - | - |

## 是否通过验证

{'**是** ✓' if best_result.passed else '**否** ✗'}

## Top 10 参数组合

| 排名 | base_gain | decay | noise | coupling | λ | drift | autocorr | 得分 |
|------|-----------|-------|-------|----------|---|-------|----------|------|
"""
    
    for i, r in enumerate(top10, 1):
        status = '✓' if r.passed else '✗'
        report += (
            f"| {i} | {r.params.base_gain:.4f} | {r.params.decay_rate:.3f} | "
            f"{r.params.noise_scale:.3f} | {r.params.coupling_strength:.2f} | "
            f"{r.lyapunov_lambda:.6f} | {r.drift_rate:.6f} | "
            f"{r.autocorrelation:.4f} | {r.overall_score:.4f} {status} |\n"
        )
    
    report += f"""
## 对完整系统的启示

### 参数调整建议

根据简化模型的调优结果，建议在完整系统中尝试以下调整：

1. **临界增益精确控制**:
   - 临界增益约为 {critical_gain:.4f}（简化模型）
   - 在完整系统中，建议在 base_gain = {critical_gain*0.5:.4f} ~ {critical_gain*2:.4f} 范围搜索

2. **衰减率调整**:
   - 当前 decay_rate = 0.1 可能偏大
   - 建议尝试更小的衰减率（0.02-0.08）

3. **噪声注入**:
   - 适当的噪声（0.01-0.05）可以帮助系统在边缘混沌区域波动
   - 噪声太小系统过于稳定，太大则混沌失控

4. **耦合强度**:
   - 中等耦合强度（0.3-0.7）效果较好
   - 过高的耦合可能导致系统过于稳定

### 下一步

1. 将最佳参数范围映射到完整系统
2. 运行完整系统验证（可以先用低维配置快速验证）
3. 验证 Lyapunov 指数是否进入目标区间
4. 逐步提高维度到 2048
"""
    
    with open(output_path / 'fine_tuning_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"结果已保存到 {output_path}/fine_tuning_results.json")
    logger.info(f"报告已保存到 {output_path}/fine_tuning_report.md")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description='精细调优 - 搜索边缘混沌区域')
    parser.add_argument(
        '--fast-dim', type=int, default=64,
        help='快变量维度（默认64）'
    )
    parser.add_argument(
        '--output', type=str, default='fast_tuning_results',
        help='输出目录（默认fast_tuning_results）'
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
    logger.info("精细调优 - 搜索边缘混沌区域")
    logger.info("=" * 80)
    logger.info(f"快变量维度: {args.fast_dim}")
    logger.info(f"输出目录: {args.output}")
    logger.info("")
    
    results = fine_tune_around_transition(
        fast_dim=args.fast_dim,
        output_dir=args.output
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("调优完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
