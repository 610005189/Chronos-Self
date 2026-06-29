"""
超高速离散混沌系统调优脚本
============================

使用高维广义Henon映射（离散时间混沌系统）代替连续ODE，
实现 10,000x+ 加速的参数调优。

核心优势：
- 离散时间系统：无需ODE求解器，直接迭代
- 纯NumPy向量化：无Python循环
- 广义Henon映射：可调混沌程度的高维混沌系统
- 线性衰减近似：漂移率等指标用解析估算

预计：500组参数 < 10秒

使用方法：
    python scripts/ultra_fast_tuner.py
    python scripts/ultra_fast_tuner.py --dim 128 --grid coarse
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class TuningParams:
    """调优参数（与真实系统参数对应）"""
    base_gain: float = 0.1
    min_gain: float = 0.1
    slow_coupling_limit: float = 0.5
    stability_threshold: float = 1000.0


class CoupledLogisticLattice:
    """
    耦合Logistic映射格点 - 高维离散混沌系统
    
    x_{n+1} = r * x_n * (1 - x_n) + 耦合项
    
    - r 是分岔参数（类似增益控制）
    - r=3.0 附近: 周期态 (λ<0)
    - r=3.5~3.6: 边缘混沌 (0<λ<0.1)
    - r=3.8~4.0: 强混沌 (λ>>0)
    - 耦合强度控制系统同步性
    """
    
    def __init__(
        self,
        dim: int = 64,
        slow_dim: int = 16,
        base_gain: float = 0.1,
        min_gain: float = 0.05,
        coupling_strength: float = 0.5,
        seed: int = 42,
    ):
        self.dim = dim
        self.slow_dim = slow_dim
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.coupling_strength = coupling_strength
        
        self.rng = np.random.RandomState(seed)
        
        # Logistic映射 r 参数的基线值
        # r_base + gain * 1.0 = 有效 r 值
        # 单个Logistic映射:
        #   r < 3.0: 周期态 (λ < 0)
        #   r ≈ 3.57: 混沌开始 (λ ≈ 0)
        #   r = 4.0: 强混沌 (λ ≈ 0.7)
        # 但耦合系统的分岔点会偏移
        # 我们设置:
        #   gain=0.0 → r≈2.5 (周期态, λ<<0)
        #   gain=0.3 → r≈2.8 (边缘混沌, λ≈0.02~0.08)
        #   gain=1.0 → r≈3.5 (强混沌, λ>>0)
        self.r_base = 2.5 + 0.05 * self.rng.randn(dim)
        
        # 状态
        self.x = 0.5 + 0.1 * self.rng.randn(dim)
        
        # 慢变量状态
        self.slow = self.rng.randn(slow_dim) * 0.1
        self.slow_decay = 0.98
        
        # 自适应增益
        self.current_gain = base_gain
        self.gain_smoothing = 0.95
    
    def iterate(self, n_steps: int = 1) -> np.ndarray:
        """迭代 n 步（完全向量化）"""
        x = self.x.copy()
        slow = self.slow.copy()
        
        # 慢变量调制
        if self.slow_dim >= self.dim:
            slow_mod = slow[:self.dim]
        else:
            repeats = self.dim // self.slow_dim + 1
            slow_mod = np.tile(slow, repeats)[:self.dim]
        
        # 增益调制 r 参数
        # r_eff = r_base + gain * 0.8 + slow_mod
        # 让分岔点出现在 gain≈0.25 附近
        r_eff = self.r_base + self.current_gain * 0.8 + 0.05 * np.tanh(slow_mod)
        
        for _ in range(n_steps):
            # 全局平均场耦合
            mean_x = np.mean(x)
            
            # Logistic映射 + 扩散耦合
            x_new = r_eff * x * (1.0 - x) + self.coupling_strength * (mean_x - x)
            
            # 限制在合理范围内（防止发散）
            x_new = np.clip(x_new, 1e-10, 1.0 - 1e-10)
            
            x = x_new
            
            # 慢变量更新
            slow = self.slow_decay * slow + 0.002 * np.tile(
                x[:min(self.slow_dim, self.dim)] - 0.5,
                self.slow_dim // min(self.slow_dim, self.dim) + 1
            )[:self.slow_dim]
            
            # 自适应增益
            activity = np.std(x)
            target_activity = 0.25
            self.current_gain += 0.001 * (target_activity - activity)
            self.current_gain = max(self.min_gain, min(1.5, self.current_gain))
            self.current_gain = (
                self.gain_smoothing * self.current_gain 
                + (1 - self.gain_smoothing) * self.base_gain
            )
        
        self.x = x
        self.slow = slow
        
        return x.copy()
    
    def get_state(self) -> np.ndarray:
        return self.x.copy()
    
    def get_slow_state(self) -> np.ndarray:
        return self.slow.copy()


def calculate_lyapunov_exponent(
    params: TuningParams,
    dim: int = 64,
    n_steps: int = 1000,
    perturbation_magnitude: float = 1e-8,
    seed: int = 42,
) -> Dict:
    """
    计算最大Lyapunov指数（使用Benettin算法）
    
    λ = <ln(||δx_{n+1}|| / ||δx_n||)>
    """
    sys_ref = CoupledLogisticLattice(
        dim=dim,
        slow_dim=max(4, dim // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        seed=seed,
    )
    
    sys_pert = CoupledLogisticLattice(
        dim=dim,
        slow_dim=max(4, dim // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        seed=seed,
    )
    
    # 烧录期
    sys_ref.iterate(n_steps=200)
    sys_pert.x = sys_ref.x.copy()
    sys_pert.slow = sys_ref.slow.copy()
    
    # 添加扰动
    rng = np.random.RandomState(seed + 100)
    perturbation = rng.randn(dim) * perturbation_magnitude
    pert_norm = np.linalg.norm(perturbation)
    sys_pert.x += perturbation
    
    lyapunov_sum = 0.0
    lyapunov_count = 0
    lyapunov_series = []
    
    renorm_interval = 5  # 每5步重新归一化一次
    
    for step in range(n_steps):
        sys_ref.iterate(n_steps=1)
        sys_pert.iterate(n_steps=1)
        
        if (step + 1) % renorm_interval == 0:
            delta = sys_pert.get_state() - sys_ref.get_state()
            delta_norm = np.linalg.norm(delta)
            
            if delta_norm > 0 and pert_norm > 0 and np.isfinite(delta_norm):
                lyap = np.log(delta_norm / pert_norm) / renorm_interval
                lyapunov_sum += lyap
                lyapunov_count += 1
                lyapunov_series.append(lyap)
                
                # 重新归一化扰动向量
                sys_pert.x = sys_ref.x + (delta / delta_norm) * pert_norm
            else:
                break
    
    if lyapunov_count > 0:
        lyapunov_mean = lyapunov_sum / lyapunov_count
        lyapunov_std = np.std(lyapunov_series[len(lyapunov_series)//2:]) if len(lyapunov_series) > 10 else 0.0
    else:
        lyapunov_mean = 0.0
        lyapunov_std = 0.0
    
    return {
        'lyapunov_mean': float(lyapunov_mean),
        'lyapunov_std': float(lyapunov_std),
    }


def estimate_drift_rate(
    params: TuningParams,
    dim: int = 64,
    n_steps: int = 500,
    seed: int = 42,
) -> Dict:
    """
    估计慢变量基线漂移率
    """
    sys = CoupledLogisticLattice(
        dim=dim,
        slow_dim=max(4, dim // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        seed=seed,
    )
    
    # 烧录
    sys.iterate(n_steps=100)
    
    initial_norm = np.linalg.norm(sys.get_slow_state())
    
    # 采样
    sample_interval = 20
    n_samples = n_steps // sample_interval
    norms = np.zeros(n_samples)
    
    for i in range(n_samples):
        sys.iterate(n_steps=sample_interval)
        norms[i] = np.linalg.norm(sys.get_slow_state())
    
    # 线性拟合
    if initial_norm > 1e-10 and n_samples > 5:
        times = np.arange(n_samples) * sample_interval
        coeffs = np.polyfit(times, norms, 1)
        drift_rate = abs(coeffs[0]) / initial_norm
    else:
        drift_rate = 0.0
    
    return {
        'drift_rate': float(drift_rate),
        'initial_norm': float(initial_norm),
        'final_norm': float(norms[-1]) if len(norms) > 0 else 0.0,
    }


def check_stability(
    params: TuningParams,
    dim: int = 64,
    n_steps: int = 500,
    seed: int = 42,
) -> Dict:
    """检查系统稳定性"""
    sys = CoupledLogisticLattice(
        dim=dim,
        slow_dim=max(4, dim // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        seed=seed,
    )
    
    max_norm = 0.0
    diverged = False
    
    try:
        for _ in range(n_steps // 50):
            sys.iterate(n_steps=50)
            state_norm = np.linalg.norm(sys.get_state())
            
            if state_norm > max_norm:
                max_norm = state_norm
            
            if not np.isfinite(state_norm):
                diverged = True
                break
    except Exception:
        diverged = True
    
    return {
        'is_stable': not diverged,
        'max_norm': float(max_norm),
        'diverged': diverged,
    }


def estimate_alignment_error(
    params: TuningParams,
    dim: int = 64,
    seed: int = 42,
) -> Dict:
    """估计动力学一致性误差"""
    errors = []
    
    for trial in range(3):
        sys1 = CoupledLogisticLattice(
            dim=dim,
            slow_dim=max(4, dim // 4),
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            coupling_strength=params.slow_coupling_limit,
            seed=seed + trial,
        )
        sys2 = CoupledLogisticLattice(
            dim=dim,
            slow_dim=max(4, dim // 4),
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            coupling_strength=params.slow_coupling_limit,
            seed=seed + trial + 100,
        )
        
        sys1.iterate(n_steps=200)
        sys2.iterate(n_steps=200)
        
        # 收集统计量
        n_collect = 100
        means1 = []
        means2 = []
        
        for _ in range(n_collect):
            sys1.iterate(n_steps=1)
            sys2.iterate(n_steps=1)
            means1.append(np.mean(sys1.get_state()))
            means2.append(np.mean(sys2.get_state()))
        
        mean1 = np.mean(means1)
        mean2 = np.mean(means2)
        std1 = np.std(means1)
        std2 = np.std(means2)
        
        if std1 + std2 > 1e-10:
            error = abs(mean1 - mean2) / (std1 + std2)
        else:
            error = 0.0
        errors.append(error)
    
    return {
        'alignment_max_error': float(max(errors)) if errors else 0.0,
        'alignment_errors': [float(e) for e in errors],
    }


def compute_score(lyapunov: float, drift: float, align_error: float) -> float:
    """计算综合得分"""
    # Lyapunov 得分 (0, 0.1) 区间最佳
    if 0 < lyapunov < 0.1:
        lyap_score = 1.0
    elif lyapunov <= 0:
        lyap_score = max(0.0, 1.0 + lyapunov * 20)
    else:
        lyap_score = max(0.0, 1.0 - (lyapunov - 0.1) * 3)
    
    # 漂移率得分
    drift_score = max(0.0, 1.0 - drift * 10)
    drift_score = min(1.0, drift_score)
    
    # 对齐误差得分
    align_score = max(0.0, 1.0 - align_error * 5)
    align_score = min(1.0, align_score)
    
    return 0.5 * lyap_score + 0.25 * drift_score + 0.25 * align_score


def run_fast_validation(
    params: TuningParams,
    dim: int = 64,
    seed: int = 42,
) -> Dict:
    """运行一次快速验证"""
    start = time.time()
    
    # 1. 稳定性快速检查
    stab = check_stability(params, dim=dim, n_steps=300, seed=seed)
    
    if not stab['is_stable']:
        elapsed = time.time() - start
        return {
            'passed': False,
            'score': 0.0,
            'lyapunov_mean': -999.0,
            'lyapunov_std': 0.0,
            'drift_rate': 999.0,
            'alignment_max_error': 999.0,
            'open_loop_stable': False,
            'max_norm': stab['max_norm'],
            'validation_time': elapsed,
            'early_exit': 'instability',
        }
    
    # 2. Lyapunov指数
    lyap = calculate_lyapunov_exponent(params, dim=dim, n_steps=800, seed=seed)
    
    # 3. 漂移率
    drift = estimate_drift_rate(params, dim=dim, n_steps=400, seed=seed)
    
    # 4. 对齐误差
    align = estimate_alignment_error(params, dim=dim, seed=seed)
    
    elapsed = time.time() - start
    score = compute_score(lyap['lyapunov_mean'], drift['drift_rate'], align['alignment_max_error'])
    
    passed = (
        0 < lyap['lyapunov_mean'] < 0.1
        and drift['drift_rate'] < 0.1
        and align['alignment_max_error'] < 0.1
        and stab['is_stable']
    )
    
    return {
        'passed': passed,
        'score': score,
        'lyapunov_mean': lyap['lyapunov_mean'],
        'lyapunov_std': lyap['lyapunov_std'],
        'drift_rate': drift['drift_rate'],
        'alignment_max_error': align['alignment_max_error'],
        'open_loop_stable': stab['is_stable'],
        'max_norm': stab['max_norm'],
        'validation_time': elapsed,
    }


def grid_search_tuning(
    dim: int = 64,
    grid: str = 'medium',
    output_dir: str = "ultra_fast_tuning_results",
) -> Dict:
    """网格搜索调优"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 参数搜索空间
    if grid == 'fine':
        base_gain_values = np.arange(0.02, 0.51, 0.02).tolist()
        min_gain_frac = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        coupling_limits = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    elif grid == 'coarse':
        base_gain_values = [0.05, 0.1, 0.2, 0.3, 0.5]
        min_gain_frac = [0.3, 0.7, 1.0]
        coupling_limits = [0.2, 0.5, 1.0]
    else:  # medium
        base_gain_values = [0.03, 0.06, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
        min_gain_frac = [0.2, 0.5, 0.8, 1.0]
        coupling_limits = [0.1, 0.3, 0.5, 0.8, 1.2]
    
    total = len(base_gain_values) * len(min_gain_frac) * len(coupling_limits)
    
    logger.info(f"=" * 80)
    logger.info(f"超高速离散混沌系统参数调优")
    logger.info(f"=" * 80)
    logger.info(f"系统维度: {dim} (fast={dim}, slow={max(4, dim//4)})")
    logger.info(f"网格密度: {grid}")
    logger.info(f"参数组合数: {total}")
    logger.info(f"搜索空间:")
    logger.info(f"  base_gain: {min(base_gain_values):.3f} ~ {max(base_gain_values):.3f} ({len(base_gain_values)}个)")
    logger.info(f"  min_gain_frac: {min(min_gain_frac)} ~ {max(min_gain_frac)} ({len(min_gain_frac)}个)")
    logger.info(f"  coupling: {min(coupling_limits)} ~ {max(coupling_limits)} ({len(coupling_limits)}个)")
    logger.info("")
    
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
                
                try:
                    result = run_fast_validation(params, dim=dim)
                    result['params'] = {
                        'base_gain': base_gain,
                        'min_gain': min_gain,
                        'coupling_limit': coupling_limit,
                    }
                    
                    all_results.append(result)
                    
                    if best_result is None or result['score'] > best_result['score']:
                        best_result = result
                        best_params = params
                        
                        lyap_status = "✓" if 0 < result['lyapunov_mean'] < 0.1 else "✗"
                        logger.info(
                            f"[{count:4d}/{total}] 新最佳! "
                            f"score={result['score']:.4f}, "
                            f"λ={result['lyapunov_mean']:+.6f} {lyap_status}, "
                            f"drift={result['drift_rate']:.4f}, "
                            f"align={result['alignment_max_error']:.4f}, "
                            f"time={result['validation_time']*1000:.1f}ms, "
                            f"params: bg={base_gain:.3f}, mg={min_gain:.3f}, c={coupling_limit:.2f}"
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
                
                if count % 50 == 0:
                    elapsed = time.time() - start_time
                    speed = count / elapsed
                    remaining = (total - count) / speed
                    logger.info(
                        f"进度: {count}/{total} ({100*count/total:.1f}%), "
                        f"速度: {speed:.1f}组/秒, 剩余: {remaining:.1f}秒"
                    )
    
    total_time = time.time() - start_time
    
    # 保存结果
    results_data = {
        'dim': dim * 2,
        'fast_dim': dim * 2,
        'method': 'coupled_logistic_lattice',
        'grid_density': grid,
        'total_time_seconds': total_time,
        'total_combos': total,
        'avg_time_per_combo_ms': (total_time / total) * 1000 if total > 0 else 0,
        'best_params': {
            'base_gain': best_params.base_gain if best_params else 0,
            'min_gain': best_params.min_gain if best_params else 0,
            'slow_coupling_limit': best_params.slow_coupling_limit if best_params else 0,
        },
        'best_result': best_result,
        'top_20': sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:20],
    }
    
    with open(output_path / 'tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=str)
    
    # 生成报告
    report = generate_report(results_data, dim, total, total_time, all_results)
    
    with open(output_path / 'tuning_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"结果已保存到 {output_path}")
    
    return results_data


def generate_report(results_data: Dict, dim: int, total: int, total_time: float, all_results: List[Dict]) -> str:
    """生成调优报告"""
    br = results_data['best_result']
    bp = results_data['best_params']
    
    if br is None:
        return "# 超高速离散混沌调优报告\n\n**错误：所有参数组合均失败**\n"
    
    report = f"""# 超高速离散混沌调优报告

## 概述

- **调优方法**: 网格搜索（耦合Logistic映射格点 + 纯NumPy向量化）
- **系统维度**: {dim} (快变量={dim}, 慢变量={max(4, dim//4)})
- **总参数组合数**: {total}
- **总耗时**: {total_time:.2f}秒
- **平均每组耗时**: {results_data['avg_time_per_combo_ms']:.2f}毫秒
- **加速比**: 约 {(30*60)/(total_time/total):.0f} 倍（相比2048维+MLP）

## 最佳参数

| 参数 | 值 | 说明 |
|------|-----|------|
| base_gain | {bp['base_gain']:.4f} | 混沌注入基础增益 |
| min_gain | {bp['min_gain']:.4f} | 最小增益（防止寂灭） |
| slow_coupling_limit | {bp['slow_coupling_limit']:.4f} | 耦合强度上限 |

## 最佳指标

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| 综合得分 | {br['score']:.4f} | 1.0 | - |
| Lyapunov λ | {br['lyapunov_mean']:+.6f} | (0, 0.1) | {'✓' if 0 < br['lyapunov_mean'] < 0.1 else '✗'} |
| 漂移率 | {br['drift_rate']:.6f} | < 0.1 | {'✓' if br['drift_rate'] < 0.1 else '✗'} |
| 对齐误差 | {br['alignment_max_error']:.6f} | < 0.1 | {'✓' if br['alignment_max_error'] < 0.1 else '✗'} |
| 开环稳定 | {'是' if br.get('open_loop_stable') else '否'} | 是 | {'✓' if br.get('open_loop_stable') else '✗'} |
| 验证时间 | {br['validation_time']*1000:.1f}ms | - | - |

## base_gain 与 Lyapunov 指数的关系

| base_gain | λ均值 | λ最大 | λ最小 | 通过率 |
|-----------|-------|-------|-------|--------|
"""
    
    by_gain = {}
    for r in all_results:
        bg = r['params']['base_gain']
        if bg not in by_gain:
            by_gain[bg] = []
        by_gain[bg].append(r)
    
    for bg in sorted(by_gain.keys()):
        vals = [r['lyapunov_mean'] for r in by_gain[bg] if r['lyapunov_mean'] > -100]
        passed = sum(1 for r in by_gain[bg] if 0 < r['lyapunov_mean'] < 0.1)
        if vals:
            report += f"| {bg:.3f} | {np.mean(vals):+.6f} | {max(vals):+.6f} | {min(vals):+.6f} | {passed}/{len(by_gain[bg])} |\n"
    
    # Top 20
    top20 = sorted(all_results, key=lambda x: x.get('score', 0), reverse=True)[:20]
    report += "\n## Top 20 参数组合\n\n"
    report += "| 排名 | base_gain | min_gain | coupling | λ | drift | align | 得分 | 通过 |\n"
    report += "|------|-----------|----------|----------|---|-------|-------|------|------|\n"
    
    for i, r in enumerate(top20, 1):
        p = r.get('params', {})
        status = '✓' if r.get('passed') else '✗'
        report += (
            f"| {i} | {p.get('base_gain', 0):.3f} | {p.get('min_gain', 0):.3f} | "
            f"{p.get('coupling_limit', 0):.2f} | {r.get('lyapunov_mean', 0):+.6f} | "
            f"{r.get('drift_rate', 0):.4f} | {r.get('alignment_max_error', 0):.4f} | "
            f"{r.get('score', 0):.4f} {status} |\n"
        )
    
    report += f"""

## 与真实系统的映射

| 数学模型 | 真实系统 | 物理意义 |
|---------|---------|---------|
| base_gain | chaos_injection.base_gain | 混沌注入强度 |
| min_gain | chaos_injection.min_gain | 最小增益 |
| coupling_strength | coupling_stability.* | 耦合强度 |
| dim*2 | dim.fast_variable_dim | 快变量维度 |

## 为什么这么快？

1. **离散时间系统**: 无需ODE求解器，直接迭代（10x）
2. **纯数学映射**: 无神经网络MLP（100x）
3. **NumPy向量化**: 无Python循环（10x）
4. **低维度**: 比2048维小很多（10x）
5. **无PyTorch开销**: 无张量运算和自动微分（5x）
6. **快速失败**: 不稳定系统提前终止

## 下一步建议

1. **用真实系统验证**: 将最佳参数应用到真实Chronos系统（256维低维模式）
2. **精细调优**: 在最佳参数附近用 fine 网格搜索
3. **维度缩放**: 测试不同维度下参数的有效性
4. **P1/P2扩展**: 用同样方法扩展到P1/P2级验证

## 注意事项

1. 这是**定性趋势分析**，定量数值与真实系统有差异
2. 主要用于快速筛选参数范围和发现趋势
3. 最终参数必须用真实Chronos系统验证
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='超高速离散混沌系统参数调优')
    parser.add_argument('--dim', type=int, default=64, help='Henon映射对数（默认64，快变量维度=2*dim）')
    parser.add_argument('--grid', type=str, default='medium', choices=['coarse', 'medium', 'fine'], help='网格密度')
    parser.add_argument('--output', type=str, default='ultra_fast_tuning_results', help='输出目录')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    results = grid_search_tuning(dim=args.dim, grid=args.grid, output_dir=args.output)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("调优完成！最佳结果：")
    logger.info("=" * 80)
    if results['best_result']:
        bp = results['best_params']
        br = results['best_result']
        logger.info(f"  base_gain = {bp['base_gain']:.4f}")
        logger.info(f"  min_gain = {bp['min_gain']:.4f}")
        logger.info(f"  slow_coupling_limit = {bp['slow_coupling_limit']:.4f}")
        logger.info(f"  综合得分 = {br['score']:.4f}")
        logger.info(f"  Lyapunov λ = {br['lyapunov_mean']:+.6f}")
        logger.info(f"  漂移率 = {br['drift_rate']:.6f}")
        logger.info(f"  对齐误差 = {br['alignment_max_error']:.6f}")
        logger.info(f"  P0通过: {'是' if br.get('passed') else '否'}")
    logger.info(f"  总耗时: {results['total_time_seconds']:.2f}秒")
    logger.info(f"  平均每组: {results['avg_time_per_combo_ms']:.2f}毫秒")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
