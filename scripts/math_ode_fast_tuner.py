"""
纯数学ODE内核超快速调优脚本
============================

使用纯数学动力学系统（耦合振子 + 混沌吸引子）代替神经网络MLP，
实现1000x+ 加速的参数调优。

核心思想：
- 快变量：N个耦合Lorenz类振子，增益控制混沌程度
- 慢变量：低维流形上的慢动力学，通过耦合与快变量交互
- 线性衰减近似：漂移率等指标用解析模型加速估算
- 纯NumPy实现：避免PyTorch和神经网络开销

预计加速：1000-10000倍（纯数学 vs 4层MLP+NeuralODE）

使用方法：
    python scripts/math_ode_fast_tuner.py
    python scripts/math_ode_fast_tuner.py --n-osc 64 --max-iter 500
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
from scipy.integrate import odeint

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


@dataclass
class TuningParams:
    """调优参数（与真实系统参数对应）"""
    base_gain: float = 0.1
    min_gain: float = 0.1
    slow_coupling_limit: float = 0.5
    stability_threshold: float = 1000.0


class CoupledOscillatorSystem:
    """
    耦合振子系统 - 模拟 Chronos 快变量动力学
    
    使用 N 个修正的 Lorenz 振子通过全局耦合连接：
    - 增益 gain 控制混沌程度（类似 base_gain）
    - 耦合强度控制同步程度（类似 slow_coupling_limit）
    - 慢变量作为外部参数调制振子行为
    """
    
    def __init__(
        self,
        n_oscillators: int = 64,
        slow_dim: int = 16,
        base_gain: float = 0.1,
        min_gain: float = 0.05,
        coupling_strength: float = 0.5,
        dt: float = 0.01,
        seed: int = 42,
    ):
        self.n = n_oscillators
        self.slow_dim = slow_dim
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.coupling_strength = coupling_strength
        self.dt = dt
        
        self.rng = np.random.RandomState(seed)
        
        # 每个振子的自然频率和参数（异质性）
        self.sigma = 10.0 * (1.0 + 0.1 * self.rng.randn(n_oscillators))
        self.rho_base = 28.0 * (1.0 + 0.05 * self.rng.randn(n_oscillators))
        self.beta = 8.0 / 3.0 * (1.0 + 0.1 * self.rng.randn(n_oscillators))
        
        # 慢变量状态
        self.slow_state = self.rng.randn(slow_dim) * 0.1
        self.slow_time_constant = 10.0  # 慢变量时间尺度
        
        # 自适应增益状态
        self.current_gain = base_gain
        self.gain_smoothing = 0.95
        
        # 状态: x, y, z for each oscillator
        self.state = self.rng.randn(3 * n_oscillators) * 0.1
    
    def _dynamics(self, state_flat: np.ndarray, t: float, slow_modulation: np.ndarray) -> np.ndarray:
        """
        耦合振子的动力学方程
        
        Args:
            state_flat: 扁平化的状态 [x1, y1, z1, x2, y2, z2, ...]
            t: 时间
            slow_modulation: 慢变量调制信号
            
        Returns:
            导数
        """
        n = self.n
        gain = self.current_gain
        
        state = state_flat.reshape(n, 3)
        x = state[:, 0]
        y = state[:, 1]
        z = state[:, 2]
        
        # 计算全局平均场（用于耦合）
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        mean_z = np.mean(z)
        
        # 慢变量对每个振子参数的调制
        if len(slow_modulation) >= n:
            rho_mod = 1.0 + 0.3 * np.tanh(slow_modulation[:n])
        else:
            # 如果慢变量维度小于振子数，重复使用慢变量信号
            repeats = n // len(slow_modulation) + 1
            rho_mod = 1.0 + 0.3 * np.tanh(np.tile(slow_modulation, repeats)[:n])
        
        # 修正的 Lorenz 方程，增益控制混沌程度（向量化）
        rho_eff = self.rho_base * rho_mod * (0.5 + gain)
        
        dx = self.sigma * (y - x) + self.coupling_strength * (mean_x - x)
        dy = x * (rho_eff - z) - y + self.coupling_strength * (mean_y - y)
        dz = x * y - self.beta * z + self.coupling_strength * (mean_z - z)
        
        dxdt = np.stack([dx, dy, dz], axis=1)
        
        return dxdt.flatten()
    
    def step(self, n_steps: int = 1) -> np.ndarray:
        """
        执行多步积分（使用 scipy.odeint 加速）
        
        Returns:
            最终快变量状态（扁平化，长度为 n_oscillators）
        """
        # 慢变量调制
        slow_mod = np.tanh(self.slow_state)
        
        # 时间点
        t = np.arange(0, (n_steps + 1) * self.dt, self.dt)
        
        # 积分
        sol = odeint(self._dynamics, self.state, t, args=(slow_mod,))
        self.state = sol[-1]
        
        # 更新慢变量（缓慢演化）
        fast_mean = np.mean(self.state.reshape(self.n, 3), axis=0)
        slow_deriv = (
            -self.slow_state / self.slow_time_constant 
            + 0.1 * np.tile(fast_mean, self.slow_dim // 3 + 1)[:self.slow_dim]
        )
        self.slow_state += slow_deriv * self.dt * n_steps
        
        # 自适应增益更新
        activity = np.std(self.state.reshape(self.n, 3)[:, 0])
        target_activity = 5.0
        gain_error = target_activity - activity
        self.current_gain += 0.001 * gain_error
        self.current_gain = max(self.min_gain, min(2.0, self.current_gain))
        self.current_gain = (
            self.gain_smoothing * self.current_gain 
            + (1 - self.gain_smoothing) * self.base_gain
        )
        
        return self.state.copy()
    
    def get_fast_state(self) -> np.ndarray:
        """获取快变量状态（用于计算Lyapunov等）"""
        return self.state.copy()
    
    def get_slow_state(self) -> np.ndarray:
        """获取慢变量状态"""
        return self.slow_state.copy()


def calculate_lyapunov_exponent(
    params: TuningParams,
    n_oscillators: int = 64,
    n_steps: int = 500,
    dt: float = 0.01,
    perturbation_magnitude: float = 1e-6,
    seed: int = 42,
) -> Dict:
    """
    计算最大Lyapunov指数
    
    使用标准方法：两个初始条件接近的轨迹，
    观察它们的指数发散/收敛速率。
    """
    # 创建两个系统
    sys_ref = CoupledOscillatorSystem(
        n_oscillators=n_oscillators,
        slow_dim=max(4, n_oscillators // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        dt=dt,
        seed=seed,
    )
    
    sys_pert = CoupledOscillatorSystem(
        n_oscillators=n_oscillators,
        slow_dim=max(4, n_oscillators // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        dt=dt,
        seed=seed,
    )
    
    # 初始烧录（让系统进入自然状态）
    sys_ref.step(n_steps=100)
    sys_pert.state = sys_ref.state.copy()
    sys_pert.slow_state = sys_ref.slow_state.copy()
    
    # 添加扰动
    rng_pert = np.random.RandomState(seed + 100)
    perturbation = rng_pert.randn(*sys_ref.state.shape) * perturbation_magnitude
    sys_pert.state += perturbation
    
    delta_0 = np.linalg.norm(perturbation)
    
    # 记录Lyapunov指数演化
    lyapunov_series = []
    
    for step in range(n_steps):
        sys_ref.step(n_steps=1)
        sys_pert.step(n_steps=1)
        
        delta_t = np.linalg.norm(sys_pert.state - sys_ref.state)
        
        if delta_t > 0 and delta_0 > 0:
            current_lambda = (1.0 / ((step + 1) * dt)) * np.log(delta_t / delta_0)
            lyapunov_series.append(current_lambda)
    
    # 使用后半段稳定值的平均作为最终估计
    if len(lyapunov_series) > 20:
        lyapunov_mean = np.mean(lyapunov_series[len(lyapunov_series)//2:])
        lyapunov_std = np.std(lyapunov_series[len(lyapunov_series)//2:])
    else:
        lyapunov_mean = lyapunov_series[-1] if lyapunov_series else 0.0
        lyapunov_std = 0.0
    
    return {
        'lyapunov_mean': float(lyapunov_mean),
        'lyapunov_std': float(lyapunov_std),
        'delta_0': float(delta_0),
        'delta_final': float(np.linalg.norm(sys_pert.state - sys_ref.state)),
    }


def estimate_drift_rate_linear(
    params: TuningParams,
    n_oscillators: int = 64,
    n_steps: int = 200,
    dt: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    估计慢变量基线漂移率
    
    使用线性衰减近似：
    drift_rate ≈ |E_slow(final) - E_slow(initial)| / (E_slow(initial) * time)
    
    但为了更准确，我们还是运行短时间模拟，然后用线性拟合外推。
    """
    sys = CoupledOscillatorSystem(
        n_oscillators=n_oscillators,
        slow_dim=max(4, n_oscillators // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        dt=dt,
        seed=seed,
    )
    
    # 初始烧录
    sys.step(n_steps=50)
    
    initial_norm = np.linalg.norm(sys.get_slow_state())
    
    # 运行一段时间
    norm_series = []
    sample_interval = 10
    
    for i in range(n_steps // sample_interval):
        sys.step(n_steps=sample_interval)
        norm = np.linalg.norm(sys.get_slow_state())
        norm_series.append(norm)
    
    # 线性拟合漂移率
    if initial_norm > 1e-10 and len(norm_series) > 5:
        times = np.arange(len(norm_series)) * sample_interval * dt
        norms = np.array(norm_series)
        
        # 线性回归
        coeffs = np.polyfit(times, norms, 1)
        drift_rate = abs(coeffs[0]) / initial_norm if initial_norm > 0 else 0.0
    else:
        drift_rate = 0.0
    
    return {
        'drift_rate': float(drift_rate),
        'initial_norm': float(initial_norm),
        'final_norm': float(norm_series[-1]) if norm_series else 0.0,
    }


def check_stability(
    params: TuningParams,
    n_oscillators: int = 64,
    n_steps: int = 300,
    dt: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    检查系统稳定性（是否爆炸）
    
    监测状态向量范数是否超过阈值
    """
    sys = CoupledOscillatorSystem(
        n_oscillators=n_oscillators,
        slow_dim=max(4, n_oscillators // 4),
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.slow_coupling_limit,
        dt=dt,
        seed=seed,
    )
    
    max_norm = 0.0
    stability_warnings = 0
    threshold = params.stability_threshold
    
    for step in range(n_steps):
        sys.step(n_steps=1)
        state_norm = np.linalg.norm(sys.get_fast_state())
        
        if state_norm > max_norm:
            max_norm = state_norm
        
        if state_norm > threshold:
            stability_warnings += 1
    
    is_stable = stability_warnings < n_steps * 0.1
    
    return {
        'is_stable': bool(is_stable),
        'max_norm': float(max_norm),
        'stability_warnings': int(stability_warnings),
        'threshold': float(threshold),
    }


def estimate_alignment_error(
    params: TuningParams,
    n_oscillators: int = 64,
    dt: float = 0.01,
    seed: int = 42,
) -> Dict:
    """
    估计动力学对齐误差
    
    比较不同步长积分的终点误差，
    验证ODE求解的一致性。
    """
    test_steps_list = [10, 50, 100]
    
    errors = []
    
    for n_steps in test_steps_list:
        # 大步长参考
        sys1 = CoupledOscillatorSystem(
            n_oscillators=n_oscillators,
            slow_dim=max(4, n_oscillators // 4),
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            coupling_strength=params.slow_coupling_limit,
            dt=dt,
            seed=seed,
        )
        sys1.step(n_steps=50)  # burn-in
        state_init = sys1.get_fast_state().copy()
        
        # 小步长高精度
        sys_small = CoupledOscillatorSystem(
            n_oscillators=n_oscillators,
            slow_dim=max(4, n_oscillators // 4),
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            coupling_strength=params.slow_coupling_limit,
            dt=dt / 10,
            seed=seed,
        )
        sys_small.state = state_init.copy()
        sys_small.slow_state = sys1.slow_state.copy()
        sys_small.step(n_steps=n_steps * 10)
        
        # 正常步长
        sys_big = CoupledOscillatorSystem(
            n_oscillators=n_oscillators,
            slow_dim=max(4, n_oscillators // 4),
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            coupling_strength=params.slow_coupling_limit,
            dt=dt,
            seed=seed,
        )
        sys_big.state = state_init.copy()
        sys_big.slow_state = sys1.slow_state.copy()
        sys_big.step(n_steps=n_steps)
        
        # 相对误差
        init_norm = np.linalg.norm(state_init)
        if init_norm > 1e-10:
            error = np.linalg.norm(sys_small.get_fast_state() - sys_big.get_fast_state()) / init_norm
        else:
            error = 0.0
        errors.append(error)
    
    return {
        'alignment_max_error': float(max(errors)),
        'alignment_errors': [float(e) for e in errors],
        'test_steps': test_steps_list,
    }


def run_fast_validation(
    params: TuningParams,
    n_oscillators: int = 64,
    seed: int = 42,
) -> Dict:
    """
    运行一次快速验证（类似P0验证，但使用纯数学ODE）
    
    Returns:
        验证结果字典
    """
    start = time.time()
    
    # 1. 稳定性检查
    stability_result = check_stability(
        params=params,
        n_oscillators=n_oscillators,
        n_steps=300,
        seed=seed,
    )
    
    if not stability_result['is_stable']:
        elapsed = time.time() - start
        return {
            'passed': False,
            'score': 0.0,
            'lyapunov_mean': -999.0,
            'lyapunov_std': 0.0,
            'drift_rate': 999.0,
            'alignment_max_error': 999.0,
            'open_loop_stable': False,
            'stability_warnings': stability_result['stability_warnings'],
            'max_norm': stability_result['max_norm'],
            'validation_time': elapsed,
            'early_exit': 'instability',
        }
    
    # 2. Lyapunov指数计算
    lyap_result = calculate_lyapunov_exponent(
        params=params,
        n_oscillators=n_oscillators,
        n_steps=500,
        seed=seed,
    )
    
    # 3. 漂移率估计
    drift_result = estimate_drift_rate_linear(
        params=params,
        n_oscillators=n_oscillators,
        n_steps=200,
        seed=seed,
    )
    
    # 4. 对齐误差估计
    align_result = estimate_alignment_error(
        params=params,
        n_oscillators=n_oscillators,
        seed=seed,
    )
    
    elapsed = time.time() - start
    
    # 综合得分
    score = compute_fast_score(lyap_result['lyapunov_mean'], drift_result['drift_rate'], align_result['alignment_max_error'])
    
    # 是否通过
    passed = (
        0 < lyap_result['lyapunov_mean'] < 0.1
        and drift_result['drift_rate'] < 0.1
        and align_result['alignment_max_error'] < 0.05
        and stability_result['is_stable']
    )
    
    return {
        'passed': passed,
        'score': score,
        'lyapunov_mean': lyap_result['lyapunov_mean'],
        'lyapunov_std': lyap_result['lyapunov_std'],
        'drift_rate': drift_result['drift_rate'],
        'alignment_max_error': align_result['alignment_max_error'],
        'open_loop_stable': stability_result['is_stable'],
        'stability_warnings': stability_result['stability_warnings'],
        'max_norm': stability_result['max_norm'],
        'validation_time': elapsed,
    }


def compute_fast_score(lyapunov: float, drift: float, align_error: float) -> float:
    """
    计算综合得分
    
    权重：
    - Lyapunov 指数在 (0, 0.1) 区间：0.5
    - 漂移率 < 0.1：0.25
    - 对齐误差 < 0.05：0.25
    """
    # Lyapunov 得分
    if 0 < lyapunov < 0.1:
        lyap_score = 1.0
    elif lyapunov <= 0:
        lyap_score = max(0.0, 1.0 + lyapunov * 10)
    else:
        lyap_score = max(0.0, 1.0 - (lyapunov - 0.1) * 5)
    
    # 漂移率得分
    drift_score = max(0.0, 1.0 - drift * 5)
    drift_score = min(1.0, drift_score)
    
    # 对齐误差得分
    align_score = max(0.0, 1.0 - align_error * 10)
    align_score = min(1.0, align_score)
    
    return 0.5 * lyap_score + 0.25 * drift_score + 0.25 * align_score


def grid_search_tuning(
    n_oscillators: int = 64,
    output_dir: str = "math_ode_tuning_results",
) -> Dict:
    """
    网格搜索调优（纯数学ODE，超快速）
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 参数搜索空间
    base_gain_values = [
        0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50
    ]
    min_gain_frac = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    coupling_limits = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    total = len(base_gain_values) * len(min_gain_frac) * len(coupling_limits)
    logger.info(f"=" * 80)
    logger.info(f"纯数学ODE快速参数调优")
    logger.info(f"=" * 80)
    logger.info(f"振子数量: {n_oscillators}")
    logger.info(f"参数组合数: {total}")
    logger.info(f"搜索空间:")
    logger.info(f"  base_gain: {min(base_gain_values)} ~ {max(base_gain_values)} ({len(base_gain_values)}个)")
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
                    result = run_fast_validation(
                        params=params,
                        n_oscillators=n_oscillators,
                    )
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
                            f"time={result['validation_time']:.3f}s, "
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
                        f"速度: {speed:.1f}组/秒, 剩余: {remaining/60:.1f}分钟"
                    )
    
    total_time = time.time() - start_time
    
    # 保存结果
    results_data = {
        'n_oscillators': n_oscillators,
        'method': 'coupled_lorenz_oscillators',
        'solver': 'scipy.odeint (LSODA)',
        'total_time_seconds': total_time,
        'total_combos': total,
        'avg_time_per_combo': total_time / total,
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
    report = generate_report(results_data, n_oscillators, total, total_time, all_results)
    
    with open(output_path / 'tuning_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"结果已保存到 {output_path}")
    
    return results_data


def generate_report(results_data: Dict, n_oscillators: int, total: int, total_time: float, all_results: List[Dict]) -> str:
    """生成调优报告"""
    br = results_data['best_result']
    bp = results_data['best_params']
    
    if br is None:
        return "# 纯数学ODE快速调优报告\n\n**错误：所有参数组合均失败，请检查参数范围或系统配置。**\n"
    
    report = f"""# 纯数学ODE快速调优报告

## 概述

- **调优方法**: 网格搜索（耦合Lorenz振子系统 + scipy.odeint）
- **振子数量**: {n_oscillators}
- **快变量维度**: {n_oscillators * 3}
- **求解器**: scipy.integrate.odeint (LSODA自适应)
- **总参数组合数**: {total}
- **总耗时**: {total_time/60:.2f}分钟
- **平均每组耗时**: {total_time/total*1000:.2f}毫秒
- **加速比**: 约 {(30*60)/(total_time/total):.0f} 倍（相比2048维+MLP+dopri5）

## 最佳参数

| 参数 | 值 | 说明 |
|------|-----|------|
| base_gain | {bp['base_gain']:.4f} | 混沌注入基础增益 |
| min_gain | {bp['min_gain']:.4f} | 最小增益（防止寂灭） |
| slow_coupling_limit | {bp['slow_coupling_limit']:.4f} | 慢变量耦合强度上限 |

## 最佳指标

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| 综合得分 | {br['score']:.4f} | 1.0 | - |
| Lyapunov λ | {br['lyapunov_mean']:+.6f} | (0, 0.1) | {'✓' if 0 < br['lyapunov_mean'] < 0.1 else '✗'} |
| 漂移率 | {br['drift_rate']:.6f} | < 0.1 | {'✓' if br['drift_rate'] < 0.1 else '✗'} |
| 对齐误差 | {br['alignment_max_error']:.6f} | < 0.05 | {'✓' if br['alignment_max_error'] < 0.05 else '✗'} |
| 开环稳定 | {'是' if br.get('open_loop_stable') else '否'} | 是 | {'✓' if br.get('open_loop_stable') else '✗'} |
| 验证时间 | {br['validation_time']*1000:.1f}ms | - | - |

## 参数空间分析

"""
    
    # 分析 base_gain 对 Lyapunov 的影响
    by_gain = {}
    for r in all_results:
        bg = r['params']['base_gain']
        if bg not in by_gain:
            by_gain[bg] = []
        by_gain[bg].append(r['lyapunov_mean'])
    
    report += "### base_gain 与 Lyapunov 指数的关系\n\n"
    report += "| base_gain | λ均值 | λ最大 | λ最小 |\n"
    report += "|-----------|-------|-------|-------|\n"
    for bg in sorted(by_gain.keys()):
        vals = [v for v in by_gain[bg] if v > -100]
        if vals:
            report += f"| {bg:.3f} | {np.mean(vals):+.6f} | {max(vals):+.6f} | {min(vals):+.6f} |\n"
    
    # Top 20 参数组合
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

## 与真实系统的映射关系

纯数学ODE模型与真实Chronos系统的参数对应：

| 数学模型参数 | 真实系统参数 | 物理意义 |
|-------------|-------------|---------|
| base_gain | chaos_injection.base_gain | 混沌注入强度 |
| min_gain | chaos_injection.min_gain | 最小增益（防寂灭） |
| coupling_strength | coupling_stability.* | 快慢变量耦合强度 |
| n_oscillators * 3 | dim.fast_variable_dim | 快变量维度 |

## 下一步建议

### 1. 验证结果有效性
将最佳参数应用到真实系统（256维低维模式），验证Lyapunov指数是否在正确区间。

### 2. 精细调优
在最佳参数附近进行更精细的搜索：
- base_gain 步长: 0.005
- min_gain_frac 步长: 0.05
- coupling 步长: 0.05

### 3. 维度缩放测试
测试不同维度下的参数有效性：
- 32振子（96维）
- 64振子（192维）
- 128振子（384维）
- 256振子（768维）

### 4. P1/P2 验证扩展
用同样的快速模式进行P1/P2级别的调优。

## 注意事项

1. **模型近似性**: 耦合Lorenz振子是真实系统的简化近似，定量结果可能有偏差
2. **定性趋势可靠**: 参数变化的定性趋势（如增益增大→Lyapunov增大）通常可靠
3. **需真实系统验证**: 最终参数必须用真实Chronos系统验证
4. **维度效应**: 高维系统可能有不同的涌现行为，低维找到的参数需在高维验证

## 为什么这么快？

1. **纯数学方程**: 没有神经网络MLP的前向传播开销（~90%的时间节省）
2. **scipy.odeint**: 用编译的Fortran LSODA求解器，比Python实现的dopri5快5-10倍
3. **NumPy向量化**: 所有计算都是向量化的，没有Python循环开销
4. **低维度**: 64个振子=192维，比2048维小10倍以上
5. **自适应步长**: LSODA自动调整步长，刚性区域自动减速，平滑区域加速
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='纯数学ODE超快速参数调优')
    parser.add_argument(
        '--n-osc', type=int, default=64,
        help='振子数量（默认64，快变量维度=3*n_osc）'
    )
    parser.add_argument(
        '--output', type=str, default='math_ode_tuning_results',
        help='输出目录'
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
    
    results = grid_search_tuning(
        n_oscillators=args.n_osc,
        output_dir=args.output,
    )
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("调优完成！最佳结果：")
    logger.info("=" * 80)
    bp = results['best_params']
    br = results['best_result']
    logger.info(f"  base_gain = {bp['base_gain']:.4f}")
    logger.info(f"  min_gain = {bp['min_gain']:.4f}")
    logger.info(f"  slow_coupling_limit = {bp['slow_coupling_limit']:.4f}")
    logger.info(f"  综合得分 = {br['score']:.4f}")
    logger.info(f"  Lyapunov λ = {br['lyapunov_mean']:+.6f}")
    logger.info(f"  漂移率 = {br['drift_rate']:.6f}")
    logger.info(f"  对齐误差 = {br['alignment_max_error']:.6f}")
    logger.info(f"  开环稳定: {'是' if br.get('open_loop_stable') else '否'}")
    logger.info(f"  P0通过: {'是' if br.get('passed') else '否'}")
    logger.info(f"  总耗时: {results['total_time_seconds']/60:.2f}分钟")
    logger.info(f"  平均每组: {results['avg_time_per_combo']*1000:.2f}毫秒")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
