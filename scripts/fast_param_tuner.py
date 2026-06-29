"""
轻量级快速参数调优脚本
========================

使用简化的动力学模型（低维 + Euler法 + 纯数学ODE内核）快速搜索混沌注入参数，
找到边缘混沌区间后再用完整系统验证。

速度提升：
- 维度：2048 → 64  (快变量)
- 求解器：dopri5(6次函数评估/步) → Euler(1次/步)
- 动力学函数：4层MLP(8M FLOPs) → 简化非线性动力学(<10K FLOPs)
- 预计加速：500-1000倍

使用方法：
    python scripts/fast_param_tuner.py
    python scripts/fast_param_tuner.py --target-lyapunov 0.05
    python scripts/fast_param_tuner.py --output fast_tuning_results
"""

import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TuningParams:
    """调优参数"""
    base_gain: float = 0.1
    min_gain: float = 0.1
    coupling_strength: float = 0.5
    decay_rate: float = 0.1
    noise_scale: float = 0.01
    attractor_gain: float = 1.0


@dataclass
class TuningResult:
    """调优结果"""
    params: TuningParams
    lyapunov_lambda: float = 0.0
    drift_rate: float = 0.0
    autocorrelation: float = 0.0
    overall_score: float = 0.0
    passed: bool = False
    metrics: Dict = field(default_factory=dict)


class SimplifiedDynamics:
    """
    简化动力学系统
    
    使用低维非线性动力学方程模拟Chronos系统的核心行为：
    - 快变量：阻尼振荡 + 混沌注入 + 噪声
    - 慢变量：线性衰减 + 快变量耦合
    - 耦合系统：自适应耦合系数
    
    维度：fast_dim=64, slow_dim=16
    方法：Euler法（固定步长）
    """
    
    def __init__(
        self,
        fast_dim: int = 64,
        slow_dim: int = 16,
        params: Optional[TuningParams] = None,
        dt: float = 0.01,
        seed: int = 42
    ):
        self.fast_dim = fast_dim
        self.slow_dim = slow_dim
        self.params = params or TuningParams()
        self.dt = dt
        self.rng = np.random.RandomState(seed)
        
        # 初始化状态
        self.E_fast = self.rng.randn(fast_dim) * 0.1
        self.E_slow = self.rng.randn(slow_dim) * 0.01
        
        # 内部振荡频率（随机分布）
        self.frequencies = np.abs(self.rng.randn(fast_dim)) * 2.0 + 0.5
        
        # 耦合矩阵（稀疏随机）
        self.coupling_matrix = self.rng.randn(slow_dim, fast_dim) * 0.01
        
        # 吸引子状态（简化的Lorenz-like 3维系统）
        self.attractor_state = np.array([1.0, 1.0, 1.0])
        self.attractor_params = {
            'sigma': 10.0,
            'rho': 28.0,
            'beta': 8.0 / 3.0
        }
        
        # 混沌注入投影矩阵
        self.injector_W = self.rng.randn(fast_dim, 3) * 0.1 / np.sqrt(3)
        
        # 当前增益
        self.current_gain = self.params.base_gain
        
        # 自适应增益状态
        self.variance_estimate = 1.0
        self.target_variance = 1.0
        
    def reset(self, seed: Optional[int] = None):
        """重置系统状态"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.E_fast = self.rng.randn(self.fast_dim) * 0.1
        self.E_slow = self.rng.randn(self.slow_dim) * 0.01
        self.attractor_state = np.array([1.0, 1.0, 1.0])
        self.current_gain = self.params.base_gain
        self.variance_estimate = 1.0
    
    def _step_attractor(self) -> np.ndarray:
        """步进吸引子（简化Lorenz系统）"""
        x, y, z = self.attractor_state
        s = self.attractor_params['sigma']
        r = self.attractor_params['rho']
        b = self.attractor_params['beta']
        
        dx = s * (y - x)
        dy = x * (r - z) - y
        dz = x * y - b * z
        
        self.attractor_state += self.dt * np.array([dx, dy, dz]) * self.params.attractor_gain
        return self.attractor_state.copy()
    
    def _update_adaptive_gain(self):
        """更新自适应增益"""
        current_var = np.var(self.E_fast)
        
        alpha = 0.01
        self.variance_estimate = (1 - alpha) * self.variance_estimate + alpha * current_var
        
        # 自适应增益公式：g = g0 * 2.0 / (1.0 + 0.3 * Var/σ²_target)
        var_ratio = self.variance_estimate / self.target_variance
        target_gain = self.params.base_gain * 2.0 / (1.0 + 0.3 * var_ratio)
        
        # 应用最小增益限制
        target_gain = max(target_gain, self.params.min_gain)
        
        # 平滑更新
        self.current_gain = 0.99 * self.current_gain + 0.01 * target_gain
    
    def step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行单步演化（Euler法）
        
        Returns:
            (E_fast_new, E_slow_new)
        """
        # 1. 步进吸引子
        attractor_out = self._step_attractor()
        
        # 2. 更新自适应增益
        self._update_adaptive_gain()
        
        # 3. 计算快变量动力学
        # 阻尼振荡项
        oscillation = -self.params.decay_rate * self.E_fast
        
        # 内部非线性项（简单的立方非线性）
        nonlinear = 0.1 * self.E_fast**3 - 0.05 * self.E_fast
        
        # 慢变量耦合项
        slow_coupling = self.params.coupling_strength * (self.coupling_matrix.T @ self.E_slow)
        
        # 混沌注入项
        chaos_injection = self.current_gain * (self.injector_W @ attractor_out)
        
        # 噪声项
        noise = self.params.noise_scale * self.rng.randn(self.fast_dim)
        
        # 组合
        dE_fast = oscillation + nonlinear + slow_coupling + chaos_injection + noise
        
        # Euler更新
        E_fast_new = self.E_fast + self.dt * dE_fast
        
        # 4. 计算慢变量动力学（线性衰减 + 快变量驱动）
        slow_decay = -0.01 * self.E_slow
        fast_drive = 0.001 * (self.coupling_matrix @ self.E_fast)
        dE_slow = slow_decay + fast_drive
        
        E_slow_new = self.E_slow + self.dt * dE_slow
        
        # 5. 范数裁剪（防止发散）
        fast_norm = np.linalg.norm(E_fast_new)
        if fast_norm > 100.0:
            E_fast_new = E_fast_new * (100.0 / fast_norm)
        
        self.E_fast = E_fast_new
        self.E_slow = E_slow_new
        
        return E_fast_new, E_slow_new
    
    def run(self, num_steps: int, record_interval: int = 100) -> Dict:
        """
        运行指定步数
        
        Returns:
            包含轨迹和统计信息的字典
        """
        trajectory = []
        slow_trajectory = []
        gain_trajectory = []
        
        for step in range(num_steps):
            E_f, E_s = self.step()
            
            if step % record_interval == 0:
                trajectory.append(E_f.copy())
                slow_trajectory.append(E_s.copy())
                gain_trajectory.append(self.current_gain)
        
        return {
            'fast_trajectory': np.array(trajectory),
            'slow_trajectory': np.array(slow_trajectory),
            'gain_trajectory': np.array(gain_trajectory),
            'final_fast': E_f.copy(),
            'final_slow': E_s.copy(),
            'num_steps': num_steps
        }


def calculate_lyapunov_exponent(
    params: TuningParams,
    fast_dim: int = 64,
    num_steps: int = 2000,
    perturbation: float = 1e-6,
    transient_steps: int = 500
) -> float:
    """
    计算最大Lyapunov指数
    
    使用两个独立系统（参考+扰动），用切线近似法计算。
    """
    # 创建两个系统
    sys_ref = SimplifiedDynamics(fast_dim=fast_dim, params=params, seed=42)
    sys_pert = SimplifiedDynamics(fast_dim=fast_dim, params=params, seed=42)
    
    # 先运行瞬态期
    for _ in range(transient_steps):
        sys_ref.step()
        sys_pert.step()
    
    # 添加微小扰动
    sys_pert.E_fast = sys_ref.E_fast + perturbation * np.random.RandomState(123).randn(fast_dim)
    delta_0 = np.linalg.norm(sys_pert.E_fast - sys_ref.E_fast)
    
    if delta_0 < 1e-12:
        return 0.0
    
    lyapunov_sum = 0.0
    num_measurements = 0
    renorm_interval = 50
    
    for step in range(num_steps):
        sys_ref.step()
        sys_pert.step()
        
        if step % renorm_interval == 0 and step > 0:
            delta_t = np.linalg.norm(sys_pert.E_fast - sys_ref.E_fast)
            
            if delta_t > 1e-12 and delta_0 > 1e-12:
                lyapunov_sum += np.log(delta_t / delta_0)
                num_measurements += 1
                
                # 重归一化扰动轨迹
                scale = delta_0 / delta_t
                sys_pert.E_fast = sys_ref.E_fast + (sys_pert.E_fast - sys_ref.E_fast) * scale
    
    if num_measurements > 0:
        dt = sys_ref.dt
        lyapunov = lyapunov_sum / (num_measurements * renorm_interval * dt)
        return lyapunov
    
    return 0.0


def calculate_drift_rate(
    params: TuningParams,
    fast_dim: int = 64,
    num_steps: int = 5000
) -> float:
    """
    计算慢变量漂移率
    """
    sys = SimplifiedDynamics(fast_dim=fast_dim, params=params, seed=42)
    
    # 初始范数
    initial_norm = np.linalg.norm(sys.E_slow)
    
    # 运行
    for _ in range(num_steps):
        sys.step()
    
    # 最终范数
    final_norm = np.linalg.norm(sys.E_slow)
    
    # 漂移率（相对变化率/单位时间）
    if initial_norm > 1e-10:
        drift_rate = abs(final_norm - initial_norm) / initial_norm / (num_steps * sys.dt)
    else:
        drift_rate = 0.0
    
    return drift_rate


def calculate_autocorrelation(
    params: TuningParams,
    fast_dim: int = 64,
    num_steps: int = 5000,
    lag: int = 100
) -> float:
    """
    计算快变量自相关系数（表征DMN稳定性）
    """
    sys = SimplifiedDynamics(fast_dim=fast_dim, params=params, seed=42)
    
    # 收集轨迹
    trajectory = []
    for step in range(num_steps):
        sys.step()
        if step % 10 == 0:
            trajectory.append(sys.E_fast.copy())
    
    trajectory = np.array(trajectory)
    
    # 计算平均自相关
    if len(trajectory) > lag:
        x = trajectory[:-lag].flatten()
        y = trajectory[lag:].flatten()
        
        x_mean = x.mean()
        y_mean = y.mean()
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
        
        if denominator > 1e-10:
            return numerator / denominator
    
    return 0.0


def evaluate_params(params: TuningParams, fast_dim: int = 64) -> TuningResult:
    """
    评估一组参数
    """
    result = TuningResult(params=params)
    
    # 1. 计算Lyapunov指数
    try:
        result.lyapunov_lambda = calculate_lyapunov_exponent(
            params, fast_dim=fast_dim
        )
    except Exception as e:
        logger.warning(f"Lyapunov计算失败: {e}")
        result.lyapunov_lambda = -999
    
    # 2. 计算漂移率
    try:
        result.drift_rate = calculate_drift_rate(
            params, fast_dim=fast_dim
        )
    except Exception as e:
        logger.warning(f"漂移率计算失败: {e}")
        result.drift_rate = 999
    
    # 3. 计算自相关
    try:
        result.autocorrelation = calculate_autocorrelation(
            params, fast_dim=fast_dim
        )
    except Exception as e:
        logger.warning(f"自相关计算失败: {e}")
        result.autocorrelation = 0.0
    
    # 4. 计算得分
    # Lyapunov得分：在(0, 0.1)区间内满分
    if 0 < result.lyapunov_lambda < 0.1:
        lyapunov_score = 1.0
    elif result.lyapunov_lambda <= 0:
        lyapunov_score = max(0.0, 1.0 + result.lyapunov_lambda * 5)  # λ=-0.2时为0
    else:
        lyapunov_score = max(0.0, 1.0 - (result.lyapunov_lambda - 0.1) * 10)  # λ=0.2时为0
    
    # 漂移率得分：< 0.05满分
    drift_score = max(0.0, 1.0 - result.drift_rate * 10)
    drift_score = min(1.0, drift_score)
    
    # 自相关得分：> 0.3满分
    autocorr_score = max(0.0, min(1.0, (result.autocorrelation - 0.1) / 0.2))
    
    # 总体得分（加权）
    result.overall_score = 0.5 * lyapunov_score + 0.25 * drift_score + 0.25 * autocorr_score
    
    # 通过判定
    result.passed = (
        0 < result.lyapunov_lambda < 0.1 and
        result.drift_rate < 0.05 and
        result.autocorrelation > 0.3
    )
    
    result.metrics = {
        'lyapunov_score': lyapunov_score,
        'drift_score': drift_score,
        'autocorr_score': autocorr_score
    }
    
    return result


def grid_search_tuning(
    fast_dim: int = 64,
    output_dir: str = "fast_tuning_results"
) -> Dict:
    """
    网格搜索参数调优
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 参数搜索空间
    base_gain_range = [0.05, 0.075, 0.1, 0.15, 0.2, 0.25]
    min_gain_range = [0.02, 0.05, 0.08, 0.1]
    coupling_range = [0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    
    best_result = None
    all_results = []
    
    total_combos = len(base_gain_range) * len(min_gain_range) * len(coupling_range)
    logger.info(f"开始网格搜索，共 {total_combos} 组参数...")
    
    start_time = time.time()
    combo_idx = 0
    
    for base_gain in base_gain_range:
        for min_gain in min_gain_range:
            if min_gain > base_gain:
                continue
                
            for coupling in coupling_range:
                combo_idx += 1
                
                params = TuningParams(
                    base_gain=base_gain,
                    min_gain=min_gain,
                    coupling_strength=coupling
                )
                
                result = evaluate_params(params, fast_dim=fast_dim)
                all_results.append(result)
                
                if best_result is None or result.overall_score > best_result.overall_score:
                    best_result = result
                    logger.info(
                        f"[{combo_idx}/{total_combos}] 新最佳! "
                        f"score={result.overall_score:.4f}, "
                        f"λ={result.lyapunov_lambda:.6f}, "
                        f"drift={result.drift_rate:.6f}, "
                        f"autocorr={result.autocorrelation:.4f}, "
                        f"params: base_gain={base_gain:.3f}, "
                        f"min_gain={min_gain:.3f}, "
                        f"coupling={coupling:.3f}"
                    )
                
                if combo_idx % 10 == 0:
                    elapsed = time.time() - start_time
                    speed = combo_idx / elapsed
                    remaining = (total_combos - combo_idx) / speed
                    logger.info(
                        f"进度: {combo_idx}/{total_combos} ({100*combo_idx/total_combos:.1f}%), "
                        f"速度: {speed:.2f} 组/秒, 预计剩余: {remaining:.1f}秒"
                    )
    
    total_time = time.time() - start_time
    logger.info(f"网格搜索完成! 总耗时: {total_time:.2f}秒")
    
    # 保存结果
    results_data = {
        'best_params': {
            'base_gain': best_result.params.base_gain,
            'min_gain': best_result.params.min_gain,
            'coupling_strength': best_result.params.coupling_strength,
            'decay_rate': best_result.params.decay_rate,
            'noise_scale': best_result.params.noise_scale,
            'attractor_gain': best_result.params.attractor_gain
        },
        'best_metrics': {
            'lyapunov_lambda': best_result.lyapunov_lambda,
            'drift_rate': best_result.drift_rate,
            'autocorrelation': best_result.autocorrelation,
            'overall_score': best_result.overall_score,
            'passed': best_result.passed
        },
        'all_results': [
            {
                'params': {
                    'base_gain': r.params.base_gain,
                    'min_gain': r.params.min_gain,
                    'coupling_strength': r.params.coupling_strength
                },
                'lyapunov_lambda': r.lyapunov_lambda,
                'drift_rate': r.drift_rate,
                'autocorrelation': r.autocorrelation,
                'overall_score': r.overall_score,
                'passed': r.passed
            }
            for r in sorted(all_results, key=lambda x: -x.overall_score)[:20]
        ],
        'search_space': {
            'base_gain_range': base_gain_range,
            'min_gain_range': min_gain_range,
            'coupling_range': coupling_range
        },
        'fast_dim': fast_dim,
        'total_time_seconds': total_time,
        'total_combos': total_combos
    }
    
    with open(output_path / 'tuning_results.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, default=float)
    
    # 生成报告
    report = f"""# 轻量级快速参数调优报告

## 概述

- **调优方法**: 网格搜索（简化动力学模型）
- **快变量维度**: {fast_dim}
- **总参数组合数**: {total_combos}
- **总耗时**: {total_time:.2f}秒
- **搜索速度**: {total_combos/total_time:.2f} 组/秒

## 最佳参数

| 参数 | 值 |
|------|-----|
| base_gain | {best_result.params.base_gain} |
| min_gain | {best_result.params.min_gain} |
| coupling_strength | {best_result.params.coupling_strength} |

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

| 排名 | base_gain | min_gain | coupling | λ | drift | autocorr | 得分 |
|------|-----------|----------|----------|---|-------|----------|------|
"""
    
    top10 = sorted(all_results, key=lambda x: -x.overall_score)[:10]
    for i, r in enumerate(top10, 1):
        status = '✓' if r.passed else '✗'
        report += (
            f"| {i} | {r.params.base_gain:.3f} | {r.params.min_gain:.3f} | "
            f"{r.params.coupling_strength:.3f} | {r.lyapunov_lambda:.6f} | "
            f"{r.drift_rate:.6f} | {r.autocorrelation:.4f} | "
            f"{r.overall_score:.4f} {status} |\n"
        )
    
    report += """
## 说明

1. 本调优使用**简化动力学模型**（低维 + Euler法），结果仅供参考
2. 找到的参数范围需要在完整系统（2048维 + dopri5）中验证
3. 建议使用最佳参数的 ±20% 范围在完整系统中进行精细调优

## 下一步

1. 将最佳参数应用到完整系统配置
2. 运行完整验证确认效果
3. 在最佳参数附近进行精细调优
"""
    
    with open(output_path / 'tuning_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"结果已保存到 {output_path}")
    
    return results_data


def main():
    parser = argparse.ArgumentParser(description='轻量级快速参数调优')
    parser.add_argument(
        '--fast-dim', type=int, default=64,
        help='快变量维度（默认64）'
    )
    parser.add_argument(
        '--output', type=str, default='fast_tuning_results',
        help='输出目录（默认fast_tuning_results）'
    )
    parser.add_argument(
        '--target-lyapunov', type=float, default=0.05,
        help='目标Lyapunov指数（默认0.05）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("=" * 80)
    logger.info("轻量级快速参数调优")
    logger.info("=" * 80)
    logger.info(f"快变量维度: {args.fast_dim}")
    logger.info(f"目标Lyapunov指数: {args.target_lyapunov}")
    logger.info(f"输出目录: {args.output}")
    logger.info("")
    
    # 执行网格搜索
    results = grid_search_tuning(
        fast_dim=args.fast_dim,
        output_dir=args.output
    )
    
    # 输出最佳结果
    best = results['best_metrics']
    best_p = results['best_params']
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("调优完成！最佳结果：")
    logger.info("=" * 80)
    logger.info(f"  base_gain = {best_p['base_gain']}")
    logger.info(f"  min_gain = {best_p['min_gain']}")
    logger.info(f"  coupling_strength = {best_p['coupling_strength']}")
    logger.info(f"  Lyapunov λ = {best['lyapunov_lambda']:.6f}")
    logger.info(f"  漂移率 = {best['drift_rate']:.6f}")
    logger.info(f"  自相关 = {best['autocorrelation']:.4f}")
    logger.info(f"  总体得分 = {best['overall_score']:.4f}")
    logger.info(f"  是否通过: {'是' if best['passed'] else '否'}")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"详细结果请查看: {args.output}/tuning_report.md")


if __name__ == '__main__':
    main()
