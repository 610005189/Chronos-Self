"""
分层验证脚本 - 方案A + 方案B
================================

分层策略：
1. 纯数学超高速验证（耦合Logistic映射格点，秒级完成）
2. 真实系统快速验证（256维+Euler法，分钟级）
3. 完整测试（可选）

方案A：调整快动力学参数（decay_rate, max_grad_norm）
方案B：增加显式阻尼项（damping_coeff）
"""

import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class TuningParams:
    base_gain: float = 0.1
    min_gain: float = 0.1
    coupling: float = 0.5
    decay_rate: float = 0.85
    max_grad_norm: float = 10.0
    damping_coeff: float = 0.0


@dataclass
class ValidationResult:
    stage: str
    params: Dict
    lyapunov_mean: float
    lyapunov_max: float
    lyapunov_min: float
    drift_rate: float
    alignment_error: float
    score: float
    passed: bool
    runtime_ms: float


class CoupledLogisticLattice:
    def __init__(self, dim=128, slow_dim=16, base_gain=0.1, min_gain=0.05, 
                 coupling_strength=0.5, decay_rate=0.85, damping_coeff=0.0, seed=42):
        self.rng = np.random.RandomState(seed)
        self.dim = dim
        self.slow_dim = slow_dim
        self.base_gain = base_gain
        self.min_gain = min_gain
        self.coupling_strength = coupling_strength
        self.decay_rate = decay_rate
        self.damping_coeff = damping_coeff
        
        self.r_base = 2.5 + 0.05 * self.rng.randn(dim)
        
        self.state = 0.5 + 0.1 * self.rng.randn(dim)
        self.slow_state = 0.5 + 0.1 * self.rng.randn(slow_dim)
        
        self.current_gain = base_gain
        
    def _apply_coupling(self):
        left = np.roll(self.state, 1)
        right = np.roll(self.state, -1)
        coupling = self.coupling_strength * 0.5 * (left + right - 2 * self.state)
        return coupling
    
    def iterate(self, n_steps=1):
        for _ in range(n_steps):
            slow_mod = np.mean(self.slow_state)
            
            r_eff = self.r_base + self.current_gain * 0.5 + 0.02 * np.tanh(slow_mod)
            
            coupling = self._apply_coupling()
            
            x_new = r_eff * self.state * (1 - self.state) + coupling
            
            total_decay = 1.0 - self.decay_rate * 0.05 - self.damping_coeff * 0.05
            total_decay = max(0.95, min(1.0, total_decay))
            x_new = x_new * total_decay
            
            self.state = np.clip(x_new, 1e-10, 1.0 - 1e-10)
            
            slow_drift = 0.0005 * (np.mean(self.state) - self.slow_state)
            self.slow_state = np.clip(self.slow_state + slow_drift, 0.1, 0.9)
            
            gain_drift = 0.001 * (0.5 - np.std(self.state))
            self.current_gain = max(self.min_gain, min(2.0 * self.base_gain, 
                                                      self.current_gain + gain_drift))
        
        return self.state.copy()
    
    def calculate_lyapunov_exponent(self, n_steps=500, n_perturbations=3):
        le_list = []
        
        for _ in range(n_perturbations):
            sys_ref = CoupledLogisticLattice(
                dim=self.dim, slow_dim=self.slow_dim,
                base_gain=self.base_gain, min_gain=self.min_gain,
                coupling_strength=self.coupling_strength,
                decay_rate=self.decay_rate,
                damping_coeff=self.damping_coeff,
                seed=self.rng.randint(0, 10000)
            )
            
            rng_pert = np.random.RandomState(self.rng.randint(0, 10000))
            pert = rng_pert.randn(*sys_ref.state.shape) * 1e-8
            sys_pert = CoupledLogisticLattice(
                dim=self.dim, slow_dim=self.slow_dim,
                base_gain=self.base_gain, min_gain=self.min_gain,
                coupling_strength=self.coupling_strength,
                decay_rate=self.decay_rate,
                damping_coeff=self.damping_coeff,
                seed=self.rng.randint(0, 10000)
            )
            sys_pert.state = sys_ref.state + pert
            
            log_sum = 0.0
            valid_count = 0
            
            for _ in range(n_steps):
                sys_ref.iterate(1)
                sys_pert.iterate(1)
                
                diff = sys_pert.state - sys_ref.state
                norm = np.linalg.norm(diff)
                
                if norm > 1e-4:
                    sys_pert.state = sys_ref.state + (diff / norm) * 1e-8
                    log_sum += np.log(norm / 1e-8)
                    valid_count += 1
            
            if valid_count > 0:
                le_list.append(log_sum / valid_count)
        
        if le_list:
            return np.mean(le_list), np.max(le_list), np.min(le_list)
        return 0.0, 0.0, 0.0
    
    def calculate_drift_rate(self, n_steps=500):
        initial_mean = np.mean(self.state)
        for _ in range(n_steps):
            self.iterate(1)
        final_mean = np.mean(self.state)
        return abs(final_mean - initial_mean) / n_steps
    
    def calculate_alignment_error(self, n_steps=100):
        errors = []
        for _ in range(n_steps):
            prev_state = self.state.copy()
            self.iterate(1)
            errors.append(np.mean((self.state - prev_state) ** 2))
        return np.mean(errors)


def compute_score(lyapunov, drift_rate, alignment_error):
    if lyapunov <= 0:
        lyap_score = max(0.0, np.exp(-(lyapunov ** 2) / 0.005))
    elif lyapunov >= 0.1:
        lyap_score = max(0.0, np.exp(-((lyapunov - 0.1) ** 2) / 0.005))
    else:
        lyap_score = np.exp(-((lyapunov - 0.05) ** 2) / 0.0025)
    
    drift_score = max(0.0, 1.0 - min(1.0, drift_rate * 100.0))
    align_score = max(0.0, 1.0 - min(1.0, alignment_error * 1000.0))
    
    return 0.6 * lyap_score + 0.2 * drift_score + 0.2 * align_score


def run_math_validation(params: TuningParams, seed=42) -> ValidationResult:
    start = time.time()
    
    lattice = CoupledLogisticLattice(
        dim=128,
        slow_dim=16,
        base_gain=params.base_gain,
        min_gain=params.min_gain,
        coupling_strength=params.coupling,
        decay_rate=params.decay_rate,
        damping_coeff=params.damping_coeff,
        seed=seed,
    )
    
    lattice.iterate(200)
    
    lyap_mean, lyap_max, lyap_min = lattice.calculate_lyapunov_exponent(n_steps=300)
    drift_rate = lattice.calculate_drift_rate(n_steps=200)
    alignment_error = lattice.calculate_alignment_error(n_steps=100)
    score = compute_score(lyap_mean, drift_rate, alignment_error)
    
    passed = (0 < lyap_mean < 0.1) and (drift_rate < 0.001) and (alignment_error < 0.001)
    
    runtime_ms = (time.time() - start) * 1000
    
    return ValidationResult(
        stage="math",
        params=asdict(params),
        lyapunov_mean=lyap_mean,
        lyapunov_max=lyap_max,
        lyapunov_min=lyap_min,
        drift_rate=drift_rate,
        alignment_error=alignment_error,
        score=score,
        passed=passed,
        runtime_ms=runtime_ms,
    )


def run_fast_real_validation(params: TuningParams, fast_dim=128, seed=42) -> ValidationResult:
    start = time.time()
    
    try:
        from chronos_core.utils.config import ChronosConfig, DimensionalityConfig, \
            ChaosInjectionConfig, CouplingStabilityConfig, NeuralODEConfig, NumericsConfig
        from chronos_core.core.state import SelfState
        from chronos_core.core.integration_engine import IntegrationEngine, IntegrationEngineConfig
        from chronos_core.validation.p0_validation import P0Validation, P0ValidationConfig
        import torch
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        slow_dim = fast_dim // 4
        
        config = ChronosConfig()
        config.dim = DimensionalityConfig(
            fast_variable_dim=fast_dim,
            slow_variable_dim=slow_dim,
            core_subspace_dim=min(64, fast_dim // 4),
            semantic_dim=min(256, fast_dim // 2),
            physical_dim=min(256, fast_dim // 2),
            fusion_dim=min(512, fast_dim),
        )
        
        config.neural_ode = NeuralODEConfig(
            integration_method="rk4",
            atol=1e-4,
            rtol=1e-3,
            max_steps=100,
            dt=0.01,
        )
        
        config.numerics = NumericsConfig(
            solver_type="euler",
            spectral_norm_enabled=True,
            attention_mode="linear",
            checkpointing_enabled=False,
            fourier_enabled=False,
        )
        
        config.chaos_injection = ChaosInjectionConfig(
            base_gain=params.base_gain,
            min_gain=params.min_gain,
            chaos_injection_gain=params.base_gain,
            attractor_switch_interval=2000,
            attractor_noise_scale=0.01,
            gain_smoothing=0.95,
        )
        
        config.coupling_stability = CouplingStabilityConfig(
            coupling_adaptation_coeff=params.coupling,
            elastic_restoration_coeff=0.05,
            l2_perturbation_noise=0.05,
            anti_quietus_weight=0.1,
            inertia_weight=0.05,
            coupling_upper_bound=10.0,
            stability_threshold=1000.0,
            lyapunov_threshold=0.1,
        )
        
        config.device = "cpu"
        config.use_amp = False
        config.validation.p0_open_loop_hours = 0.005
        config.validation.lyapunov_window = 200
        config.validation.alignment_num_steps = [10, 30]
        
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
        
        if hasattr(engine, 'fast_dynamics') and engine.fast_dynamics:
            dyn_fn = engine.fast_dynamics.dynamics_fn
            if dyn_fn:
                if hasattr(dyn_fn, 'decay_layer') and hasattr(dyn_fn.decay_layer, 'weight_orig'):
                    torch.nn.init.constant_(dyn_fn.decay_layer.weight_orig, -params.decay_rate)
                if hasattr(dyn_fn.config, 'max_gradient_norm'):
                    dyn_fn.config.max_gradient_norm = params.max_grad_norm
                if not hasattr(dyn_fn.config, 'damping_coeff'):
                    setattr(dyn_fn.config, 'damping_coeff', params.damping_coeff)
                else:
                    dyn_fn.config.damping_coeff = params.damping_coeff
        
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
            E_fast=torch.randn(fast_dim) * 0.1,
            E_slow=torch.randn(slow_dim) * 0.1,
            timestamp=0.0,
        )
        
        result = validator.run_full_validation(initial_state, verbose=False)
        
        lyap_mean = result.lyapunov_mean
        lyap_max = result.lyapunov_max
        lyap_min = result.lyapunov_min
        drift_rate = result.drift_rate
        alignment_error = result.alignment_max_error
        score = result.overall_score
        passed = result.is_passed
        
    except Exception as e:
        print(f"  ✗ Real system validation error: {str(e)}")
        lyap_mean = 999.0
        lyap_max = 999.0
        lyap_min = 999.0
        drift_rate = 999.0
        alignment_error = 999.0
        score = 0.0
        passed = False
    
    runtime_ms = (time.time() - start) * 1000
    
    return ValidationResult(
        stage="real_fast",
        params=asdict(params),
        lyapunov_mean=lyap_mean,
        lyapunov_max=lyap_max,
        lyapunov_min=lyap_min,
        drift_rate=drift_rate,
        alignment_error=alignment_error,
        score=score,
        passed=passed,
        runtime_ms=runtime_ms,
    )


def generate_param_grid() -> List[TuningParams]:
    params_list = []
    
    base_gains = [0.05, 0.1, 0.2, 0.3]
    min_gains = [0.02, 0.05, 0.1]
    couplings = [0.2, 0.4, 0.6]
    decay_rates = [0.5, 1.0, 2.0, 4.0]
    grad_norms = [1.0, 3.0, 5.0, 10.0]
    dampings = [0.0, 0.5, 1.0, 2.0]
    
    for bg in base_gains:
        for mg in min_gains:
            if mg > bg:
                continue
            for c in couplings:
                for dr in decay_rates:
                    for gn in grad_norms:
                        for dm in dampings:
                            params_list.append(TuningParams(
                                base_gain=bg,
                                min_gain=mg,
                                coupling=c,
                                decay_rate=dr,
                                max_grad_norm=gn,
                                damping_coeff=dm,
                            ))
    
    return params_list


def main():
    parser = argparse.ArgumentParser(description="分层验证脚本 - 方案A + 方案B")
    parser.add_argument('--mode', default='both', choices=['math', 'real', 'both'],
                        help='验证模式：纯数学 / 真实系统 / 两者都运行')
    parser.add_argument('--fast-dim', type=int, default=128, help='真实系统快速验证维度')
    parser.add_argument('--output', default='layered_validation_results', help='输出目录')
    parser.add_argument('--top-n', type=int, default=5, help='数学验证后保留前N个参数进行真实系统验证')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"分层验证脚本 - 方案A + 方案B")
    print(f"模式: {args.mode}")
    print(f"真实系统维度: {args.fast_dim}")
    print(f"{'='*70}")
    
    all_params = generate_param_grid()
    print(f"\n总参数组合数: {len(all_params)}")
    
    math_results = []
    
    if args.mode in ['math', 'both']:
        print(f"\n{'='*50}")
        print("阶段1: 纯数学超高速验证")
        print(f"{'='*50}")
        
        start_total = time.time()
        
        for i, params in enumerate(all_params):
            result = run_math_validation(params)
            
            math_results.append({
                'params': params,
                'lyapunov': result.lyapunov_mean,
                'score': result.score,
                'passed': result.passed,
                'runtime': result.runtime_ms,
            })
            
            if (i + 1) % 20 == 0 or i == len(all_params) - 1:
                elapsed = time.time() - start_total
                print(f"  [{i+1}/{len(all_params)}] 已完成 {elapsed:.2f}s, "
                      f"平均 {elapsed/(i+1)*1000:.1f}ms/组")
        
        math_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n数学验证 Top {args.top_n} 结果:")
        for i, r in enumerate(math_results[:args.top_n]):
            p = r['params']
            print(f"  [{i+1}] λ={r['lyapunov']:.4f}, score={r['score']:.4f}")
            print(f"      bg={p.base_gain:.3f}, mg={p.min_gain:.3f}, c={p.coupling:.3f}")
            print(f"      decay={p.decay_rate:.3f}, grad={p.max_grad_norm:.3f}, damp={p.damping_coeff:.3f}")
        
        math_total_time = time.time() - start_total
        print(f"\n数学验证总耗时: {math_total_time:.2f}s ({math_total_time/len(all_params)*1000:.1f}ms/组)")
        
        with open(output_dir / 'math_results.json', 'w', encoding='utf-8') as f:
            json.dump([{
                'params': asdict(r['params']),
                'lyapunov': r['lyapunov'],
                'score': r['score'],
                'passed': r['passed'],
            } for r in math_results], f, indent=2, ensure_ascii=False)
    
    if args.mode in ['real', 'both']:
        print(f"\n{'='*50}")
        print(f"阶段2: 真实系统快速验证 (Top {args.top_n})")
        print(f"{'='*50}")
        
        if not math_results:
            top_params = all_params[:args.top_n]
        else:
            top_params = [r['params'] for r in math_results[:args.top_n]]
        
        real_results = []
        
        for i, params in enumerate(top_params):
            print(f"\n  [{i+1}/{len(top_params)}] 测试参数:")
            print(f"      bg={params.base_gain:.3f}, mg={params.min_gain:.3f}, c={params.coupling:.3f}")
            print(f"      decay={params.decay_rate:.3f}, grad={params.max_grad_norm:.3f}, damp={params.damping_coeff:.3f}")
            
            result = run_fast_real_validation(params, fast_dim=args.fast_dim)
            
            print(f"      ✓ λ={result.lyapunov_mean:.4f}, drift={result.drift_rate:.6f}, align={result.alignment_error:.6f}")
            print(f"      ✓ score={result.score:.4f}, passed={result.passed}, time={result.runtime_ms/1000:.1f}s")
            
            real_results.append({
                'params': params,
                'lyapunov': result.lyapunov_mean,
                'drift_rate': result.drift_rate,
                'alignment_error': result.alignment_error,
                'score': result.score,
                'passed': result.passed,
                'runtime': result.runtime_ms,
            })
        
        real_results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n真实系统验证 Top 结果:")
        for i, r in enumerate(real_results):
            p = r['params']
            print(f"  [{i+1}] λ={r['lyapunov']:.4f}, score={r['score']:.4f}, passed={r['passed']}")
            print(f"      bg={p.base_gain:.3f}, mg={p.min_gain:.3f}, c={p.coupling:.3f}")
            print(f"      decay={p.decay_rate:.3f}, grad={p.max_grad_norm:.3f}, damp={p.damping_coeff:.3f}")
        
        with open(output_dir / 'real_fast_results.json', 'w', encoding='utf-8') as f:
            json.dump([{
                'params': asdict(r['params']),
                'lyapunov': r['lyapunov'],
                'drift_rate': r['drift_rate'],
                'alignment_error': r['alignment_error'],
                'score': r['score'],
                'passed': r['passed'],
            } for r in real_results], f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"分层验证完成!")
    print(f"结果保存到: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
