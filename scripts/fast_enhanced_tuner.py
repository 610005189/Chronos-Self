"""
快速增强版自动化精细调优工作流 - 方案A + 方案B
==============================================

专注于核心参数：
- 方案A：decay_rate (0.5-4.0), max_grad_norm (1.0-10.0)
- 方案B：damping_coeff (0.0-2.0)

使用方法：
    python scripts/fast_enhanced_tuner.py
"""

import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime

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
from chronos_core.core.fast_dynamics import FastDynamicsConfig


@dataclass
class TuningParams:
    base_gain: float = 0.1
    min_gain: float = 0.1
    coupling: float = 0.5
    
    decay_rate: float = 0.85
    max_gradient_norm: float = 10.0
    damping_coeff: float = 0.0


@dataclass
class TuningResult:
    round_num: int
    round_name: str
    fast_dim: int
    slow_dim: int
    params: Dict
    lyapunov_mean: float
    lyapunov_max: float
    lyapunov_min: float
    drift_rate: float
    alignment_max_error: float
    alignment_avg_error: float
    score: float
    passed: bool
    validation_time: float
    error: Optional[str] = None


def create_config(
    fast_dim: int,
    slow_dim: int,
    params: TuningParams,
    lyapunov_window: int = 300,
    open_loop_hours: float = 0.005,
) -> ChronosConfig:
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
        imex_update_interval=100,
        imex_dt_safety_factor=0.9,
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
    
    config.validation.p0_open_loop_hours = open_loop_hours
    config.validation.lyapunov_window = lyapunov_window
    config.validation.alignment_num_steps = [10, 30]
    
    return config


def apply_enhanced_params(
    engine: IntegrationEngine,
    params: TuningParams,
) -> None:
    if hasattr(engine, 'fast_dynamics') and engine.fast_dynamics:
        fast_sys = engine.fast_dynamics
        if hasattr(fast_sys, 'dynamics_fn') and fast_sys.dynamics_fn:
            dyn_fn = fast_sys.dynamics_fn
            
            dyn_fn.config.decay_rate = params.decay_rate
            
            dyn_fn.config.max_gradient_norm = params.max_gradient_norm
            
            dyn_fn.config.damping_coeff = params.damping_coeff
            
            if hasattr(dyn_fn, 'decay_layer'):
                if hasattr(dyn_fn.decay_layer, 'weight_u'):
                    dyn_fn.decay_layer = nn.utils.remove_spectral_norm(dyn_fn.decay_layer)
                
                nn.init.constant_(dyn_fn.decay_layer.weight, -params.decay_rate)
                dyn_fn.decay_layer.bias = None


def run_validation(
    config: ChronosConfig,
    fast_dim: int,
    params: TuningParams,
    seed: int = 42,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    
    apply_enhanced_params(engine, params)
    
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
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
        timestamp=0.0,
    )
    
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


def compute_comprehensive_score(
    lyapunov: float,
    drift_rate: float,
    alignment_error: float,
    target_center: float = 0.05,
    target_range: Tuple[float, float] = (0.0, 0.1),
) -> float:
    lyap_low, lyap_high = target_range
    lyap_width = lyap_high - lyap_low
    
    if lyapunov <= lyap_low:
        lyap_score = max(0.0, np.exp(-((lyapunov - lyap_low) ** 2) / (2 * (lyap_width * 0.3) ** 2)))
    elif lyapunov >= lyap_high:
        lyap_score = max(0.0, np.exp(-((lyapunov - lyap_high) ** 2) / (2 * (lyap_width * 0.3) ** 2)))
    else:
        lyap_score = np.exp(-((lyapunov - target_center) ** 2) / (2 * (lyap_width * 0.3) ** 2))
    
    lyap_score = float(min(1.0, max(0.0, lyap_score)))
    
    drift_penalty = min(1.0, drift_rate * 5.0)
    drift_score = max(0.0, 1.0 - drift_penalty)
    
    align_penalty = min(1.0, alignment_error * 10.0)
    align_score = max(0.0, 1.0 - align_penalty)
    
    total_score = 0.6 * lyap_score + 0.2 * drift_score + 0.2 * align_score
    
    return float(total_score)


def is_result_valid(result: Dict) -> bool:
    if result.get('error') is not None:
        return False
    lyap = result.get('lyapunov_mean', 0)
    if not np.isfinite(lyap):
        return False
    if lyap < -1.0 or lyap > 20.0:
        return False
    drift = result.get('drift_rate', 1.0)
    if not np.isfinite(drift) or drift > 10.0:
        return False
    align = result.get('alignment_max_error', 1.0)
    if not np.isfinite(align) or align > 10.0:
        return False
    return True


class FastEnhancedTuner:
    def __init__(
        self,
        output_dir: str = "fast_enhanced_results",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.history: List[TuningResult] = []
        self.round_best: List[Dict] = []
        self.start_time = 0.0
    
    def save_history(self):
        history_data = {
            'start_time': self.start_time,
            'total_elapsed_seconds': time.time() - self.start_time,
            'num_rounds': len(self.round_best),
            'round_best': self.round_best,
            'results': [asdict(r) for r in self.history],
        }
        with open(self.output_dir / 'tuning_history.json', 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
    
    def run_round(
        self,
        round_num: int,
        round_name: str,
        fast_dim: int,
        slow_dim: int,
        param_grid: List[TuningParams],
    ) -> Tuple[TuningParams, float]:
        print(f"\n{'='*60}")
        print(f"Round {round_num}: {round_name}")
        print(f"Fast dim: {fast_dim}, Slow dim: {slow_dim}")
        print(f"Params: {len(param_grid)} combinations")
        print(f"{'='*60}")
        
        best_score = -1.0
        best_params = None
        best_lyapunov = float('inf')
        
        for i, params in enumerate(param_grid):
            print(f"\n[{i+1}/{len(param_grid)}] Testing params:")
            print(f"  decay_rate={params.decay_rate:.4f}, max_grad_norm={params.max_gradient_norm:.4f}, damping={params.damping_coeff:.4f}")
            
            try:
                config = create_config(fast_dim, slow_dim, params)
                result = run_validation(config, fast_dim, params, seed=self.seed)
                
                if is_result_valid(result):
                    lyap = result['lyapunov_mean']
                    drift = result['drift_rate']
                    align = result['alignment_max_error']
                    score = compute_comprehensive_score(lyap, drift, align)
                    
                    print(f"  ✓ Lyapunov={lyap:.4f}, drift={drift:.4f}, align={align:.4f}, score={score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_params = params
                        best_lyapunov = lyap
                        print(f"  ✨ New best!")
                    
                    self.history.append(TuningResult(
                        round_num=round_num,
                        round_name=round_name,
                        fast_dim=fast_dim,
                        slow_dim=slow_dim,
                        params={
                            'base_gain': params.base_gain,
                            'min_gain': params.min_gain,
                            'coupling': params.coupling,
                            'decay_rate': params.decay_rate,
                            'max_gradient_norm': params.max_gradient_norm,
                            'damping_coeff': params.damping_coeff,
                        },
                        lyapunov_mean=lyap,
                        lyapunov_max=result['lyapunov_max'],
                        lyapunov_min=result['lyapunov_min'],
                        drift_rate=drift,
                        alignment_max_error=align,
                        alignment_avg_error=result['alignment_avg_error'],
                        score=score,
                        passed=result['passed'],
                        validation_time=result['validation_time'],
                    ))
                else:
                    print(f"  ✗ Invalid result")
                
                if (i + 1) % 5 == 0:
                    self.save_history()
            
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        
        self.save_history()
        
        if best_params is None:
            raise ValueError("No valid results found in this round")
        
        self.round_best.append({
            'round_num': round_num,
            'round_name': round_name,
            'fast_dim': fast_dim,
            'slow_dim': slow_dim,
            'params': {
                'base_gain': best_params.base_gain,
                'min_gain': best_params.min_gain,
                'coupling': best_params.coupling,
                'decay_rate': best_params.decay_rate,
                'max_gradient_norm': best_params.max_gradient_norm,
                'damping_coeff': best_params.damping_coeff,
            },
            'lyapunov_mean': best_lyapunov,
            'score': best_score,
        })
        
        print(f"\nRound {round_num} complete!")
        print(f"Best: decay_rate={best_params.decay_rate:.4f}, max_grad_norm={best_params.max_gradient_norm:.4f}, damping={best_params.damping_coeff:.4f}")
        print(f"      Lyapunov={best_lyapunov:.4f}, Score={best_score:.4f}")
        
        return best_params, best_lyapunov
    
    def run(self):
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print("Fast Enhanced Auto Tuner - 方案A + 方案B")
        print("  方案A: decay_rate (0.5-4.0), max_grad_norm (1.0-10.0)")
        print("  方案B: damping_coeff (0.0-2.0)")
        print(f"{'='*70}")
        
        decay_rates = np.linspace(0.5, 4.0, 6)
        grad_norms = np.linspace(1.0, 10.0, 5)
        dampings = np.linspace(0.0, 2.0, 4)
        
        round_1_params = []
        for dr in decay_rates:
            for gn in grad_norms:
                for dm in dampings:
                    round_1_params.append(TuningParams(
                        base_gain=0.1,
                        min_gain=0.05,
                        coupling=0.3,
                        decay_rate=float(dr),
                        max_gradient_norm=float(gn),
                        damping_coeff=float(dm),
                    ))
        
        print(f"\nRound 1 (粗搜 128D): {len(round_1_params)} parameter combinations")
        best_params, best_lyap = self.run_round(
            round_num=1,
            round_name="粗搜",
            fast_dim=128,
            slow_dim=32,
            param_grid=round_1_params,
        )
        
        if 0 < best_lyap < 0.1:
            print(f"\n🎉 Lyapunov进入目标区间 (0, 0.1)!")
            self.final_validation(best_params)
            return
        
        decay_rates_fine = np.linspace(
            max(0.5, best_params.decay_rate - 1.0),
            min(4.0, best_params.decay_rate + 1.0),
            5
        )
        grad_norms_fine = np.linspace(
            max(1.0, best_params.max_gradient_norm - 3.0),
            min(10.0, best_params.max_gradient_norm + 3.0),
            4
        )
        dampings_fine = np.linspace(
            max(0.0, best_params.damping_coeff - 0.8),
            min(2.0, best_params.damping_coeff + 0.8),
            4
        )
        
        round_2_params = []
        for dr in decay_rates_fine:
            for gn in grad_norms_fine:
                for dm in dampings_fine:
                    round_2_params.append(TuningParams(
                        base_gain=0.1,
                        min_gain=0.05,
                        coupling=0.3,
                        decay_rate=float(dr),
                        max_gradient_norm=float(gn),
                        damping_coeff=float(dm),
                    ))
        
        print(f"\nRound 2 (精搜 256D): {len(round_2_params)} parameter combinations")
        best_params, best_lyap = self.run_round(
            round_num=2,
            round_name="精搜",
            fast_dim=256,
            slow_dim=64,
            param_grid=round_2_params,
        )
        
        self.final_validation(best_params)
    
    def final_validation(self, best_params: TuningParams):
        print(f"\n{'='*70}")
        print("FINAL VALIDATION (256D)")
        print(f"{'='*70}")
        
        config = create_config(
            256,
            64,
            best_params,
            lyapunov_window=500,
            open_loop_hours=0.01,
        )
        result = run_validation(config, 256, best_params, seed=self.seed)
        
        print(f"\nFinal Results:")
        print(f"  Parameters:")
        print(f"    decay_rate = {best_params.decay_rate:.6f}")
        print(f"    max_grad_norm = {best_params.max_gradient_norm:.6f}")
        print(f"    damping_coeff = {best_params.damping_coeff:.6f}")
        print(f"  Metrics:")
        print(f"    Lyapunov λ (mean) = {result['lyapunov_mean']:.6f}")
        print(f"    Lyapunov λ (max) = {result['lyapunov_max']:.6f}")
        print(f"    Lyapunov λ (min) = {result['lyapunov_min']:.6f}")
        print(f"    Drift rate = {result['drift_rate']:.6f}")
        print(f"    Alignment max error = {result['alignment_max_error']:.6f}")
        print(f"    Overall score = {result['score']:.6f}")
        print(f"    Target reached = {'✅ YES' if 0 < result['lyapunov_mean'] < 0.1 else '❌ NO'}")
        
        best_result = {
            'parameters': {
                'decay_rate': best_params.decay_rate,
                'max_grad_norm': best_params.max_gradient_norm,
                'damping_coeff': best_params.damping_coeff,
            },
            'metrics': {
                'lyapunov_mean': result['lyapunov_mean'],
                'lyapunov_max': result['lyapunov_max'],
                'lyapunov_min': result['lyapunov_min'],
                'drift_rate': result['drift_rate'],
                'alignment_max_error': result['alignment_max_error'],
                'overall_score': result['score'],
            },
            'target': {
                'lyapunov_range': '(0, 0.1)',
                'reached': 0 < result['lyapunov_mean'] < 0.1,
            },
            'tuning_summary': {
                'total_rounds': len(self.round_best),
                'total_params_tested': len(self.history),
                'total_time_seconds': time.time() - self.start_time,
            },
        }
        
        with open(self.output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print("Fast Enhanced Tuning Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="Fast Enhanced Auto Tuner")
    parser.add_argument('--output', default='fast_enhanced_results', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    tuner = FastEnhancedTuner(
        output_dir=args.output,
        seed=args.seed,
    )
    tuner.run()


if __name__ == "__main__":
    main()
