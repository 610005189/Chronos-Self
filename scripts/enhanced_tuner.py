"""
增强版自动化精细调优工作流 - 方案A + 方案B
============================================

方案A：调整快动力学参数（decay_rate, max_gradient_norm）
方案B：增加显式阻尼项（damping_coeff）

核心思想：系统混沌主要来源于MLP演化函数的高Lipschitz常数，
需要通过线性耗散和梯度裁剪来控制扩张速率。

使用方法：
    python scripts/enhanced_tuner.py
    python scripts/enhanced_tuner.py --output enhanced_tuning_results
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

logger = logging.getLogger(__name__)


@dataclass
class TuningParams:
    base_gain: float = 0.1
    min_gain: float = 0.1
    coupling: float = 0.5
    
    # 方案A：调整快动力学系统参数
    decay_rate: float = 0.85        # 线性衰减率（越高耗散越强）
    max_gradient_norm: float = 10.0  # 梯度裁剪阈值（越低限制越强）
    
    # 方案B：增加显式阻尼项
    damping_coeff: float = 0.0      # 额外阻尼系数


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
    """
    方案A + 方案B：在引擎初始化后应用增强参数
    
    Args:
        engine: 积分引擎
        params: 调优参数
    """
    if hasattr(engine, 'fast_dynamics') and engine.fast_dynamics:
        fast_sys = engine.fast_dynamics
        if hasattr(fast_sys, 'dynamics_fn') and fast_sys.dynamics_fn:
            dyn_fn = fast_sys.dynamics_fn
            
            # 方案A：调整衰减率
            # 注意：当使用谱归一化时，需要先移除谱归一化再修改权重
            if hasattr(dyn_fn, 'decay_layer'):
                if hasattr(dyn_fn.decay_layer, 'weight_u'):
                    # 先移除谱归一化（否则修改 weight_orig 不会影响前向传播）
                    dyn_fn.decay_layer = nn.utils.remove_spectral_norm(dyn_fn.decay_layer)
                # 然后修改权重
                nn.init.constant_(dyn_fn.decay_layer.weight, -params.decay_rate)
                dyn_fn.decay_layer.bias = None
            
            # 方案A：调整梯度裁剪阈值
            if hasattr(dyn_fn.config, 'max_gradient_norm'):
                dyn_fn.config.max_gradient_norm = params.max_gradient_norm
            
            # 方案B：注入额外阻尼系数到配置
            if not hasattr(dyn_fn.config, 'damping_coeff'):
                setattr(dyn_fn.config, 'damping_coeff', params.damping_coeff)
            else:
                dyn_fn.config.damping_coeff = params.damping_coeff
            
            logger.info(f"Applied enhanced params: decay_rate={params.decay_rate}, "
                       f"max_grad_norm={params.max_gradient_norm}, damping_coeff={params.damping_coeff}")


def run_validation(
    config: ChronosConfig,
    fast_dim: int,
    params: TuningParams,
    seed: int = 42,
    timeout_seconds: float = 300.0,
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


def generate_param_grid(
    center: Optional[TuningParams],
    base_gain_range: Tuple[float, float],
    min_gain_range: Tuple[float, float],
    coupling_range: Tuple[float, float],
    decay_rate_range: Tuple[float, float],
    max_grad_norm_range: Tuple[float, float],
    damping_range: Tuple[float, float],
    n_base_gain: int = 3,
    n_min_gain: int = 3,
    n_coupling: int = 3,
    n_decay: int = 3,
    n_grad_norm: int = 3,
    n_damping: int = 3,
) -> List[TuningParams]:
    base_gains = np.linspace(base_gain_range[0], base_gain_range[1], n_base_gain)
    min_gains = np.linspace(min_gain_range[0], min_gain_range[1], n_min_gain)
    couplings = np.linspace(coupling_range[0], coupling_range[1], n_coupling)
    decay_rates = np.linspace(decay_rate_range[0], decay_rate_range[1], n_decay)
    grad_norms = np.linspace(max_grad_norm_range[0], max_grad_norm_range[1], n_grad_norm)
    dampings = np.linspace(damping_range[0], damping_range[1], n_damping)
    
    params_list = []
    for bg in base_gains:
        bg = float(bg)
        for mg in min_gains:
            mg = float(min(mg, bg * 0.99))
            mg = max(mg, 0.001)
            for c in couplings:
                c = float(c)
                for dr in decay_rates:
                    dr = float(dr)
                    for gn in grad_norms:
                        gn = float(gn)
                        for dm in dampings:
                            dm = float(dm)
                            params_list.append(TuningParams(
                                base_gain=bg,
                                min_gain=mg,
                                coupling=c,
                                decay_rate=dr,
                                max_gradient_norm=gn,
                                damping_coeff=dm,
                            ))
    
    return params_list


def generate_narrow_range(
    best_params: TuningParams,
    fraction: float = 0.3,
) -> Dict:
    def _range(center, frac, min_val=0.001, max_val=100.0):
        half_width = center * frac
        low = max(min_val, center - half_width)
        high = min(max_val, center + half_width)
        return (low, high)
    
    return {
        'base_gain_range': _range(best_params.base_gain, fraction),
        'min_gain_range': _range(best_params.min_gain, fraction),
        'coupling_range': _range(best_params.coupling, fraction),
        'decay_rate_range': _range(best_params.decay_rate, fraction, min_val=0.1, max_val=20.0),
        'max_grad_norm_range': _range(best_params.max_gradient_norm, fraction, min_val=0.5, max_val=20.0),
        'damping_range': _range(best_params.damping_coeff, fraction, min_val=0.0, max_val=10.0),
    }


class EnhancedAutoTuner:
    def __init__(
        self,
        output_dir: str = "enhanced_tuning_results",
        fast_dim_coarse: int = 128,
        slow_dim_coarse: int = 32,
        fast_dim_fine: int = 256,
        slow_dim_fine: int = 64,
        seed: int = 42,
        convergence_threshold: float = 0.02,
        max_rounds: int = 5,
        total_time_limit_minutes: float = 180.0,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fast_dim_coarse = fast_dim_coarse
        self.slow_dim_coarse = slow_dim_coarse
        self.fast_dim_fine = fast_dim_fine
        self.slow_dim_fine = slow_dim_fine
        self.seed = seed
        self.convergence_threshold = convergence_threshold
        self.max_rounds = max_rounds
        self.total_time_limit = total_time_limit_minutes * 60.0
        
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
            elapsed_total = time.time() - self.start_time
            if elapsed_total > self.total_time_limit:
                print(f"Time limit exceeded! ({elapsed_total:.1f}s > {self.total_time_limit:.1f}s)")
                break
            
            print(f"\n[{i+1}/{len(param_grid)}] Testing params:")
            print(f"  base_gain={params.base_gain:.4f}, min_gain={params.min_gain:.4f}")
            print(f"  coupling={params.coupling:.4f}, decay_rate={params.decay_rate:.4f}")
            print(f"  max_grad_norm={params.max_gradient_norm:.4f}, damping={params.damping_coeff:.4f}")
            
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
                        lyapunov_mean=0.0,
                        lyapunov_max=0.0,
                        lyapunov_min=0.0,
                        drift_rate=0.0,
                        alignment_max_error=0.0,
                        alignment_avg_error=0.0,
                        score=0.0,
                        passed=False,
                        validation_time=0.0,
                        error="invalid_result",
                    ))
                
                if (i + 1) % 5 == 0:
                    self.save_history()
                    print(f"  History saved ({len(self.history)} results)")
            
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
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
                    lyapunov_mean=0.0,
                    lyapunov_max=0.0,
                    lyapunov_min=0.0,
                    drift_rate=0.0,
                    alignment_max_error=0.0,
                    alignment_avg_error=0.0,
                    score=0.0,
                    passed=False,
                    validation_time=0.0,
                    error=str(e),
                ))
        
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
            'valid_count': sum(1 for r in self.history if r.round_num == round_num and r.error is None),
            'error_count': sum(1 for r in self.history if r.round_num == round_num and r.error is not None),
            'round_time_seconds': time.time() - self.start_time - sum(
                rb.get('round_time_seconds', 0) for rb in self.round_best[:-1]
            ),
        })
        
        print(f"\nRound {round_num} complete!")
        print(f"Best: base_gain={best_params.base_gain:.4f}, min_gain={best_params.min_gain:.4f}")
        print(f"      coupling={best_params.coupling:.4f}, decay_rate={best_params.decay_rate:.4f}")
        print(f"      max_grad_norm={best_params.max_gradient_norm:.4f}, damping={best_params.damping_coeff:.4f}")
        print(f"      Lyapunov={best_lyapunov:.4f}, Score={best_score:.4f}")
        
        return best_params, best_lyapunov
    
    def run(self):
        self.start_time = time.time()
        
        print(f"\n{'='*70}")
        print("Enhanced Auto Tuner - 方案A + 方案B")
        print("  方案A: 调整快动力学参数 (decay_rate, max_gradient_norm)")
        print("  方案B: 增加显式阻尼项 (damping_coeff)")
        print(f"{'='*70}")
        
        round_1_params = generate_param_grid(
            center=None,
            base_gain_range=(0.01, 0.3),
            min_gain_range=(0.005, 0.15),
            coupling_range=(0.1, 0.5),
            decay_rate_range=(0.5, 4.0),
            max_grad_norm_range=(1.0, 10.0),
            damping_range=(0.0, 2.0),
            n_base_gain=3,
            n_min_gain=2,
            n_coupling=2,
            n_decay=4,
            n_grad_norm=3,
            n_damping=3,
        )
        print(f"\nRound 1 (粗搜): {len(round_1_params)} parameter combinations")
        
        best_params, best_lyap = self.run_round(
            round_num=1,
            round_name="粗搜",
            fast_dim=self.fast_dim_coarse,
            slow_dim=self.slow_dim_coarse,
            param_grid=round_1_params,
        )
        
        if len(self.round_best) >= 2:
            prev_lyap = self.round_best[-2]['lyapunov_mean']
            if abs(best_lyap - prev_lyap) < self.convergence_threshold:
                print(f"\nConvergence detected! Lyapunov changed by {abs(best_lyap - prev_lyap):.4f}")
                print("Skipping further rounds, proceeding to final validation")
                self.final_validation(best_params)
                return
        
        narrow_range = generate_narrow_range(best_params, fraction=0.3)
        round_2_params = generate_param_grid(
            center=best_params,
            **narrow_range,
            n_base_gain=3,
            n_min_gain=3,
            n_coupling=3,
            n_decay=3,
            n_grad_norm=3,
            n_damping=3,
        )
        print(f"\nRound 2 (中搜): {len(round_2_params)} parameter combinations")
        
        best_params, best_lyap = self.run_round(
            round_num=2,
            round_name="中搜",
            fast_dim=self.fast_dim_coarse,
            slow_dim=self.slow_dim_coarse,
            param_grid=round_2_params,
        )
        
        if len(self.round_best) >= 2:
            prev_lyap = self.round_best[-2]['lyapunov_mean']
            if abs(best_lyap - prev_lyap) < self.convergence_threshold:
                print(f"\nConvergence detected! Lyapunov changed by {abs(best_lyap - prev_lyap):.4f}")
                print("Skipping further rounds, proceeding to final validation")
                self.final_validation(best_params)
                return
        
        narrow_range = generate_narrow_range(best_params, fraction=0.2)
        round_3_params = generate_param_grid(
            center=best_params,
            **narrow_range,
            n_base_gain=3,
            n_min_gain=3,
            n_coupling=3,
            n_decay=3,
            n_grad_norm=3,
            n_damping=3,
        )
        print(f"\nRound 3 (精搜): {len(round_3_params)} parameter combinations")
        
        best_params, best_lyap = self.run_round(
            round_num=3,
            round_name="精搜",
            fast_dim=self.fast_dim_fine,
            slow_dim=self.slow_dim_fine,
            param_grid=round_3_params,
        )
        
        self.final_validation(best_params)
    
    def final_validation(self, best_params: TuningParams):
        print(f"\n{'='*70}")
        print("FINAL VALIDATION (256D)")
        print(f"{'='*70}")
        
        config = create_config(
            self.fast_dim_fine,
            self.slow_dim_fine,
            best_params,
            lyapunov_window=500,
            open_loop_hours=0.01,
        )
        result = run_validation(config, self.fast_dim_fine, best_params, seed=self.seed)
        
        print(f"\nFinal Results:")
        print(f"  Parameters:")
        print(f"    base_gain = {best_params.base_gain:.6f}")
        print(f"    min_gain = {best_params.min_gain:.6f}")
        print(f"    coupling = {best_params.coupling:.6f}")
        print(f"    decay_rate = {best_params.decay_rate:.6f}")
        print(f"    max_grad_norm = {best_params.max_gradient_norm:.6f}")
        print(f"    damping_coeff = {best_params.damping_coeff:.6f}")
        print(f"  Metrics:")
        print(f"    Lyapunov λ (mean) = {result['lyapunov_mean']:.6f}")
        print(f"    Lyapunov λ (max) = {result['lyapunov_max']:.6f}")
        print(f"    Lyapunov λ (min) = {result['lyapunov_min']:.6f}")
        print(f"    Drift rate = {result['drift_rate']:.6f}")
        print(f"    Alignment max error = {result['alignment_max_error']:.6f}")
        print(f"    Alignment avg error = {result['alignment_avg_error']:.6f}")
        print(f"    Overall score = {result['score']:.6f}")
        print(f"    P0 passed = {result['passed']}")
        print(f"    Open loop stable = {result['open_loop_stable']}")
        print(f"    Validation time = {result['validation_time']:.2f}s")
        
        best_result = {
            'parameters': {
                'base_gain': best_params.base_gain,
                'min_gain': best_params.min_gain,
                'coupling': best_params.coupling,
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
                'alignment_avg_error': result['alignment_avg_error'],
                'overall_score': result['score'],
                'p0_passed': result['passed'],
                'open_loop_stable': result['open_loop_stable'],
            },
            'target': {
                'lyapunov_range': '(0, 0.1)',
                'drift_rate_limit': '< 0.1',
                'alignment_error_limit': '< 0.05',
            },
            'tuning_summary': {
                'total_rounds': len(self.round_best),
                'total_params_tested': len(self.history),
                'total_time_seconds': time.time() - self.start_time,
            },
        }
        
        with open(self.output_dir / 'best_params.json', 'w', encoding='utf-8') as f:
            json.dump(best_result, f, indent=2, ensure_ascii=False)
        
        self.generate_report(best_result)
        
        print(f"\n{'='*70}")
        print("Enhanced Tuning Complete!")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*70}")
    
    def generate_report(self, best_result: Dict):
        report = f"""# 增强版自动化精细调优报告（方案A + 方案B）

## 概述

- **调优目标**: 找到使 Lyapunov 指数进入边缘混沌区间 (0, 0.1) 的参数组合
- **调优方法**: 多轮渐进式网格搜索（粗搜 → 中搜 → 精搜 → 最终验证）
- **方案A**: 调整快动力学参数（decay_rate, max_gradient_norm）
- **方案B**: 增加显式阻尼项（damping_coeff）
- **调优时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **总耗时**: {(time.time() - self.start_time) / 60:.2f} 分钟
- **总参数组合数**: {len(self.history)}
- **结果**: {'✅ 通过' if best_result['metrics']['p0_passed'] else '❌ 未通过'}

## 维度配置

| 阶段 | fast_dim | slow_dim |
|------|----------|----------|
| 粗搜/中搜 | {self.fast_dim_coarse} | {self.slow_dim_coarse} |
| 精搜/最终验证 | {self.fast_dim_fine} | {self.slow_dim_fine} |

## 搜索参数范围

| 参数 | 粗搜范围 | 说明 |
|------|----------|------|
| base_gain | (0.01, 0.3) | 混沌注入基础增益 |
| min_gain | (0.005, 0.15) | 混沌注入最小增益 |
| coupling | (0.1, 0.5) | 耦合自适应系数 |
| decay_rate | (0.5, 4.0) | **方案A**: 线性衰减率 |
| max_grad_norm | (1.0, 10.0) | **方案A**: 梯度裁剪阈值 |
| damping_coeff | (0.0, 2.0) | **方案B**: 额外阻尼系数 |

## 各轮搜索结果

"""
        
        for rb in self.round_best:
            report += f"""### 第{rb['round_num']}轮: {rb['round_name']}

| 指标 | 值 |
|------|-----|
| 维度 | {rb['fast_dim']}d |
| 有效结果数 | {rb['valid_count']} |
| 错误数 | {rb['error_count']} |
| 耗时 | {rb['round_time_seconds'] / 60:.2f} 分钟 |

**本轮最佳参数**:
- base_gain = {rb['params']['base_gain']:.4f}
- min_gain = {rb['params']['min_gain']:.4f}
- coupling = {rb['params']['coupling']:.4f}
- decay_rate = {rb['params']['decay_rate']:.4f}
- max_grad_norm = {rb['params']['max_gradient_norm']:.4f}
- damping_coeff = {rb['params']['damping_coeff']:.4f}

**对应指标**:
- Lyapunov λ = {rb['lyapunov_mean']:.4f}
- 综合得分 = {rb['score']:.4f}

"""
        
        report += f"""## 最佳参数

| 参数 | 值 |
|------|-----|
| base_gain | {best_result['parameters']['base_gain']:.6f} |
| min_gain | {best_result['parameters']['min_gain']:.6f} |
| coupling | {best_result['parameters']['coupling']:.6f} |
| decay_rate | {best_result['parameters']['decay_rate']:.6f} |
| max_grad_norm | {best_result['parameters']['max_grad_norm']:.6f} |
| damping_coeff | {best_result['parameters']['damping_coeff']:.6f} |

## 最终验证指标（{self.fast_dim_fine}维）

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| Lyapunov λ (均值) | {best_result['metrics']['lyapunov_mean']:.6f} | (0, 0.1) | {'✅ 通过' if 0 < best_result['metrics']['lyapunov_mean'] < 0.1 else '❌ 未达到'} |
| Lyapunov λ (最大) | {best_result['metrics']['lyapunov_max']:.6f} | - | - |
| Lyapunov λ (最小) | {best_result['metrics']['lyapunov_min']:.6f} | - | - |
| 漂移率 | {best_result['metrics']['drift_rate']:.6f} | < 0.1 | {'✅ 通过' if best_result['metrics']['drift_rate'] < 0.1 else '❌ 未达到'} |
| 对齐误差（最大） | {best_result['metrics']['alignment_max_error']:.6f} | < 0.05 | {'✅ 通过' if best_result['metrics']['alignment_max_error'] < 0.05 else '❌ 未达到'} |
| 对齐误差（平均） | {best_result['metrics']['alignment_avg_error']:.6f} | - | - |
| 综合得分 | {best_result['metrics']['overall_score']:.6f} | - | - |
| P0 验证通过 | {best_result['metrics']['p0_passed']} | True | {'✅' if best_result['metrics']['p0_passed'] else '❌'} |
| 开环稳定 | {best_result['metrics']['open_loop_stable']} | True | {'✅' if best_result['metrics']['open_loop_stable'] else '❌'} |

## 调优方法说明

### 方案A：调整快动力学参数
- **decay_rate**: 控制线性衰减层的强度，越高耗散越强，Lyapunov 越低
- **max_grad_norm**: 控制梯度裁剪阈值，越低对扩张性动力学的限制越强

### 方案B：增加显式阻尼项
- **damping_coeff**: 在动力学方程中添加 `-damping_coeff * y` 项，直接增加线性耗散

### 综合评分函数
- Lyapunov 指数（60%权重）：高斯型评分，在目标中心 0.05 处得分最高
- 漂移率（20%权重）：线性惩罚
- 对齐误差（20%权重）：线性惩罚

## 统计信息

| 指标 | 值 |
|------|-----|
| 总轮次 | {best_result['tuning_summary']['total_rounds']} |
| 总测试参数组合数 | {best_result['tuning_summary']['total_params_tested']} |
| 总耗时 | {best_result['tuning_summary']['total_time_seconds'] / 60:.2f} 分钟 |

## 文件清单

- `tuning_history.json` - 完整历史记录
- `best_params.json` - 最佳参数及指标
- `tuning_report.md` - 本报告
"""
        
        with open(self.output_dir / 'tuning_report.md', 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    parser = argparse.ArgumentParser(description="Enhanced Auto Tuner - 方案A + 方案B")
    parser.add_argument('--output', default='enhanced_tuning_results', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--time-limit', type=float, default=180, help='总时间限制（分钟）')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.WARNING)
    
    tuner = EnhancedAutoTuner(
        output_dir=args.output,
        seed=args.seed,
        total_time_limit_minutes=args.time_limit,
    )
    tuner.run()


if __name__ == "__main__":
    main()
