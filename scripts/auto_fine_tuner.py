"""
自动化精细调优工作流
======================

多轮网格搜索：宽网格 → 中网格 → 精细网格
每轮在上一轮最佳参数附近缩小搜索范围
收敛检测：连续2轮最佳 Lyapunov 变化 < 0.02 则停止

使用方法：
    python scripts/auto_fine_tuner.py
    python scripts/auto_fine_tuner.py --output real_system_tuning
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

logger = logging.getLogger(__name__)


@dataclass
class TuningParams:
    base_gain: float = 0.1
    min_gain: float = 0.1
    coupling: float = 0.5
    stability_threshold: float = 1000.0


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
        stability_threshold=params.stability_threshold,
        lyapunov_threshold=0.1,
    )
    
    config.device = "cpu"
    config.use_amp = False
    
    config.validation.p0_open_loop_hours = open_loop_hours
    config.validation.lyapunov_window = lyapunov_window
    config.validation.alignment_num_steps = [10, 30]
    
    return config


def run_validation(
    config: ChronosConfig,
    fast_dim: int,
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
    """
    综合评分函数
    
    - Lyapunov 接近目标中心 0.05 得分最高（高斯型）
    - 漂移率和对齐误差作为惩罚项
    
    得分范围: [0, 1]
    """
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
    if lyap < -1.0 or lyap > 10.0:
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
    n_base_gain: int = 5,
    n_min_gain: int = 3,
    n_coupling: int = 3,
) -> List[TuningParams]:
    base_gains = np.linspace(base_gain_range[0], base_gain_range[1], n_base_gain)
    min_gains = np.linspace(min_gain_range[0], min_gain_range[1], n_min_gain)
    couplings = np.linspace(coupling_range[0], coupling_range[1], n_coupling)
    
    params_list = []
    for bg in base_gains:
        bg = float(bg)
        for mg in min_gains:
            mg = float(min(mg, bg * 0.99))
            mg = max(mg, 0.001)
            for c in couplings:
                c = float(c)
                params_list.append(TuningParams(
                    base_gain=bg,
                    min_gain=mg,
                    coupling=c,
                ))
    
    return params_list


def generate_narrow_range(
    best_params: TuningParams,
    fraction: float = 0.3,
    n_points: int = 5,
) -> Dict:
    def _range(center, frac, min_val=0.001, max_val=10.0):
        half_width = center * frac
        low = max(min_val, center - half_width)
        high = min(max_val, center + half_width)
        return (low, high)
    
    return {
        'base_gain_range': _range(best_params.base_gain, fraction),
        'min_gain_range': _range(best_params.min_gain, fraction),
        'coupling_range': _range(best_params.coupling, fraction),
    }


class AutoFineTuner:
    def __init__(
        self,
        output_dir: str = "real_system_tuning",
        fast_dim_coarse: int = 128,
        slow_dim_coarse: int = 32,
        fast_dim_fine: int = 256,
        slow_dim_fine: int = 64,
        seed: int = 42,
        convergence_threshold: float = 0.02,
        max_rounds: int = 5,
        total_time_limit_minutes: float = 120.0,
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
        self.start_time = time.time()
        
        self.history_file = self.output_dir / "tuning_history.json"
        self.best_params_file = self.output_dir / "best_params.json"
        self.report_file = self.output_dir / "tuning_report.md"
    
    def _time_remaining(self) -> float:
        return self.total_time_limit - (time.time() - self.start_time)
    
    def _save_history(self):
        history_data = {
            'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
            'total_elapsed_seconds': time.time() - self.start_time,
            'num_rounds': len(self.round_best),
            'round_best': self.round_best,
            'results': [
                {
                    'round_num': r.round_num,
                    'round_name': r.round_name,
                    'fast_dim': r.fast_dim,
                    'slow_dim': r.slow_dim,
                    'params': r.params,
                    'lyapunov_mean': r.lyapunov_mean,
                    'lyapunov_max': r.lyapunov_max,
                    'lyapunov_min': r.lyapunov_min,
                    'drift_rate': r.drift_rate,
                    'alignment_max_error': r.alignment_max_error,
                    'alignment_avg_error': r.alignment_avg_error,
                    'score': r.score,
                    'passed': r.passed,
                    'validation_time': r.validation_time,
                    'error': r.error,
                }
                for r in self.history
            ],
        }
        
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, default=str)
    
    def _save_best(self, best_params: TuningParams, best_result: Dict):
        best_data = {
            'params': {
                'base_gain': best_params.base_gain,
                'min_gain': best_params.min_gain,
                'coupling': best_params.coupling,
            },
            'metrics': {
                'lyapunov_mean': best_result['lyapunov_mean'],
                'lyapunov_max': best_result['lyapunov_max'],
                'lyapunov_min': best_result['lyapunov_min'],
                'drift_rate': best_result['drift_rate'],
                'alignment_max_error': best_result['alignment_max_error'],
                'alignment_avg_error': best_result['alignment_avg_error'],
                'score': best_result['score'],
                'passed': best_result['passed'],
            },
            'fast_dim': best_result.get('fast_dim', self.fast_dim_fine),
            'slow_dim': best_result.get('slow_dim', self.slow_dim_fine),
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.best_params_file, 'w', encoding='utf-8') as f:
            json.dump(best_data, f, indent=2, default=str)
    
    def _has_converged(self) -> bool:
        if len(self.round_best) < 2:
            return False
        
        recent = self.round_best[-2:]
        lyap_diff = abs(recent[0]['lyapunov_mean'] - recent[1]['lyapunov_mean'])
        
        logger.info(f"收敛检测: |λ1 - λ2| = {lyap_diff:.6f}, 阈值 = {self.convergence_threshold}")
        
        return lyap_diff < self.convergence_threshold
    
    def run_round(
        self,
        round_num: int,
        round_name: str,
        params_list: List[TuningParams],
        fast_dim: int,
        slow_dim: int,
    ) -> Tuple[Optional[TuningParams], Optional[Dict]]:
        total = len(params_list)
        logger.info(f"{'='*80}")
        logger.info(f"第 {round_num} 轮: {round_name}")
        logger.info(f"维度: fast={fast_dim}, slow={slow_dim}")
        logger.info(f"参数组合数: {total}")
        logger.info(f"剩余时间: {self._time_remaining()/60:.1f} 分钟")
        logger.info(f"{'='*80}")
        
        best_score = -1.0
        best_params = None
        best_result = None
        valid_count = 0
        error_count = 0
        
        round_start = time.time()
        
        for i, params in enumerate(params_list, 1):
            if self._time_remaining() < 60:
                logger.warning(f"时间不足，提前终止本轮（已完成 {i}/{total}）")
                break
            
            logger.info(f"  [{i}/{total}] 测试: base_gain={params.base_gain:.4f}, "
                       f"min_gain={params.min_gain:.4f}, coupling={params.coupling:.4f}")
            
            try:
                config = create_config(fast_dim, slow_dim, params)
                result = run_validation(config, fast_dim, seed=self.seed)
                
                if not is_result_valid(result):
                    logger.warning(f"    结果无效，跳过")
                    error_count += 1
                    tuning_result = TuningResult(
                        round_num=round_num,
                        round_name=round_name,
                        fast_dim=fast_dim,
                        slow_dim=slow_dim,
                        params={
                            'base_gain': params.base_gain,
                            'min_gain': params.min_gain,
                            'coupling': params.coupling,
                        },
                        lyapunov_mean=float('nan'),
                        lyapunov_max=float('nan'),
                        lyapunov_min=float('nan'),
                        drift_rate=float('nan'),
                        alignment_max_error=float('nan'),
                        alignment_avg_error=float('nan'),
                        score=0.0,
                        passed=False,
                        validation_time=result.get('validation_time', 0),
                        error='invalid_result',
                    )
                    self.history.append(tuning_result)
                    continue
                
                score = compute_comprehensive_score(
                    result['lyapunov_mean'],
                    result['drift_rate'],
                    result['alignment_max_error'],
                )
                result['score'] = score
                valid_count += 1
                
                tuning_result = TuningResult(
                    round_num=round_num,
                    round_name=round_name,
                    fast_dim=fast_dim,
                    slow_dim=slow_dim,
                    params={
                        'base_gain': params.base_gain,
                        'min_gain': params.min_gain,
                        'coupling': params.coupling,
                    },
                    lyapunov_mean=result['lyapunov_mean'],
                    lyapunov_max=result['lyapunov_max'],
                    lyapunov_min=result['lyapunov_min'],
                    drift_rate=result['drift_rate'],
                    alignment_max_error=result['alignment_max_error'],
                    alignment_avg_error=result['alignment_avg_error'],
                    score=score,
                    passed=result['passed'],
                    validation_time=result['validation_time'],
                )
                self.history.append(tuning_result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
                    best_result['fast_dim'] = fast_dim
                    best_result['slow_dim'] = slow_dim
                    
                    logger.info(f"    ★ 新最佳! score={score:.4f}, "
                               f"λ={result['lyapunov_mean']:.6f}, "
                               f"drift={result['drift_rate']:.4f}, "
                               f"align={result['alignment_max_error']:.4f}, "
                               f"time={result['validation_time']:.1f}s")
                
            except Exception as e:
                    logger.error(f"    验证失败: {e}")
                    error_count += 1
                    tuning_result = TuningResult(
                        round_num=round_num,
                        round_name=round_name,
                        fast_dim=fast_dim,
                        slow_dim=slow_dim,
                        params={
                            'base_gain': params.base_gain,
                            'min_gain': params.min_gain,
                            'coupling': params.coupling,
                        },
                        lyapunov_mean=float('nan'),
                        lyapunov_max=float('nan'),
                        lyapunov_min=float('nan'),
                        drift_rate=float('nan'),
                        alignment_max_error=float('nan'),
                        alignment_avg_error=float('nan'),
                        score=0.0,
                        passed=False,
                        validation_time=0,
                        error=str(e),
                    )
                    self.history.append(tuning_result)
            
            if i % 5 == 0:
                self._save_history()
        
        round_time = time.time() - round_start
        
        if best_params is not None:
            self.round_best.append({
                'round_num': round_num,
                'round_name': round_name,
                'fast_dim': fast_dim,
                'slow_dim': slow_dim,
                'params': {
                    'base_gain': best_params.base_gain,
                    'min_gain': best_params.min_gain,
                    'coupling': best_params.coupling,
                },
                'lyapunov_mean': best_result['lyapunov_mean'],
                'score': best_score,
                'valid_count': valid_count,
                'error_count': error_count,
                'round_time_seconds': round_time,
            })
        else:
            logger.error(f"本轮没有找到有效参数！")
            self.round_best.append({
                'round_num': round_num,
                'round_name': round_name,
                'fast_dim': fast_dim,
                'slow_dim': slow_dim,
                'params': None,
                'lyapunov_mean': None,
                'score': 0,
                'valid_count': valid_count,
                'error_count': error_count,
                'round_time_seconds': round_time,
            })
        
        self._save_history()
        
        if best_params is not None:
            self._save_best(best_params, best_result)
        
        logger.info(f"")
        logger.info(f"第 {round_num} 轮完成:")
        logger.info(f"  有效: {valid_count}, 错误: {error_count}")
        if best_params:
            logger.info(f"  最佳 base_gain = {best_params.base_gain:.4f}")
            logger.info(f"  最佳 min_gain = {best_params.min_gain:.4f}")
            logger.info(f"  最佳 coupling = {best_params.coupling:.4f}")
            logger.info(f"  最佳 λ = {best_result['lyapunov_mean']:.6f}")
            logger.info(f"  最佳得分 = {best_score:.4f}")
        logger.info(f"  本轮耗时: {round_time/60:.1f} 分钟")
        logger.info(f"")
        
        return best_params, best_result
    
    def run_full_tuning(self) -> Dict:
        logger.info("=" * 80)
        logger.info("自动化精细调优工作流")
        logger.info("=" * 80)
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"粗搜维度: fast={self.fast_dim_coarse}, slow={self.slow_dim_coarse}")
        logger.info(f"精搜维度: fast={self.fast_dim_fine}, slow={self.slow_dim_fine}")
        logger.info(f"收敛阈值: {self.convergence_threshold}")
        logger.info(f"总时间限制: {self.total_time_limit/60:.0f} 分钟")
        logger.info("=" * 80)
        logger.info("")
        
        self.start_time = time.time()
        
        # ========== 第 1 轮：粗搜 ==========
        round1_params = generate_param_grid(
            center=None,
            base_gain_range=(0.02, 0.3),
            min_gain_range=(0.01, 0.1),
            coupling_range=(0.2, 0.8),
            n_base_gain=5,
            n_min_gain=3,
            n_coupling=3,
        )
        
        best_params, best_result = self.run_round(
            round_num=1,
            round_name="粗搜",
            params_list=round1_params,
            fast_dim=self.fast_dim_coarse,
            slow_dim=self.slow_dim_coarse,
        )
        
        if best_params is None:
            logger.error("粗搜没有找到有效参数，调优失败")
            self._generate_report()
            return {'success': False, 'reason': 'no_valid_params_in_round1'}
        
        # ========== 第 2 轮：中搜 ==========
        if self._has_converged():
            logger.info("已收敛，跳过后续轮次")
        else:
            ranges = generate_narrow_range(best_params, fraction=0.3)
            round2_params = generate_param_grid(
                center=best_params,
                base_gain_range=ranges['base_gain_range'],
                min_gain_range=ranges['min_gain_range'],
                coupling_range=ranges['coupling_range'],
                n_base_gain=4,
                n_min_gain=3,
                n_coupling=3,
            )
            
            best_params_r2, best_result_r2 = self.run_round(
                round_num=2,
                round_name="中搜",
                params_list=round2_params,
                fast_dim=self.fast_dim_coarse,
                slow_dim=self.slow_dim_coarse,
            )
            
            if best_params_r2 is not None:
                best_params = best_params_r2
                best_result = best_result_r2
        
        # ========== 第 3 轮：精搜 ==========
        if self._has_converged():
            logger.info("已收敛，跳过精搜")
        else:
            ranges = generate_narrow_range(best_params, fraction=0.15)
            round3_params = generate_param_grid(
                center=best_params,
                base_gain_range=ranges['base_gain_range'],
                min_gain_range=ranges['min_gain_range'],
                coupling_range=ranges['coupling_range'],
                n_base_gain=4,
                n_min_gain=3,
                n_coupling=3,
            )
            
            best_params_r3, best_result_r3 = self.run_round(
                round_num=3,
                round_name="精搜",
                params_list=round3_params,
                fast_dim=self.fast_dim_fine,
                slow_dim=self.slow_dim_fine,
            )
            
            if best_params_r3 is not None:
                best_params = best_params_r3
                best_result = best_result_r3
        
        # ========== 最终验证 ==========
        logger.info("=" * 80)
        logger.info("最终验证（256维）")
        logger.info("=" * 80)
        
        final_result = None
        if best_params is not None:
            try:
                config = create_config(
                    self.fast_dim_fine,
                    self.slow_dim_fine,
                    best_params,
                    lyapunov_window=500,
                    open_loop_hours=0.01,
                )
                final_result = run_validation(
                    config,
                    self.fast_dim_fine,
                    seed=self.seed,
                )
                score = compute_comprehensive_score(
                    final_result['lyapunov_mean'],
                    final_result['drift_rate'],
                    final_result['alignment_max_error'],
                )
                final_result['score'] = score
                final_result['fast_dim'] = self.fast_dim_fine
                final_result['slow_dim'] = self.slow_dim_fine
                
                logger.info(f"最终验证结果:")
                logger.info(f"  Lyapunov λ = {final_result['lyapunov_mean']:.6f}")
                logger.info(f"  漂移率 = {final_result['drift_rate']:.6f}")
                logger.info(f"  对齐误差 = {final_result['alignment_max_error']:.6f}")
                logger.info(f"  综合得分 = {score:.4f}")
                logger.info(f"  P0通过: {'是' if final_result['passed'] else '否'}")
                
                tuning_result = TuningResult(
                    round_num=99,
                    round_name="最终验证",
                    fast_dim=self.fast_dim_fine,
                    slow_dim=self.slow_dim_fine,
                    params={
                        'base_gain': best_params.base_gain,
                        'min_gain': best_params.min_gain,
                        'coupling': best_params.coupling,
                    },
                    lyapunov_mean=final_result['lyapunov_mean'],
                    lyapunov_max=final_result['lyapunov_max'],
                    lyapunov_min=final_result['lyapunov_min'],
                    drift_rate=final_result['drift_rate'],
                    alignment_max_error=final_result['alignment_max_error'],
                    alignment_avg_error=final_result['alignment_avg_error'],
                    score=score,
                    passed=final_result['passed'],
                    validation_time=final_result['validation_time'],
                )
                self.history.append(tuning_result)
                self._save_best(best_params, final_result)
                
            except Exception as e:
                logger.error(f"最终验证失败: {e}")
        
        total_time = time.time() - self.start_time
        
        self._save_history()
        self._generate_report()
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("调优完成！")
        logger.info("=" * 80)
        logger.info(f"总耗时: {total_time/60:.2f} 分钟")
        if best_params:
            logger.info(f"最佳参数:")
            logger.info(f"  base_gain = {best_params.base_gain:.6f}")
            logger.info(f"  min_gain = {best_params.min_gain:.6f}")
            logger.info(f"  coupling = {best_params.coupling:.6f}")
            if final_result:
                logger.info(f"最终验证 (256维):")
                logger.info(f"  Lyapunov λ = {final_result['lyapunov_mean']:.6f}")
                logger.info(f"  目标区间 (0, 0.1): {'✓ 达到' if 0 < final_result['lyapunov_mean'] < 0.1 else '✗ 未达到'}")
        logger.info("=" * 80)
        
        return {
            'success': best_params is not None,
            'best_params': asdict(best_params) if best_params else None,
            'final_result': final_result,
            'total_time_seconds': total_time,
            'num_rounds': len(self.round_best),
        }
    
    def _generate_report(self):
        total_time = time.time() - self.start_time
        
        report = f"""# 自动化精细调优报告

## 概述

- **调优方法**: 多轮网格搜索（粗搜 → 中搜 → 精搜 → 最终验证
- **调优时间**: {datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S')}
- **总耗时**: {total_time/60:.2f} 分钟
- **总轮次**: {len(self.round_best)} 轮
- **收敛阈值**: Lyapunov 变化 < {self.convergence_threshold}

## 维度配置

| 阶段 | fast_dim | slow_dim |
|------|----------|----------|
| 粗搜/中搜 | {self.fast_dim_coarse} | {self.slow_dim_coarse} |
| 精搜/最终验证 | {self.fast_dim_fine} | {self.slow_dim_fine} |

## 各轮最佳结果

| 轮次 | 名称 | base_gain | min_gain | coupling | λ均值 | 得分 | 维度 |
|------|------|-----------|----------|----------|------|------|------|
"""
        
        for rb in self.round_best:
            p = rb.get('params') or {}
            lyap = rb.get('lyapunov_mean')
            lyap_str = f"{lyap:.6f}" if lyap is not None and np.isfinite(lyap) else "N/A"
            report += (
                f"| {rb['round_num']} | {rb['round_name']} | "
                f"{p.get('base_gain', 'N/A'):.4f} | "
                f"{p.get('min_gain', 'N/A'):.4f} | "
                f"{p.get('coupling', 'N/A'):.4f} | "
                f"{lyap_str} | "
                f"{rb.get('score', 0):.4f} | "
                f"{rb['fast_dim']} |\n"
            )
        
        # 最佳参数
        best = self.round_best[-1] if self.round_best else None
        if best and best.get('params'):
            bp = best['params']
            
            # 找最终验证结果
            final_results = [r for r in self.history if r.round_name == "最终验证"]
            final = final_results[-1] if final_results else None
            
            report += f"""
## 最佳参数

| 参数 | 值 |
|------|-----|
| base_gain | {bp.get('base_gain', 0):.6f} |
| min_gain | {bp.get('min_gain', 0):.6f} |
| coupling | {bp.get('coupling', 0):.6f} |
"""
            
            if final:
                in_target = 0 < final.lyapunov_mean < 0.1
                report += f"""
## 最终验证指标（256维）

| 指标 | 值 | 目标 | 状态 |
|------|-----|------|------|
| Lyapunov λ (均值) | {final.lyapunov_mean:.6f} | (0, 0.1) | {'✓ 达到' if in_target else '✗ 未达到'} |
| Lyapunov λ (最大) | {final.lyapunov_max:.6f} | - | - |
| Lyapunov λ (最小) | {final.lyapunov_min:.6f} | - | - |
| 漂移率 | {final.drift_rate:.6f} | < 0.1 | {'✓' if final.drift_rate < 0.1 else '✗'} |
| 对齐误差（最大） | {final.alignment_max_error:.6f} | < 0.05 | {'✓' if final.alignment_max_error < 0.05 else '✗'} |
| 对齐误差（平均） | {final.alignment_avg_error:.6f} | - | - |
| 综合得分 | {final.score:.4f} | - | - |
| P0 验证通过 | {'是 ✓' if final.passed else '否 ✗'} | - | - |
| 验证时间 | {final.validation_time:.1f}秒 | - | - |
"""
        
        report += f"""
## 统计信息

- 总测试参数组合数: {len(self.history)}
- 有效结果数: {sum(1 for r in self.history if r.error is None)}
- 失败/跳过数: {sum(1 for r in self.history if r.error is not None)}
- 总耗时: {total_time/60:.2f} 分钟
- 平均每组耗时: {total_time/max(1, len(self.history)):.1f} 秒

## 目标达成情况

"""
        
        # 检查是否达到目标
        target_met = False
        if final and 0 < final.lyapunov_mean < 0.1:
            target_met = True
            report += "**Lyapunov 目标区间 (0, 0.1): ✓ 已达到 ✓\n\n"
        else:
            report += "**Lyapunov 目标区间 (0, 0.1): ✗ 未达到 ✗\n\n"
            # 分析原因
            if final:
                if final.lyapunov_mean <= 0:
                    report += "- Lyapunov 指数 ≤ 0，系统处于稳定/收敛状态，需要增大 base_gain\n"
                elif final.lyapunov_mean >= 0.1:
                    report += "- Lyapunov 指数 ≥ 0.1，系统混沌程度过高，需要减小 base_gain\n"
        
        report += """
## 调优方法说明

### 综合评分函数
- **Lyapunov 指数（60%权重）**：高斯型评分，在目标中心 0.05 处得分最高
- **漂移率（20%权重）**：线性惩罚，漂移率越低得分越高
- **对齐误差（20%权重）**：线性惩罚，误差越低得分越高

### 收敛检测
连续两轮最佳 Lyapunov 指数变化小于 0.02 时停止搜索

### 搜索策略
1. **粗搜**：大范围参数空间，低维度加速
2. **中搜**：在上轮最佳 ±30% 范围内搜索
3. **精搜**：在上轮最佳 ±15% 范围内搜索，高维度确认
4. **最终验证**：256维完整验证
"""
        
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存到 {self.report_file}")


def main():
    parser = argparse.ArgumentParser(description='自动化精细调优工作流')
    parser.add_argument(
        '--output', type=str, default='real_system_tuning',
        help='输出目录（默认 real_system_tuning）'
    )
    parser.add_argument(
        '--fast-dim-coarse', type=int, default=128,
        help='粗搜快变量维度（默认128）'
    )
    parser.add_argument(
        '--slow-dim-coarse', type=int, default=32,
        help='粗搜慢变量维度（默认32）'
    )
    parser.add_argument(
        '--fast-dim-fine', type=int, default=256,
        help='精搜快变量维度（默认256）'
    )
    parser.add_argument(
        '--slow-dim-fine', type=int, default=64,
        help='精搜慢变量维度（默认64）'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子（默认42）'
    )
    parser.add_argument(
        '--time-limit', type=float, default=120.0,
        help='总时间限制（分钟，默认120）'
    )
    parser.add_argument(
        '--convergence-threshold', type=float, default=0.02,
        help='收敛阈值（默认0.02）'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='详细输出'
    )
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S',
    )
    
    tuner = AutoFineTuner(
        output_dir=args.output,
        fast_dim_coarse=args.fast_dim_coarse,
        slow_dim_coarse=args.slow_dim_coarse,
        fast_dim_fine=args.fast_dim_fine,
        slow_dim_fine=args.slow_dim_fine,
        seed=args.seed,
        convergence_threshold=args.convergence_threshold,
        total_time_limit_minutes=args.time_limit,
    )
    
    result = tuner.run_full_tuning()
    
    return 0 if result.get('success', False) else 1


if __name__ == '__main__':
    sys.exit(main())
