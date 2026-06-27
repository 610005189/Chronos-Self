"""
P0 Quick Validation Script
==========================

Minimal script to run P0 validation with 5000 steps.
Directly tests core dynamics without full system integration.
"""

import torch
import time
import sys

sys.path.insert(0, '.')

from chronos_core.core.state import SelfState
from chronos_core.core.external_input import ExternalInput
from chronos_core.core.integration_engine import create_integration_engine_from_config
from chronos_core.validation.p0_validation import P0Validation
from chronos_core.utils.config import ChronosConfig


def run_p0_quick_validation(steps=5000, device='cpu'):
    print(f"Starting P0 Quick Validation ({steps} steps, {device})")
    start_time = time.time()
    
    try:
        config = ChronosConfig()
        config.dim.fast_variable_dim = 256
        config.dim.slow_variable_dim = 64
        config.dim.fusion_dim = 256
        config.dim.core_subspace_dim = 64
        config.device = device
        
        integration_engine = create_integration_engine_from_config(config, device=device)
        
        validation = P0Validation(engine=integration_engine, config=config, device=device)
        
        initial_state = SelfState(
            E_fast=torch.randn(config.dim.fast_variable_dim) * 0.1,
            E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
            timestamp=0.0
        )
        
        result = validation.run_full_validation(initial_state=initial_state, verbose=True)
        
        elapsed_time = time.time() - start_time
        print(f"\nValidation Complete!")
        print(f"Total Time: {elapsed_time:.2f}s")
        print(f"Overall Score: {result.overall_score:.4f}")
        print(f"Is Passed: {result.is_passed}")
        print(f"\n--- Detailed Results ---")
        print(f"Open Loop Passed: {result.open_loop_passed}")
        print(f"Drift Rate: {result.drift_rate:.6f}")
        print(f"Lyapunov Exponent: {result.lyapunov_exponent:.6f}")
        print(f"Dynamics Alignment Passed: {result.alignment_passed}")
        
        return {
            'total_score': result.overall_score,
            'is_passed': result.is_passed,
            'drift_rate': result.drift_rate,
            'lyapunov_exponent': result.lyapunov_exponent,
            'alignment_passed': result.alignment_passed
        }
        
    except Exception as e:
        print(f"❌ Validation Error: {e}")
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


if __name__ == '__main__':
    steps = 5000
    device = 'cpu'
    
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    if len(sys.argv) > 2:
        device = sys.argv[2]
    
    results = run_p0_quick_validation(steps=steps, device=device)
    
    if 'error' not in results:
        sys.exit(0 if results['is_passed'] else 1)
    else:
        sys.exit(1)
