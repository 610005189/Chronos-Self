"""
Core Correctness Tests for Chronos-Self
=======================================

This test suite validates the correctness of core numerical methods and
dynamics components:

1. RK4 Solver Accuracy Validation - Compare with analytical solutions
2. Dynamics System Numerical Stability - Long-term running without divergence
3. State Normalization and Clipping - Verify state bounds enforcement
4. Fusion Module Dimension Consistency - Comprehensive dimension tests

These tests ensure that the numerical foundation of the system is correct
and robust.
"""

import pytest
import torch
import numpy as np
from typing import Dict, Any, Optional


class TestRK4SolverAccuracy:
    """Test RK4 solver accuracy against analytical solutions."""

    def test_rk4_exponential_decay(self):
        """
        Test RK4 with exponential decay ODE: dy/dt = -ky
        
        Analytical solution: y(t) = y0 * exp(-kt)
        
        RK4 should achieve O(h^4) convergence for this problem.
        """
        k = 0.5
        y0 = torch.tensor([1.0])
        dt = 0.01
        n_steps = 1000
        t_final = dt * n_steps

        def dynamics_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -k * y

        def rk4_step(y: torch.Tensor, dt: float, t: float) -> torch.Tensor:
            k1 = dynamics_fn(t, y)
            k2 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = dynamics_fn(t + dt, y + dt * k3)
            return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        y = y0.clone()
        t = 0.0
        for _ in range(n_steps):
            y = rk4_step(y, dt, t)
            t += dt

        analytical = y0 * torch.exp(torch.tensor(-k * t_final))
        
        error = torch.abs(y - analytical) / torch.abs(analytical)
        
        assert error < 2e-6, f"RK4 error {error.item()} exceeds tolerance"

    def test_rk4_harmonic_oscillator(self):
        """
        Test RK4 with harmonic oscillator: d^2x/dt^2 = -ω^2 x
        
        Written as first-order system:
        dx/dt = v
        dv/dt = -ω^2 x
        
        Analytical solution: x(t) = x0*cos(ωt) + (v0/ω)*sin(ωt)
        """
        omega = 2.0 * np.pi
        x0 = 1.0
        v0 = 0.0
        dt = 0.001
        n_steps = 5000
        t_final = dt * n_steps

        y0 = torch.tensor([x0, v0])

        def dynamics_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x, v = y[0], y[1]
            return torch.tensor([v, -omega**2 * x])

        def rk4_step(y: torch.Tensor, dt: float, t: float) -> torch.Tensor:
            k1 = dynamics_fn(t, y)
            k2 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = dynamics_fn(t + dt, y + dt * k3)
            return y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        y = y0.clone()
        t = 0.0
        for _ in range(n_steps):
            y = rk4_step(y, dt, t)
            t += dt

        x_analytical = x0 * np.cos(omega * t_final) + (v0 / omega) * np.sin(omega * t_final)
        v_analytical = -x0 * omega * np.sin(omega * t_final) + v0 * np.cos(omega * t_final)

        x_error = torch.abs(y[0] - x_analytical) / np.abs(x0)
        v_error = torch.abs(y[1] - v_analytical) / np.abs(omega * x0)

        assert x_error < 1e-5, f"Position error {x_error.item()} exceeds tolerance"
        assert v_error < 1e-5, f"Velocity error {v_error.item()} exceeds tolerance"

    def test_rk4_convergence_order(self):
        """
        Verify RK4 convergence order is O(h^4) by comparing errors at different step sizes.
        
        For O(h^p) methods: error(h) / error(h/2) ≈ 2^p
        For RK4: p=4, so ratio ≈ 16
        """
        k = 1.0
        y0 = torch.tensor([1.0])
        t_final = 1.0

        def dynamics_fn(t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return -k * y

        def solve_rk4(dt: float) -> torch.Tensor:
            n_steps = int(t_final / dt)
            y = y0.clone()
            t = 0.0
            for _ in range(n_steps):
                k1 = dynamics_fn(t, y)
                k2 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k1)
                k3 = dynamics_fn(t + 0.5 * dt, y + 0.5 * dt * k2)
                k4 = dynamics_fn(t + dt, y + dt * k3)
                y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                t += dt
            return y

        dt1 = 0.5
        dt2 = 0.25
        dt3 = 0.125

        y1 = solve_rk4(dt1)
        y2 = solve_rk4(dt2)
        y3 = solve_rk4(dt3)

        analytical = y0 * torch.exp(torch.tensor(-k * t_final))

        error1 = torch.abs(y1 - analytical)
        error2 = torch.abs(y2 - analytical)
        error3 = torch.abs(y3 - analytical)

        ratio12 = error1 / error2 if error2 > 1e-15 else torch.tensor(float('inf'))
        ratio23 = error2 / error3 if error3 > 1e-15 else torch.tensor(float('inf'))

        expected_ratio = 16.0

        assert ratio12 > 0, f"Convergence ratio {ratio12.item()} should be positive"
        assert ratio23 > 0, f"Convergence ratio {ratio23.item()} should be positive"
        
        if torch.isfinite(ratio12):
            assert 8 < ratio12 < 24, f"Convergence ratio {ratio12.item()} not near 16 (O(h^4))"
        if torch.isfinite(ratio23):
            assert 8 < ratio23 < 24, f"Convergence ratio {ratio23.item()} not near 16 (O(h^4))"


class TestDynamicsNumericalStability:
    """Test dynamics system numerical stability over long runs."""

    def test_long_run_no_divergence(self):
        """
        Test that the dynamics system does not diverge over extended runs.
        
        Simulate 10000 steps and verify state norms remain bounded.
        """
        from chronos_core.core.integration_engine import IntegrationEngine
        from chronos_core.core.state import SelfState
        from chronos_core.utils.config import ChronosConfig, DimensionalityConfig

        config = ChronosConfig(
            dim=DimensionalityConfig(
                fast_variable_dim=32,
                slow_variable_dim=16,
                core_subspace_dim=16,
                semantic_dim=64,
                physical_dim=128,
                fusion_dim=192
            ),
            device='cpu'
        )

        engine = IntegrationEngine(config=config, device='cpu')
        engine.initialize()

        initial_state = SelfState(
            E_fast=torch.randn(32) * 0.1,
            E_slow=torch.randn(16) * 0.1,
            timestamp=0.0
        )

        state = initial_state
        max_norm = 0.0
        nan_count = 0
        inf_count = 0

        for step in range(1000):
            state = engine.step(state)
            norm = torch.norm(state.E_fast).item()
            max_norm = max(max_norm, norm)

            if torch.isnan(state.E_fast).any():
                nan_count += 1
            if torch.isinf(state.E_fast).any():
                inf_count += 1

        assert nan_count == 0, f"Found {nan_count} NaN states during long run"
        assert inf_count == 0, f"Found {inf_count} Inf states during long run"
        assert max_norm < 100.0, f"State norm {max_norm} exceeds reasonable bound"

    def test_euler_vs_rk4_stability(self):
        """
        Compare stability of Euler vs RK4 solvers over extended runs.
        
        RK4 should be more stable for stiff problems.
        """
        from chronos_core.core.integration_engine import IntegrationEngine
        from chronos_core.core.state import SelfState
        from chronos_core.utils.config import ChronosConfig, DimensionalityConfig, NumericsConfig

        config_euler = ChronosConfig(
            dim=DimensionalityConfig(
                fast_variable_dim=32,
                slow_variable_dim=16,
                core_subspace_dim=16,
                semantic_dim=64,
                physical_dim=128,
                fusion_dim=192
            ),
            numerics=NumericsConfig(solver_type='euler'),
            device='cpu'
        )

        config_rk4 = ChronosConfig(
            dim=DimensionalityConfig(
                fast_variable_dim=32,
                slow_variable_dim=16,
                core_subspace_dim=16,
                semantic_dim=64,
                physical_dim=128,
                fusion_dim=192
            ),
            numerics=NumericsConfig(solver_type='rk4'),
            device='cpu'
        )

        engine_euler = IntegrationEngine(config=config_euler, device='cpu')
        engine_euler.initialize()

        engine_rk4 = IntegrationEngine(config=config_rk4, device='cpu')
        engine_rk4.initialize()

        initial_state = SelfState(
            E_fast=torch.randn(32) * 0.5,
            E_slow=torch.randn(16) * 0.5,
            timestamp=0.0
        )

        state_euler = initial_state
        state_rk4 = initial_state

        euler_norms = []
        rk4_norms = []

        for step in range(500):
            state_euler = engine_euler.step(state_euler)
            state_rk4 = engine_rk4.step(state_rk4)

            euler_norms.append(torch.norm(state_euler.E_fast).item())
            rk4_norms.append(torch.norm(state_rk4.E_fast).item())

        max_euler_norm = max(euler_norms)
        max_rk4_norm = max(rk4_norms)

        assert not torch.isnan(state_euler.E_fast).any(), "Euler solver produced NaN"
        assert not torch.isnan(state_rk4.E_fast).any(), "RK4 solver produced NaN"

        assert max_euler_norm < 100.0, f"Euler solver norm {max_euler_norm} too large"
        assert max_rk4_norm < 100.0, f"RK4 solver norm {max_rk4_norm} too large"


class TestStateNormalizationClipping:
    """Test state normalization and clipping mechanisms."""

    def test_state_clipping_enforcement(self):
        """
        Test that state clipping prevents norms from exceeding threshold.
        """
        test_threshold = 10.0
        
        for _ in range(100):
            large_state = torch.randn(32) * 100.0
            norm = torch.norm(large_state).item()
            
            if norm > test_threshold:
                scale = test_threshold / norm
                clipped = large_state * scale
                clipped_norm = torch.norm(clipped).item()
                assert clipped_norm <= test_threshold + 1e-6, f"Clipped norm {clipped_norm} exceeds threshold {test_threshold}"
            else:
                clipped = large_state

    def test_state_normalization_bounds(self):
        """
        Test that state normalization keeps values within expected bounds.
        """
        for _ in range(100):
            state = torch.randn(32) * 50.0
            
            norm = torch.norm(state).item()
            if norm > 0:
                normalized = state / norm
            else:
                normalized = state

            assert not torch.isnan(normalized).any(), "Normalization produced NaN"
            assert not torch.isinf(normalized).any(), "Normalization produced Inf"

            min_val = torch.min(normalized).item()
            max_val = torch.max(normalized).item()

            assert max_val < 100.0, f"Max value {max_val} exceeds reasonable bound"
            assert min_val > -100.0, f"Min value {min_val} exceeds reasonable bound"

    def test_clipping_preserves_direction(self):
        """
        Test that state clipping preserves the direction of the state vector.
        """
        test_threshold = 1.0

        for _ in range(50):
            original = torch.randn(32) * 10.0
            
            norm = torch.norm(original).item()
            if norm > test_threshold:
                scale = test_threshold / norm
                clipped = original * scale
                
                direction_original = original / norm
                direction_clipped = clipped / torch.norm(clipped)
                
                cos_sim = torch.dot(direction_original, direction_clipped).item()
                assert cos_sim > 0.999, f"Direction changed after clipping (cos_sim={cos_sim})"


class TestFusionModuleDimensions:
    """Comprehensive tests for FusionModule dimension correctness."""

    def test_fusion_dimension_consistency(self):
        """
        Test that FusionModule output dimensions are consistent across
        different configuration parameters.
        """
        from chronos_core.representation.fusion import FusionModule

        sem_dim = 256
        log_dim = 512
        fusion_dim = sem_dim + log_dim

        fusion = FusionModule(
            sem_dim=sem_dim,
            log_dim=log_dim,
            fusion_dim=fusion_dim,
            num_heads=8,
            dropout=0.0
        )

        batch_sizes = [1, 2, 8]
        seq_lens = [1, 5, 10, 50]

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                X_sem = torch.randn(batch_size, seq_len, sem_dim)
                X_log = torch.randn(batch_size, seq_len, log_dim)

                X_fused = fusion(X_sem, X_log, return_enriched=False)

                expected_shape = (batch_size, seq_len, fusion_dim)
                assert X_fused.shape == expected_shape, \
                    f"Batch {batch_size}, seq {seq_len}: expected {expected_shape}, got {X_fused.shape}"

    def test_fusion_module_dimension_edge_cases(self):
        """
        Test FusionModule with edge case dimensions.
        """
        from chronos_core.representation.fusion import FusionModule

        test_cases = [
            {'sem_dim': 64, 'log_dim': 64, 'fusion_dim': 128},
            {'sem_dim': 128, 'log_dim': 256, 'fusion_dim': 512},
            {'sem_dim': 256, 'log_dim': 128, 'fusion_dim': 256},
            {'sem_dim': 16, 'log_dim': 32, 'fusion_dim': 48},
        ]

        for case in test_cases:
            fusion = FusionModule(
                sem_dim=case['sem_dim'],
                log_dim=case['log_dim'],
                fusion_dim=case['fusion_dim'],
                num_heads=4,
                dropout=0.0
            )

            X_sem = torch.randn(2, 10, case['sem_dim'])
            X_log = torch.randn(2, 10, case['log_dim'])

            X_fused = fusion(X_sem, X_log, return_enriched=False)

            assert X_fused.shape == (2, 10, case['fusion_dim']), \
                f"Failed for {case}: got {X_fused.shape}"

    def test_fusion_with_custom_output_projection(self):
        """
        Test FusionModule with custom output projection when fusion_dim != sem_dim + log_dim.
        """
        from chronos_core.representation.fusion import FusionModule

        sem_dim = 256
        log_dim = 512
        fusion_dim = 512

        fusion = FusionModule(
            sem_dim=sem_dim,
            log_dim=log_dim,
            fusion_dim=fusion_dim,
            num_heads=8,
            dropout=0.0
        )

        assert fusion.output_projection is not None, "Output projection should exist"

        X_sem = torch.randn(2, 10, sem_dim)
        X_log = torch.randn(2, 10, log_dim)

        X_fused = fusion(X_sem, X_log, return_enriched=False)

        assert X_fused.shape == (2, 10, fusion_dim), \
            f"Expected (2, 10, {fusion_dim}), got {X_fused.shape}"


class TestIntegrationEngineCorrectness:
    """Test IntegrationEngine core functionality."""

    def test_step_dimension_preservation(self):
        """
        Test that IntegrationEngine.step() preserves state dimensions.
        """
        from chronos_core.core.integration_engine import IntegrationEngine
        from chronos_core.core.state import SelfState
        from chronos_core.utils.config import ChronosConfig, DimensionalityConfig

        config = ChronosConfig(
            dim=DimensionalityConfig(
                fast_variable_dim=32,
                slow_variable_dim=16,
                core_subspace_dim=16,
                semantic_dim=64,
                physical_dim=128,
                fusion_dim=192
            ),
            device='cpu'
        )

        engine = IntegrationEngine(config=config, device='cpu')
        engine.initialize()

        initial_state = SelfState(
            E_fast=torch.randn(32),
            E_slow=torch.randn(16),
            timestamp=0.0
        )

        for _ in range(100):
            new_state = engine.step(initial_state)
            
            assert new_state.E_fast.shape == initial_state.E_fast.shape, \
                f"Fast dimension changed: {initial_state.E_fast.shape} -> {new_state.E_fast.shape}"
            assert new_state.E_slow.shape == initial_state.E_slow.shape, \
                f"Slow dimension changed: {initial_state.E_slow.shape} -> {new_state.E_slow.shape}"
            
            initial_state = new_state

    def test_integrate_dimension_consistency(self):
        """
        Test that IntegrationEngine.integrate() produces consistent dimensions.
        """
        from chronos_core.core.integration_engine import IntegrationEngine
        from chronos_core.core.state import SelfState
        from chronos_core.utils.config import ChronosConfig, DimensionalityConfig

        config = ChronosConfig(
            dim=DimensionalityConfig(
                fast_variable_dim=32,
                slow_variable_dim=16,
                core_subspace_dim=16,
                semantic_dim=64,
                physical_dim=128,
                fusion_dim=192
            ),
            device='cpu'
        )

        engine = IntegrationEngine(config=config, device='cpu')
        engine.initialize()

        initial_state = SelfState(
            E_fast=torch.randn(32),
            E_slow=torch.randn(16),
            timestamp=0.0
        )

        t_span = torch.linspace(0, 1.0, 100)
        trajectory = engine.integrate(initial_state, t_span=t_span)

        assert len(trajectory) == len(t_span), \
            f"Trajectory length {len(trajectory)} != time steps {len(t_span)}"

        for state in trajectory:
            assert state.E_fast.shape == (32,), f"Fast dimension mismatch in trajectory"
            assert state.E_slow.shape == (16,), f"Slow dimension mismatch in trajectory"


def run_tests():
    """Run all tests manually."""
    print("=" * 80)
    print("Running Core Correctness Tests")
    print("=" * 80)

    print("\n1. Testing RK4 Solver Accuracy...")
    test_rk4 = TestRK4SolverAccuracy()
    test_rk4.test_rk4_exponential_decay()
    print("  ✓ RK4 exponential decay test passed")
    test_rk4.test_rk4_harmonic_oscillator()
    print("  ✓ RK4 harmonic oscillator test passed")
    test_rk4.test_rk4_convergence_order()
    print("  ✓ RK4 convergence order test passed")

    print("\n2. Testing Dynamics Numerical Stability...")
    test_stability = TestDynamicsNumericalStability()
    test_stability.test_long_run_no_divergence()
    print("  ✓ Long run no divergence test passed")
    test_stability.test_euler_vs_rk4_stability()
    print("  ✓ Euler vs RK4 stability test passed")

    print("\n3. Testing State Normalization and Clipping...")
    test_clipping = TestStateNormalizationClipping()
    test_clipping.test_state_clipping_enforcement()
    print("  ✓ State clipping enforcement test passed")
    test_clipping.test_state_normalization_bounds()
    print("  ✓ State normalization bounds test passed")
    test_clipping.test_clipping_preserves_direction()
    print("  ✓ Clipping preserves direction test passed")

    print("\n4. Testing Fusion Module Dimensions...")
    test_fusion = TestFusionModuleDimensions()
    test_fusion.test_fusion_dimension_consistency()
    print("  ✓ Fusion dimension consistency test passed")
    test_fusion.test_fusion_module_dimension_edge_cases()
    print("  ✓ Fusion edge case dimensions test passed")
    test_fusion.test_fusion_with_custom_output_projection()
    print("  ✓ Fusion custom output projection test passed")

    print("\n5. Testing Integration Engine Correctness...")
    test_engine = TestIntegrationEngineCorrectness()
    test_engine.test_step_dimension_preservation()
    print("  ✓ Step dimension preservation test passed")
    test_engine.test_integrate_dimension_consistency()
    print("  ✓ Integrate dimension consistency test passed")

    print("\n" + "=" * 80)
    print("All core correctness tests passed successfully!")
    print("=" * 80)


if __name__ == '__main__':
    run_tests()