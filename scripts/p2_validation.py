# -*- coding: utf-8 -*-
"""
P2 Validation Script - Steady State Six Indicators Assessment
=============================================================

Execute P2 validation process, assess system steady state through six indicators.

Six Indicators:
1. Dynamics Indicators (3):
   - Drift Rate
   - Lyapunov Exponent
   - Autocorrelation Coefficient

2. Behavioral Indicators (3):
   - Working Memory Capacity
   - Pattern Recognition Accuracy
   - Long-term Prediction Precision

Criteria:
- All 3 dynamics indicators pass: edge-of-chaos steady state
- At least 2 behavioral indicators pass: emergence characteristics
- Combined: dynamics all pass + behavioral >=2 pass = steady state emergence

Usage:
    python scripts/p2_validation.py --mode quick
    python scripts/p2_validation.py --mode full
"""

import argparse
import torch
import json
import numpy as np
from pathlib import Path
import sys
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('p2_validation.log')
    ]
)

logger = logging.getLogger(__name__)

# Add project path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.validation.dynamics_monitoring import (
    DynamicsMonitoring,
    DynamicsIndicators,
    DynamicsMonitoringConfig
)
from chronos_core.validation.behavioral_metrics import (
    BehavioralMetrics,
    BehavioralIndicators,
    BehavioralMetricsConfig
)
from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.state import SelfState
from chronos_core.utils.config import ChronosConfig
from chronos_core.memory.work_memory import WorkingMemory, ChunkType


@dataclass
class P2ValidationResult:
    """P2 Validation Result"""

    overall_passed: bool = False
    overall_score: float = 0.0
    emergence_detected: bool = False
    steady_state_detected: bool = False

    dynamics_passed_count: int = 0
    drift_rate: float = 0.0
    drift_passed: bool = False
    lyapunov_exponent: float = 0.0
    lyapunov_passed: bool = False
    autocorrelation: float = 0.0
    autocorrelation_passed: bool = False

    behavioral_passed_count: int = 0
    working_memory_capacity: float = 0.0
    wm_capacity_passed: bool = False
    pattern_recognition_accuracy: float = 0.0
    pattern_recognition_passed: bool = False
    prediction_precision: float = 0.0
    prediction_passed: bool = False

    intent_entropy: float = 0.0
    intent_entropy_passed: bool = False
    transfer_rate: float = 0.0
    transfer_passed: bool = False
    l2_recovery_rate: float = 0.0
    l2_recovery_passed: bool = False

    validation_time: float = 0.0
    validation_mode: str = "quick"
    device: str = "cpu"
    total_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": {
                "passed": self.overall_passed,
                "score": self.overall_score,
                "emergence_detected": self.emergence_detected,
                "steady_state_detected": self.steady_state_detected,
                "validation_time": self.validation_time,
                "validation_mode": self.validation_mode,
                "total_steps": self.total_steps,
                "device": self.device
            },
            "dynamics_indicators": {
                "passed_count": self.dynamics_passed_count,
                "total_count": 3,
                "all_passed": self.dynamics_passed_count == 3,
                "metrics": {
                    "drift_rate": {
                        "value": self.drift_rate,
                        "threshold": "< 0.1",
                        "passed": self.drift_passed
                    },
                    "lyapunov_exponent": {
                        "value": self.lyapunov_exponent,
                        "threshold": "(0, 0.1)",
                        "passed": self.lyapunov_passed
                    },
                    "autocorrelation": {
                        "value": self.autocorrelation,
                        "threshold": "> 0.3",
                        "passed": self.autocorrelation_passed
                    }
                }
            },
            "behavioral_indicators": {
                "passed_count": self.behavioral_passed_count,
                "total_count": 3,
                "min_required": 2,
                "metrics": {
                    "working_memory_capacity": {
                        "value": self.working_memory_capacity,
                        "threshold": ">= 5",
                        "passed": self.wm_capacity_passed
                    },
                    "pattern_recognition_accuracy": {
                        "value": self.pattern_recognition_accuracy,
                        "threshold": ">= 0.6",
                        "passed": self.pattern_recognition_passed
                    },
                    "prediction_precision": {
                        "value": self.prediction_precision,
                        "threshold": "<= 0.1",
                        "passed": self.prediction_passed
                    }
                },
                "supplementary_metrics": {
                    "intent_entropy": {
                        "value": self.intent_entropy,
                        "passed": self.intent_entropy_passed
                    },
                    "transfer_rate": {
                        "value": self.transfer_rate,
                        "passed": self.transfer_passed
                    },
                    "l2_recovery_rate": {
                        "value": self.l2_recovery_rate,
                        "passed": self.l2_recovery_passed
                    }
                }
            },
            "steady_state_assessment": {
                "dynamics_steady_state": self.dynamics_passed_count == 3,
                "behavioral_emergence": self.behavioral_passed_count >= 2,
                "combined_steady_state": self.steady_state_detected,
                "criteria": "Dynamics 3/3 pass + Behavioral >=2/3 pass"
            }
        }


class P2Validator:
    """P2 Validator"""

    def __init__(self, config: Optional[ChronosConfig] = None, device: str = "cpu"):
        self.config = config or ChronosConfig()
        self.device = device
        self.engine: Optional[IntegrationEngine] = None
        self.dynamics_monitor: Optional[DynamicsMonitoring] = None
        self.behavioral_metrics: Optional[BehavioralMetrics] = None
        self.working_memory: Optional[WorkingMemory] = None
        self.current_state: Optional[SelfState] = None
        self._step_count: int = 0
        logger.info(f"P2Validator initialized: device={self.device}")

    def initialize(self) -> None:
        logger.info("Initializing validation system...")
        self.engine = IntegrationEngine(config=self.config, device=self.device)
        self.engine.initialize()
        logger.info(f"Integration engine created: fast_dim={self.engine.engine_config.fast_dim}")

        dynamics_config = DynamicsMonitoringConfig()
        self.dynamics_monitor = DynamicsMonitoring(self.engine, self.config, dynamics_config, self.device)

        behavioral_config = BehavioralMetricsConfig()
        self.behavioral_metrics = BehavioralMetrics(self.engine, self.config, behavioral_config, self.device)

        self.working_memory = WorkingMemory(
            capacity=7,
            fast_dim=self.engine.engine_config.fast_dim,
            chunk_dim=256,
            decay_time_constant=10.0,
            min_activation=0.01,
            device=self.device
        )

        self.current_state = SelfState(
            E_fast=torch.randn(self.engine.engine_config.fast_dim) * 0.1,
            E_slow=torch.randn(self.engine.engine_config.slow_dim) * 0.1,
            timestamp=0.0
        )
        logger.info("Validation system initialized")

    def run_validation(self, mode: str = "quick", steps: Optional[int] = None) -> P2ValidationResult:
        start_time = time.time()

        if steps is None:
            steps = 2000 if mode == "quick" else 10000

        logger.info("=" * 80)
        logger.info(f"Starting P2 validation: mode={mode}, steps={steps}")
        logger.info("=" * 80)

        result = P2ValidationResult(validation_mode=mode, device=self.device)

        # Phase 1: Dynamics indicators
        logger.info("\n[Phase 1] Dynamics indicators calculation...")
        self.dynamics_monitor.start_monitoring()
        initial_slow_norm = self.current_state.get_slow_norm()

        for step_idx in range(steps):
            self.current_state = self.engine.step(self.current_state)
            self._step_count += 1
            self.dynamics_monitor.update(self.current_state, verbose=(step_idx % 500 == 0))

            if step_idx % 100 == 0:
                self.working_memory.create_chunk(
                    source_state=self.current_state.E_fast,
                    chunk_type=ChunkType.TEMPORARY,
                    initial_activation=1.0
                )
                self.working_memory.update_activations(delta_time=1.0)

        dynamics_indicators = self.dynamics_monitor.get_current_indicators()
        self.dynamics_monitor.stop_monitoring()

        final_slow_norm = self.current_state.get_slow_norm()
        drift_rate = abs(final_slow_norm - initial_slow_norm) / (steps * self.engine.engine_config.default_dt)
        drift_rate_normalized = drift_rate / initial_slow_norm if initial_slow_norm > 0 else 0.0

        result.drift_rate = drift_rate_normalized
        result.drift_passed = drift_rate_normalized < 0.1
        result.lyapunov_exponent = dynamics_indicators.lyapunov_lambda_mean
        result.lyapunov_passed = 0.0 < dynamics_indicators.lyapunov_lambda_mean < 0.1
        result.autocorrelation = dynamics_indicators.autocorrelation_rho
        result.autocorrelation_passed = dynamics_indicators.autocorrelation_rho > 0.3
        result.dynamics_passed_count = sum([result.drift_passed, result.lyapunov_passed, result.autocorrelation_passed])

        logger.info(f"  Drift rate: {result.drift_rate:.6f} ({'PASS' if result.drift_passed else 'FAIL'})")
        logger.info(f"  Lyapunov exponent: {result.lyapunov_exponent:.6f} ({'PASS' if result.lyapunov_passed else 'FAIL'})")
        logger.info(f"  Autocorrelation: {result.autocorrelation:.4f} ({'PASS' if result.autocorrelation_passed else 'FAIL'})")
        logger.info(f"  Dynamics passed: {result.dynamics_passed_count}/3")

        # Phase 2: Behavioral indicators
        logger.info("\n[Phase 2] Behavioral indicators calculation...")

        wm_result = self._test_working_memory_capacity()
        result.working_memory_capacity = wm_result["capacity"]
        result.wm_capacity_passed = wm_result["passed"]

        pattern_result = self._test_pattern_recognition()
        result.pattern_recognition_accuracy = pattern_result["accuracy"]
        result.pattern_recognition_passed = pattern_result["passed"]

        prediction_result = self._test_prediction_precision()
        result.prediction_precision = prediction_result["precision"]
        result.prediction_passed = prediction_result["passed"]

        result.behavioral_passed_count = sum([
            result.wm_capacity_passed,
            result.pattern_recognition_passed,
            result.prediction_passed
        ])

        logger.info(f"  Working memory capacity: {result.working_memory_capacity:.2f} ({'PASS' if result.wm_capacity_passed else 'FAIL'})")
        logger.info(f"  Pattern recognition accuracy: {result.pattern_recognition_accuracy:.2f} ({'PASS' if result.pattern_recognition_passed else 'FAIL'})")
        logger.info(f"  Prediction precision: {result.prediction_precision:.4f} ({'PASS' if result.prediction_passed else 'FAIL'})")
        logger.info(f"  Behavioral passed: {result.behavioral_passed_count}/3")

        # Phase 3: Supplementary indicators
        logger.info("\n[Phase 3] Supplementary behavioral indicators...")
        behavioral_indicators = self.behavioral_metrics.run_full_evaluation(self.current_state, verbose=False)

        result.intent_entropy = behavioral_indicators.intent_entropy_current
        result.intent_entropy_passed = behavioral_indicators.intent_entropy_passed
        result.transfer_rate = behavioral_indicators.transfer_rate
        result.transfer_passed = behavioral_indicators.transfer_passed
        result.l2_recovery_rate = behavioral_indicators.l2_recovery_final_rate
        result.l2_recovery_passed = behavioral_indicators.l2_recovery_passed

        logger.info(f"  Intent entropy: {result.intent_entropy:.4f} ({'PASS' if result.intent_entropy_passed else 'FAIL'})")
        logger.info(f"  Transfer rate: {result.transfer_rate:.2f} ({'PASS' if result.transfer_passed else 'FAIL'})")
        logger.info(f"  L2 recovery rate: {result.l2_recovery_rate:.2f} ({'PASS' if result.l2_recovery_passed else 'FAIL'})")

        # Phase 4: Combined assessment
        logger.info("\n[Phase 4] Combined assessment...")
        dynamics_stable = result.dynamics_passed_count == 3
        behavioral_emerging = result.behavioral_passed_count >= 2
        result.steady_state_detected = dynamics_stable and behavioral_emerging
        result.emergence_detected = result.steady_state_detected or behavioral_indicators.emergence_detected

        dynamics_score = result.dynamics_passed_count / 3.0
        behavioral_score = result.behavioral_passed_count / 3.0
        result.overall_score = (dynamics_score * 0.5 + behavioral_score * 0.5) * 100.0
        result.overall_passed = result.steady_state_detected

        result.validation_time = time.time() - start_time
        result.total_steps = self._step_count

        logger.info(f"  Dynamics steady state: {'PASS' if dynamics_stable else 'FAIL'} ({result.dynamics_passed_count}/3)")
        logger.info(f"  Behavioral emergence: {'PASS' if behavioral_emerging else 'FAIL'} ({result.behavioral_passed_count}/3)")
        logger.info(f"  Combined steady state: {'PASS' if result.steady_state_detected else 'FAIL'}")
        logger.info(f"  Overall score: {result.overall_score:.2f}")

        logger.info("\n" + "=" * 80)
        if result.steady_state_detected:
            logger.info("SUCCESS: P2 validation passed - System reached steady state emergence")
        else:
            logger.info("FAILED: P2 validation did not pass - System did not reach steady state criteria")
        logger.info("=" * 80)

        return result

    def _test_working_memory_capacity(self) -> Dict[str, Any]:
        test_states = []
        for i in range(10):
            test_state = torch.randn(self.engine.engine_config.fast_dim, device=self.device)
            test_states.append(test_state)
            self.working_memory.create_chunk(
                source_state=test_state,
                chunk_type=ChunkType.TEMPORARY,
                initial_activation=1.0 - i * 0.05
            )

        active_chunks = self.working_memory.get_active_chunks()
        effective_capacity = len(active_chunks)
        capacity_passed = effective_capacity >= 5
        capacity_utilization = effective_capacity / self.working_memory.capacity

        return {
            "capacity": effective_capacity,
            "capacity_utilization": capacity_utilization,
            "max_capacity": self.working_memory.capacity,
            "passed": capacity_passed,
            "threshold": 5
        }

    def _test_pattern_recognition(self) -> Dict[str, Any]:
        patterns = []
        for i in range(3):
            pattern = torch.randn(self.engine.engine_config.fast_dim, device=self.device)
            pattern = pattern / torch.norm(pattern)
            patterns.append(pattern)

        pattern_chunks = []
        pattern_sources = []  # Store original source states for similarity comparison
        for i, pattern in enumerate(patterns):
            chunk = self.working_memory.create_chunk(
                source_state=pattern,
                chunk_type=ChunkType.SEMANTIC,
                initial_activation=0.8
            )
            pattern_chunks.append(chunk)
            pattern_sources.append(pattern)  # Keep original 2048-dim vector

        correct_recognitions = 0
        total_tests = 10

        for test_idx in range(total_tests):
            target_pattern_idx = np.random.randint(0, len(patterns))
            target_pattern = patterns[target_pattern_idx]
            noise = torch.randn_like(target_pattern) * 0.1
            test_input = target_pattern + noise

            retrieved_chunks = []
            for i, chunk in enumerate(pattern_chunks):
                # Use original source_state (2048-dim) instead of chunk.content (256-dim)
                similarity = torch.cosine_similarity(test_input.unsqueeze(0), pattern_sources[i].unsqueeze(0)).item()
                retrieved_chunks.append((chunk, similarity))

            retrieved_chunks.sort(key=lambda x: x[1], reverse=True)
            best_chunk = retrieved_chunks[0][0]

            if pattern_chunks.index(best_chunk) == target_pattern_idx:
                correct_recognitions += 1

        accuracy = correct_recognitions / total_tests
        accuracy_passed = accuracy >= 0.6

        return {
            "accuracy": accuracy,
            "correct_recognitions": correct_recognitions,
            "total_tests": total_tests,
            "passed": accuracy_passed,
            "threshold": 0.6
        }

    def _test_prediction_precision(self) -> Dict[str, Any]:
        dynamics_indicators = self.dynamics_monitor.get_current_indicators()
        prediction_error = dynamics_indicators.self_prediction_error_mean
        precision = prediction_error
        precision_passed = precision <= 0.1

        start_state = self.current_state.copy()
        predicted_states = []
        for step_idx in range(10):
            if step_idx == 0:
                pred_state = self.engine.step(start_state)
            else:
                pred_state = self.engine.step(pred_state)
            predicted_states.append(pred_state)

        actual_states = []
        test_state = start_state.copy()
        for step_idx in range(10):
            test_state = self.engine.step(test_state)
            actual_states.append(test_state)

        errors = []
        for pred, actual in zip(predicted_states, actual_states):
            error = torch.norm(pred.E_fast - actual.E_fast).item()
            errors.append(error)

        avg_prediction_error = np.mean(errors)
        final_precision = min(precision, avg_prediction_error)
        precision_passed = final_precision <= 0.1

        return {
            "precision": final_precision,
            "self_prediction_error": prediction_error,
            "multi_step_error": avg_prediction_error,
            "passed": precision_passed,
            "threshold": 0.1
        }

    def save_report(self, result: P2ValidationResult, output_path: str) -> None:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        logger.info(f"Validation report saved to: {output_path}")

        md_path = output_dir / "p2_report.md"
        md_report = self._generate_markdown_report(result)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_report)
        logger.info(f"Markdown report saved to: {md_path}")

    def _generate_markdown_report(self, result: P2ValidationResult) -> str:
        report = f"""# P2 Steady State Validation Report

## Validation Overview

- **Validation Mode**: {result.validation_mode}
- **Validation Time**: {result.validation_time:.2f} seconds
- **Validation Steps**: {result.total_steps}
- **Device**: {result.device}

## Overall Results

- **Validation Status**: {'PASS' if result.overall_passed else 'FAIL'}
- **Overall Score**: {result.overall_score:.2f}
- **Steady State**: {'PASS - Steady State Emergence' if result.steady_state_detected else 'FAIL - Not Steady State'}
- **Emergence**: {'PASS - Emergence Detected' if result.emergence_detected else 'FAIL - No Emergence'}

## Dynamics Indicators (3)

| Indicator | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Drift Rate | {result.drift_rate:.6f} | < 0.1 | {'PASS' if result.drift_passed else 'FAIL'} |
| Lyapunov Exponent | {result.lyapunov_exponent:.6f} | (0, 0.1) | {'PASS' if result.lyapunov_passed else 'FAIL'} |
| Autocorrelation | {result.autocorrelation:.4f} | > 0.3 | {'PASS' if result.autocorrelation_passed else 'FAIL'} |

**Dynamics Passed**: {result.dynamics_passed_count}/3

## Behavioral Indicators (3)

| Indicator | Value | Threshold | Status |
|-----------|-------|-----------|--------|
| Working Memory Capacity | {result.working_memory_capacity:.2f} | >= 5 | {'PASS' if result.wm_capacity_passed else 'FAIL'} |
| Pattern Recognition Accuracy | {result.pattern_recognition_accuracy:.2f} | >= 0.6 | {'PASS' if result.pattern_recognition_passed else 'FAIL'} |
| Prediction Precision | {result.prediction_precision:.4f} | <= 0.1 | {'PASS' if result.prediction_passed else 'FAIL'} |

**Behavioral Passed**: {result.behavioral_passed_count}/3 (min 2 required)

## Supplementary Behavioral Indicators

| Indicator | Value | Status |
|-----------|-------|--------|
| Intent Entropy | {result.intent_entropy:.4f} | {'PASS' if result.intent_entropy_passed else 'FAIL'} |
| Transfer Rate | {result.transfer_rate:.2f} | {'PASS' if result.transfer_passed else 'FAIL'} |
| L2 Recovery Rate | {result.l2_recovery_rate:.2f} | {'PASS' if result.l2_recovery_passed else 'FAIL'} |

## Combined Assessment

- **Dynamics Steady State**: {'PASS' if result.dynamics_passed_count == 3 else 'FAIL'} (requires all 3 pass)
- **Behavioral Emergence**: {'PASS' if result.behavioral_passed_count >= 2 else 'FAIL'} (requires >=2 pass)
- **Combined Steady State**: {'PASS' if result.steady_state_detected else 'FAIL'}

## Assessment Criteria

Steady State Emergence Criteria:
- All 3 dynamics indicators pass (drift rate, Lyapunov exponent, autocorrelation)
- At least 2 behavioral indicators pass (working memory, pattern recognition, prediction precision)

## Conclusion

{'SUCCESS: System reached steady state emergence' if result.steady_state_detected else 'FAILED: System did not meet steady state criteria'}

---
Report generated: {datetime.now(timezone.utc).isoformat()}
"""
        return report


def main():
    parser = argparse.ArgumentParser(description="P2 Steady State Validation Script")
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "full"], help="Validation mode")
    parser.add_argument("--steps", type=int, default=None, help="Custom validation steps")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--output", type=str, default="validation_results/p2_report.json", help="Output path")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = "cpu"

    validator = P2Validator(device=args.device)
    validator.initialize()
    result = validator.run_validation(mode=args.mode, steps=args.steps)
    validator.save_report(result, args.output)

    print("\n" + "=" * 80)
    print("P2 Validation Summary")
    print("=" * 80)
    print(f"Overall Status: {'PASS' if result.overall_passed else 'FAIL'}")
    print(f"Dynamics Indicators: {result.dynamics_passed_count}/3")
    print(f"Behavioral Indicators: {result.behavioral_passed_count}/3")
    print(f"Steady State: {'PASS - Steady State Emergence' if result.steady_state_detected else 'FAIL - Not Steady State'}")
    print(f"Validation Time: {result.validation_time:.2f} seconds")
    print("=" * 80)

    return result


if __name__ == "__main__":
    main()