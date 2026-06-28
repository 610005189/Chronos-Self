"""
Chronos-Self 训练系统 CLI
=========================
训练系统命令行接口，setup.py 中注册为 `chronos-train`。
"""

import argparse
import torch
import time
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chronos_core.training.training_system import TrainingSystem, TrainingSystemConfig
from chronos_core.validation.p0_validation import P0Validation, P0ValidationConfig
from chronos_core.core.integration_engine import create_integration_engine_from_config
from chronos_core.core.state import SelfState
from chronos_core.utils.config import ChronosConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger("TrainingCLI")


def run_pre_validation(config: ChronosConfig, device: str) -> dict:
    """Run P0 validation before training"""
    logger.info("=" * 60)
    logger.info("Pre-training P0 Validation")
    logger.info("=" * 60)

    engine = create_integration_engine_from_config(config, device=device)
    state = SelfState(
        E_fast=torch.randn(config.dim.fast_variable_dim) * 0.1,
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
        timestamp=0.0
    )

    p0_config = P0ValidationConfig(
        open_loop_hours=min(200 * 0.01 / 3600, 0.001),
        drift_calculation_window=50,
        lyapunov_calculation_steps=200,
        alignment_test_steps=[10, 50],
        alignment_num_tests=2,
    )
    validator = P0Validation(engine=engine, config=config, p0_config=p0_config, device=device)
    result = validator.run_full_validation(initial_state=state, verbose=True)

    logger.info(f"Pre-training: score={result.overall_score:.4f}, passed={result.is_passed}")
    return {
        "score": result.overall_score,
        "passed": result.is_passed,
        "lyapunov": result.lyapunov_mean,
        "drift": result.drift_rate,
        "alignment_max_error": result.alignment_max_error,
    }


def run_post_validation(config: ChronosConfig, engine, device: str) -> dict:
    """Run P0 validation after training with the trained engine"""
    logger.info("=" * 60)
    logger.info("Post-training P0 Validation")
    logger.info("=" * 60)

    state = SelfState(
        E_fast=torch.randn(config.dim.fast_variable_dim) * 0.1,
        E_slow=torch.randn(config.dim.slow_variable_dim) * 0.1,
        timestamp=0.0
    )

    p0_config = P0ValidationConfig(
        open_loop_hours=min(200 * 0.01 / 3600, 0.001),
        drift_calculation_window=50,
        lyapunov_calculation_steps=200,
        alignment_test_steps=[10, 50],
        alignment_num_tests=2,
    )
    validator = P0Validation(engine=engine, config=config, p0_config=p0_config, device=device)
    result = validator.run_full_validation(initial_state=state, verbose=True)

    logger.info(f"Post-training: score={result.overall_score:.4f}, passed={result.is_passed}")
    return {
        "score": result.overall_score,
        "passed": result.is_passed,
        "lyapunov": result.lyapunov_mean,
        "drift": result.drift_rate,
        "alignment_max_error": result.alignment_max_error,
    }


def main():
    parser = argparse.ArgumentParser(description="Chronos-Self Training CLI")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fast-dim", type=int, default=256, help="Fast variable dimension")
    parser.add_argument("--slow-dim", type=int, default=64, help="Slow variable dimension")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--mode", type=str, default="alignment",
                        choices=["standard", "alignment", "p0_validation"],
                        help="Training mode")
    parser.add_argument("--output", type=str, default="training_results", help="Output directory")
    parser.add_argument("--validate", action="store_true", default=True,
                        help="Run validation before and after training")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Config
    config = ChronosConfig()
    config.dim.fast_variable_dim = args.fast_dim
    config.dim.slow_variable_dim = args.slow_dim
    config.dim.fusion_dim = args.fast_dim
    config.dim.core_subspace_dim = args.slow_dim
    config.device = args.device

    # Pre-validation
    pre = None
    if args.validate:
        pre = run_pre_validation(config, args.device)

    # Training
    logger.info("=" * 60)
    logger.info(f"Starting training: mode={args.mode}, epochs={args.epochs}, lr={args.lr}")
    logger.info("=" * 60)

    train_config = TrainingSystemConfig(
        training_mode=args.mode,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=16,
        gradient_clip_threshold=0.5,
        validation_frequency=10,
    )
    trainer = TrainingSystem(config=train_config, global_config=config, device=args.device)
    trainer.initialize()

    t0 = time.time()
    history = trainer.train(num_epochs=args.epochs)
    train_time = time.time() - t0

    logger.info(f"Training complete: {train_time:.1f}s")

    # Save checkpoint
    checkpoint_path = output_dir / f"checkpoint_epoch_{args.epochs}.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Post-validation
    post = None
    if args.validate:
        post = run_post_validation(config, trainer.integration_engine, args.device)

    # Report
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": vars(args),
        "training_time_s": train_time,
        "pre_validation": pre,
        "post_validation": post,
        "improvement": None
    }
    if pre and post:
        report["improvement"] = {
            "score_delta": post["score"] - pre["score"],
            "lyapunov_delta": post["lyapunov"] - pre["lyapunov"],
            "alignment_delta": post["alignment_max_error"] - pre["alignment_max_error"],
        }
        logger.info(f"\n{'='*60}")
        logger.info(f"Training Report")
        logger.info(f"{'='*60}")
        logger.info(f"Score:      {pre['score']:.4f} → {post['score']:.4f} "
                    f"({'↑' if post['score'] > pre['score'] else '↓'})")
        logger.info(f"Lyapunov:   {pre['lyapunov']:.4f} → {post['lyapunov']:.4f}")
        logger.info(f"Alignment:  {pre['alignment_max_error']:.4f} → {post['alignment_max_error']:.4f}")

    report_path = output_dir / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()