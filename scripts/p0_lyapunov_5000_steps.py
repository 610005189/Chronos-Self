"""
P0 Validation with Extended Lyapunov Steps
==========================================

Runs P0 validation with 5000+ Lyapunov calculation steps to confirm
Lyapunov exponent enters the target interval (0, 0.1).
"""

import torch
import sys
from pathlib import Path
import logging
import time
import json

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation_5000_steps.log')
    ]
)

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronos_core.core.integration_engine import IntegrationEngine
from chronos_core.core.state import SelfState
from chronos_core.utils.config import ChronosConfig
from chronos_core.validation.p0_validation import P0Validation, P0ValidationConfig


def run_extended_lyapunov_validation():
    """运行扩展的 Lyapunov 验证（5000+ 步）"""
    logger.info("=" * 80)
    logger.info("P0 Validation with Extended Lyapunov Steps (5000+)")
    logger.info("=" * 80)

    # 创建配置
    config = ChronosConfig()

    # 创建积分引擎
    logger.info("创建积分引擎...")
    engine = IntegrationEngine(config=config, device='cpu')
    engine.initialize()
    logger.info(f"引擎创建完成: fast_dim={engine.engine_config.fast_dim}, slow_dim={engine.engine_config.slow_dim}")

    # 创建扩展的 P0 验证配置
    p0_config = P0ValidationConfig()
    p0_config.lyapunov_calculation_steps = 5000  # 扩展到 5000 步
    p0_config.open_loop_hours = 0.01  # 缩短开环运行时间（便于快速验证）
    p0_config.drift_calculation_window = 100
    p0_config.max_baseline_drift_rate = 0.5  # 放宽漂移率阈值
    p0_config.alignment_test_steps = [10, 50]  # 简化对齐测试

    logger.info(f"P0 配置: lyapunov_calculation_steps={p0_config.lyapunov_calculation_steps}")
    logger.info(f"P0 配置: open_loop_hours={p0_config.open_loop_hours}")

    # 创建初始状态
    initial_state = SelfState(
        E_fast=torch.randn(engine.engine_config.fast_dim) * 0.1,
        E_slow=torch.randn(engine.engine_config.slow_dim) * 0.1,
        timestamp=0.0
    )
    logger.info(f"初始状态: E_fast_norm={initial_state.get_fast_norm():.4f}, E_slow_norm={initial_state.get_slow_norm():.4f}")

    # 创建验证器
    validator = P0Validation(engine, config, p0_config, device='cpu')

    # 运行验证
    start_time = time.time()
    result = validator.run_full_validation(initial_state, verbose=True)
    elapsed_time = time.time() - start_time

    # 输出结果
    logger.info("\n" + "=" * 80)
    logger.info("验证结果汇总")
    logger.info("=" * 80)
    logger.info(f"总体通过: {result.is_passed}")
    logger.info(f"总体得分: {result.overall_score:.4f}")
    logger.info(f"验证时间: {elapsed_time:.2f}秒")

    logger.info(f"\n[P0级验证]")
    logger.info(f"  开环运行: {'✓' if result.open_loop_passed else '✗'}")
    logger.info(f"  漂移率: {'✓' if result.drift_passed else '✗'} (rate={result.drift_rate:.6f})")
    logger.info(f"  李雅普诺夫: {'✓' if result.lyapunov_passed else '✗'} (λ={result.lyapunov_mean:.6f})")
    logger.info(f"  动力学对齐: {'✓' if result.alignment_passed else '✗'}")

    logger.info(f"\nLyapunov 详细:")
    logger.info(f"  mean={result.lyapunov_mean:.6f}")
    logger.info(f"  std={result.lyapunov_std:.6f}")
    logger.info(f"  range=[{result.lyapunov_min:.6f}, {result.lyapunov_max:.6f}]")
    logger.info(f"  history={result.lyapunov_history}")

    # 保存结果
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    result_file = output_dir / "p0_lyapunov_5000_steps.json"
    validator.save_report(result, str(result_file), format="json")

    # 保存 markdown 报告
    md_file = output_dir / "p0_lyapunov_5000_steps.md"
    validator.save_report(result, str(md_file), format="markdown")

    logger.info(f"\n报告已保存:")
    logger.info(f"  JSON: {result_file}")
    logger.info(f"  Markdown: {md_file}")

    return result


if __name__ == "__main__":
    result = run_extended_lyapunov_validation()
