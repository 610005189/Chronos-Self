"""
快速验证脚本 - 使用 Jacobian Lyapunov + 短轨迹 + 低维代理
========================================================

加速策略：
1. Jacobian 谱方法替代 Wolf 算法（10~50x加速）
2. 短仿真轨迹（500步热身 + 2000步评估）
3. 低维代理（64维）初筛
4. 仅计算 Lyapunov 指数
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import logging
from multiprocessing import Pool, cpu_count
import os

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class FastTuningParams:
    """快速调优参数"""
    base_gain: float = 0.1
    min_gain: float = 0.05
    coupling: float = 0.3
    decay_rate: float = 0.85
    gamma: float = 0.5
    dynamics_scale: float = 1.0


@dataclass
class FastValidationResult:
    """快速验证结果"""
    gamma: float
    lambda_max: float
    lambda_sum: float
    positive_count: int
    runtime_ms: float
    passed: bool
    status: str = "completed"  # 'completed', 'rejected_early'


def compute_lyapunov_jacobian_fast(
    fast_dynamics,
    initial_state: torch.Tensor,
    slow_state: torch.Tensor,
    num_exponents: int = 10,
    warmup_steps: int = 200,
    calc_steps: int = 500,
    dt: float = 0.01,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    快速 Jacobian Lyapunov 谱计算

    使用 QR 分解累积计算 Lyapunov 指数，比 Wolf 算法快 10~50 倍
    """
    fast_dim = initial_state.shape[-1]
    k = min(num_exponents, fast_dim)

    Q = torch.eye(fast_dim, device=device)[:, :k]
    log_evolutions = torch.zeros(k, device=device)

    y = initial_state.clone().unsqueeze(0)
    y.requires_grad_(False)

    # 热身阶段
    for _ in range(warmup_steps):
        with torch.no_grad():
            dydt = fast_dynamics.forward(
                torch.tensor([0.0], device=device),
                y,
                E_slow=slow_state.unsqueeze(0),
                X_sem=None,
                X_log=None,
                X_fused=None,
                C_meta=None,
                B_chaos=None
            )
            y = y + dydt * dt

    # 计算阶段
    for step in range(calc_steps):
        y = y.detach().requires_grad_(True)

        def dynamics_fn(x):
            return fast_dynamics.forward(
                torch.tensor([step * dt], device=device),
                x,
                E_slow=slow_state.unsqueeze(0),
                X_sem=None,
                X_log=None,
                X_fused=None,
                C_meta=None,
                B_chaos=None
            ).squeeze(0)

        J = torch.autograd.functional.jacobian(
            dynamics_fn,
            y,
            create_graph=False
        )
        # 处理 Jacobian 的维度
        # J 可能是 [output, 1, fast_dim] 或 [1, output, 1, fast_dim] 或 [output, fast_dim]
        # 目标是转换为 [fast_dim, fast_dim]
        if J.dim() == 4:
            # [1, output, 1, fast_dim] -> [output, fast_dim]
            J = J.squeeze(0).squeeze(1)
        elif J.dim() == 3:
            # [output, 1, fast_dim] -> [output, fast_dim]
            J = J.squeeze(1)

        JQ = J @ Q[:, :k]
        Q, R = torch.linalg.qr(JQ)
        log_evolutions += torch.log(torch.diag(R).abs())

        with torch.no_grad():
            dydt = fast_dynamics.forward(
                torch.tensor([step * dt], device=device),
                y,
                E_slow=slow_state.unsqueeze(0),
                X_sem=None,
                X_log=None,
                X_fused=None,
                C_meta=None,
                B_chaos=None
            )
            y = y + dydt * dt

    lyapunov_spectrum = log_evolutions / (calc_steps * dt)
    lyapunov_spectrum = torch.sort(lyapunov_spectrum, descending=True)[0]

    return {
        "lambda_max": lyapunov_spectrum[0].item(),
        "lambda_mean": lyapunov_spectrum.mean().item(),
        "lambda_sum": lyapunov_spectrum.sum().item(),
        "positive_count": (lyapunov_spectrum > 0).sum().item(),
        "spectrum": lyapunov_spectrum.cpu().numpy().tolist()
    }


def run_fast_gamma_validation(
    gamma: float,
    fast_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    seed: int = 42,
    early_stop_threshold: Optional[float] = 5.0,
    early_stop_check_interval: int = 200,
    enable_chaos: bool = False
) -> FastValidationResult:
    """
    快速验证单个 gamma 参数

    使用低维代理模型 + Jacobian 方法

    Args:
        gamma: gamma 参数值
        fast_dim: 快速动力学维度
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
        seed: 随机种子
        early_stop_threshold: 早停阈值，当 lambda_max 超过此值时提前终止
        early_stop_check_interval: 早停检查间隔（步数）
    """
    from chronos_core.core.fast_dynamics import (
        FastDynamicsFunction, FastDynamicsConfig
    )

    start_time = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    slow_dim = fast_dim // 4

    config = FastDynamicsConfig(
        fast_dim=fast_dim,
        slow_dim=slow_dim,
        semantic_dim=slow_dim,
        physical_dim=slow_dim,
        fusion_dim=fast_dim,
        meta_cognitive_dim=slow_dim // 4,
        chaos_dim=fast_dim // 4 if enable_chaos else 0,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_layers,
        activation="tanh",
        gamma=gamma,
        dynamics_scale=1.0,
        noise_scale=0.0,
        decay_rate=0.0,
    )

    dynamics_fn = FastDynamicsFunction(config=config)
    dynamics_fn.to("cpu")

    # 手动运行 forward 以激活谱归一化的幂迭代
    # 在 eval 模式下，谱归一化的幂迭代可能不执行，导致权重没有被正确缩放
    dynamics_fn.train()  # 切换到 train 模式以执行幂迭代
    dummy_input = torch.randn(1, config.fast_dim) * 0.01
    dummy_time = torch.tensor([0.0])
    for _ in range(10):  # 多次 forward 以确保幂迭代收敛
        with torch.no_grad():
            _ = dynamics_fn.forward(dummy_time, dummy_input)
    dynamics_fn.eval()  # 切换回 eval 模式

    initial_fast = torch.randn(1, fast_dim) * 0.01  # 更小的初始状态
    slow_state = torch.randn(1, slow_dim) * 0.01

    # 计算 Jacobian Lyapunov
    k = min(10, fast_dim)
    Q = torch.eye(fast_dim)[:, :k]
    log_evolutions = torch.zeros(k)

    warmup_steps = 100
    calc_steps = 200
    dt = 0.01

    y = initial_fast.clone()

    # 热身
    with torch.no_grad():
        for _ in range(warmup_steps):
            dydt = dynamics_fn.forward(torch.tensor([0.0]), y)
            y = y + dydt * dt

    # 计算阶段
    early_stopped = False
    current_lambda_max = 0.0
    actual_steps = calc_steps

    for step in range(calc_steps):
        y = y.detach().requires_grad_(True)
        t = torch.tensor([step * dt])

        def dynamics_fn_simple(x):
            return dynamics_fn.forward(t, x).squeeze(0)

        J = torch.autograd.functional.jacobian(
            dynamics_fn_simple,
            y,
            create_graph=False
        )
        # 处理 Jacobian 的维度
        # J 可能是 [output, 1, 64] 或 [1, output, 1, 64] 或 [output, 64]
        # 目标是转换为 [64, 64]
        if J.dim() == 4:
            # [1, output, 1, 64] -> [output, 64]
            J = J.squeeze(0).squeeze(1)
        elif J.dim() == 3:
            # [output, 1, 64] -> [output, 64]
            J = J.squeeze(1)

        JQ = J @ Q[:, :k]
        Q, R = torch.linalg.qr(JQ)
        # 确保 R 是二维的
        if R.dim() == 3:
            R = R.squeeze(0)
        log_evolutions += torch.log(torch.diag(R).abs())

        with torch.no_grad():
            dydt = dynamics_fn.forward(t, y)
            y = y + dydt * dt

        # 早停检查：每隔 early_stop_check_interval 步检查一次
        if early_stop_threshold is not None and (step + 1) % early_stop_check_interval == 0:
            # 计算当前的 Lyapunov 指数估计
            current_spectrum = log_evolutions / ((step + 1) * dt)
            current_lambda_max = current_spectrum[0].item()

            # 如果超过阈值，提前终止
            if current_lambda_max > early_stop_threshold:
                early_stopped = True
                actual_steps = step + 1
                logger.info(f"Early stop at step {step + 1}: λ_max={current_lambda_max:.4f} > {early_stop_threshold}")
                break

    # 使用实际计算的步数
    lyapunov_spectrum = log_evolutions / (actual_steps * dt)
    lyapunov_spectrum = torch.sort(lyapunov_spectrum, descending=True)[0]

    runtime_ms = (time.time() - start_time) * 1000

    lambda_max = lyapunov_spectrum[0].item()
    lambda_sum = lyapunov_spectrum.sum().item()
    positive_count = (lyapunov_spectrum > 0).sum().item()
    passed = 0 < lambda_max < 1.0

    # 设置状态
    status = "rejected_early" if early_stopped else "completed"

    return FastValidationResult(
        gamma=gamma,
        lambda_max=lambda_max,
        lambda_sum=lambda_sum,
        positive_count=positive_count,
        runtime_ms=runtime_ms,
        passed=passed,
        status=status
    )


def _evaluate_single_config(args: tuple) -> FastValidationResult:
    """
    单配置评估包装函数（用于并行化）

    Args:
        args: (gamma, fast_dim, hidden_dim, num_layers, seed, early_stop_threshold, early_stop_check_interval, enable_chaos)
    """
    gamma, fast_dim, hidden_dim, num_layers, seed, early_stop_threshold, early_stop_check_interval, enable_chaos = args
    return run_fast_gamma_validation(
        gamma=gamma,
        fast_dim=fast_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed=seed,
        early_stop_threshold=early_stop_threshold,
        early_stop_check_interval=early_stop_check_interval,
        enable_chaos=enable_chaos
    )


def run_parallel_gamma_search(
    gamma_values: List[float],
    fast_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    num_processes: int = None,
    early_stop_threshold: Optional[float] = 5.0,
    early_stop_check_interval: int = 200,
    enable_chaos: bool = False
) -> List[FastValidationResult]:
    """
    并行网格搜索 gamma 参数

    Args:
        gamma_values: 要测试的 gamma 值列表
        fast_dim: 快速动力学维度
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
        num_processes: 并行进程数，默认使用 CPU 核心数（最多8个）
        early_stop_threshold: 早停阈值
        early_stop_check_interval: 早停检查间隔

    Returns:
        验证结果列表
    """
    if num_processes is None:
        num_processes = min(cpu_count(), 8)
        num_processes = max(num_processes, 1)

    # 确保进程数在合理范围内
    num_processes = max(1, min(num_processes, 8))

    print(f"使用 {num_processes} 个进程并行评估 {len(gamma_values)} 个配置...")
    print()

    # 准备参数
    args_list = [
        (gamma, fast_dim, hidden_dim, num_layers, 42 + i, early_stop_threshold, early_stop_check_interval, enable_chaos)
        for i, gamma in enumerate(gamma_values)
    ]

    # 并行执行
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        results = pool.map(_evaluate_single_config, args_list)

    total_time = time.time() - start_time
    print(f"并行搜索总耗时: {total_time:.2f}s")
    print(f"平均每配置耗时: {total_time / len(gamma_values):.2f}s")
    print()

    return results


def main():
    """快速 gamma 网格搜索"""
    import argparse

    parser = argparse.ArgumentParser(description="快速 Gamma 网格搜索")
    parser.add_argument("--parallel", action="store_true", help="使用并行模式")
    parser.add_argument("--num-processes", type=int, default=None, help="并行进程数（默认：CPU核心数，最多8个）")
    parser.add_argument("--no-early-stop", action="store_true", help="禁用早停机制")
    parser.add_argument("--early-stop-threshold", type=float, default=5.0, help="早停阈值（默认：5.0）")
    parser.add_argument("--enable-chaos", action="store_true", help="启用混沌注入（默认禁用）")
    parser.add_argument("--fine-grid", action="store_true", help="使用精细网格（0.0, 0.05, 0.1, ..., 0.5）")
    args = parser.parse_args()

    print("=" * 70)
    print("快速 Gamma 网格搜索 (Jacobian + 低维代理)")
    print("=" * 70)
    print(f"模式: {'并行' if args.parallel else '串行'}")
    print(f"早停机制: {'禁用' if args.no_early_stop else f'启用 (阈值={args.early_stop_threshold})'}")
    print(f"混沌注入: {'启用' if args.enable_chaos else '禁用'}")
    print()

    if args.fine_grid:
        gamma_grid = [0.100, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.114, 0.116, 0.118, 0.120]
    else:
        gamma_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

    print(f'{"gamma":<10} {"λ_max":<12} {"λ_sum":<12} {"正指数":<8} {"耗时(ms)":<10} {"状态":<15} {"结果"}')
    print("-" * 80)

    early_stop_threshold = None if args.no_early_stop else args.early_stop_threshold

    if args.parallel:
        results = run_parallel_gamma_search(
            gamma_values=gamma_grid,
            fast_dim=64,
            num_layers=2,
            num_processes=args.num_processes,
            early_stop_threshold=early_stop_threshold,
            enable_chaos=args.enable_chaos
        )

        for result in results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            early_stop_status = f"({result.status})" if result.status == "rejected_early" else ""
            print(
                f'{result.gamma:<10} {result.lambda_max:<12.4f} '
                f'{result.lambda_sum:<12.4f} {result.positive_count:<8} '
                f'{result.runtime_ms:<10.0f} {status:<15} {early_stop_status}'
            )
    else:
        results = []
        for gamma in gamma_grid:
            try:
                result = run_fast_gamma_validation(
                    gamma=gamma,
                    fast_dim=64,
                    num_layers=2,
                    early_stop_threshold=early_stop_threshold,
                    enable_chaos=args.enable_chaos
                )
                results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                early_stop_status = f"({result.status})" if result.status == "rejected_early" else ""
                print(
                    f'{gamma:<10} {result.lambda_max:<12.4f} '
                    f'{result.lambda_sum:<12.4f} {result.positive_count:<8} '
                    f'{result.runtime_ms:<10.0f} {status:<15} {early_stop_status}'
                )
            except Exception as e:
                print(f'{gamma:<10} {"ERROR":<12} {str(e)[:20]:<12}')

    print("-" * 80)

    if results:
        valid_results = [r for r in results if r.passed]
        rejected_early_count = sum(1 for r in results if r.status == "rejected_early")

        if valid_results:
            best = min(valid_results, key=lambda x: abs(x.lambda_max - 0.1))
            print(f"\n🎉 找到 {len(valid_results)} 个有效 gamma 值")
            print(f"   最佳 gamma={best.gamma}, λ_max={best.lambda_max:.4f}")
        else:
            best = min(results, key=lambda x: abs(x.lambda_max))
            print(f"\n⚠️  未找到完全通过的 gamma")
            print(f"   最接近 gamma={best.gamma}, λ_max={best.lambda_max:.4f}")

        avg_runtime = np.mean([r.runtime_ms for r in results])
        print(f"\n⏱️  平均单配置耗时: {avg_runtime:.0f}ms ({avg_runtime/1000:.2f}s)")

        if rejected_early_count > 0:
            print(f"   早停拒绝配置数: {rejected_early_count}/{len(results)}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
