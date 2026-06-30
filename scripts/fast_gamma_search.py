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
    spectrum: List[float] = field(default_factory=list)
    early_stopped: bool = False
    actual_steps: int = 0


def gram_schmidt_orthogonalize(Q: torch.Tensor) -> torch.Tensor:
    """
    Gram-Schmidt 正交化

    Args:
        Q: 待正交化的矩阵 [dim, k]

    Returns:
        正交化后的矩阵
    """
    Q = Q.clone()
    k = Q.shape[1]
    for i in range(k):
        v = Q[:, i]
        for j in range(i):
            v = v - torch.dot(Q[:, j], v) * Q[:, j]
        norm = torch.norm(v)
        if norm > 1e-10:
            Q[:, i] = v / norm
        else:
            Q[:, i] = torch.randn_like(v)
            Q[:, i] = Q[:, i] / torch.norm(Q[:, i])
    return Q


def compute_lyapunov_jacobian_fast(
    fast_dynamics,
    initial_state: torch.Tensor,
    slow_state: torch.Tensor,
    num_exponents: int = 10,
    warmup_steps: int = 200,
    calc_steps: int = 500,
    dt: float = 0.01,
    device: str = "cpu",
    reorth_interval: int = 10,
    early_stop_threshold: Optional[float] = None,
    early_stop_check_interval: int = 50
) -> Dict[str, Any]:
    """
    快速 Jacobian Lyapunov 谱计算

    使用 QR 分解累积计算 Lyapunov 指数，比 Wolf 算法快 10~50 倍
    支持重正交化和早停机制
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
    early_stopped = False
    actual_steps = calc_steps

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

        # 重正交化：每 reorth_interval 步执行一次 Gram-Schmidt
        if reorth_interval > 0 and (step + 1) % reorth_interval == 0:
            Q = gram_schmidt_orthogonalize(Q)

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

        # 早停检查
        if early_stop_threshold is not None and (step + 1) % early_stop_check_interval == 0:
            current_spectrum = log_evolutions / ((step + 1) * dt)
            current_lambda_max = current_spectrum[0].item()
            if current_lambda_max > early_stop_threshold:
                early_stopped = True
                actual_steps = step + 1
                logger.info(f"Early stop at step {step + 1}: λ_max={current_lambda_max:.4f} > {early_stop_threshold}")
                break

    # 使用实际计算的步数
    lyapunov_spectrum = log_evolutions / (actual_steps * dt)
    lyapunov_spectrum = torch.sort(lyapunov_spectrum, descending=True)[0]

    return {
        "lambda_max": lyapunov_spectrum[0].item(),
        "lambda_mean": lyapunov_spectrum.mean().item(),
        "lambda_sum": lyapunov_spectrum.sum().item(),
        "positive_count": (lyapunov_spectrum > 0).sum().item(),
        "spectrum": lyapunov_spectrum.cpu().numpy().tolist(),
        "early_stopped": early_stopped,
        "actual_steps": actual_steps
    }


def run_fast_gamma_validation(
    gamma: float,
    fast_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    seed: int = 42,
    warmup_steps: int = 200,
    calc_steps: int = 200,
    reorth_interval: int = 10,
    early_stop_threshold: Optional[float] = 5.0,
    early_stop_check_interval: int = 50,
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
        warmup_steps: 热身步数
        calc_steps: 计算步数
        reorth_interval: 重正交化间隔步数
        early_stop_threshold: 早停阈值，当 lambda_max 超过此值时提前终止
        early_stop_check_interval: 早停检查间隔（步数）
        enable_chaos: 是否启用混沌注入
    """
    from chronos_core.core.fast_dynamics import (
        FastDynamicsFunction, FastDynamicsConfig
    )

    start_time = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)

    slow_dim = fast_dim // 4
    semantic_dim = slow_dim
    physical_dim = slow_dim
    fusion_dim = fast_dim
    meta_cognitive_dim = slow_dim // 4

    config = FastDynamicsConfig(
        fast_dim=fast_dim,
        slow_dim=slow_dim,
        semantic_dim=semantic_dim,
        physical_dim=physical_dim,
        fusion_dim=fusion_dim,
        meta_cognitive_dim=meta_cognitive_dim,
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

    initial_fast = torch.randn(fast_dim) * 0.01
    slow_state = torch.randn(slow_dim) * 0.01

    # 使用 compute_lyapunov_jacobian_fast 函数计算
    result = compute_lyapunov_jacobian_fast(
        fast_dynamics=dynamics_fn,
        initial_state=initial_fast,
        slow_state=slow_state,
        num_exponents=min(10, fast_dim),
        warmup_steps=warmup_steps,
        calc_steps=calc_steps,
        dt=0.01,
        device="cpu",
        reorth_interval=reorth_interval,
        early_stop_threshold=early_stop_threshold,
        early_stop_check_interval=early_stop_check_interval
    )

    runtime_ms = (time.time() - start_time) * 1000

    lambda_max = result["lambda_max"]
    lambda_sum = result["lambda_sum"]
    positive_count = result["positive_count"]
    spectrum = result["spectrum"]
    early_stopped = result["early_stopped"]
    actual_steps = result["actual_steps"]
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
        status=status,
        spectrum=spectrum,
        early_stopped=early_stopped,
        actual_steps=actual_steps
    )


def _evaluate_single_config(args: tuple) -> FastValidationResult:
    """
    单配置评估包装函数（用于并行化）

    Args:
        args: (gamma, fast_dim, hidden_dim, num_layers, seed, warmup_steps, calc_steps,
               reorth_interval, early_stop_threshold, early_stop_check_interval, enable_chaos)
    """
    (gamma, fast_dim, hidden_dim, num_layers, seed, warmup_steps, calc_steps,
     reorth_interval, early_stop_threshold, early_stop_check_interval, enable_chaos) = args
    return run_fast_gamma_validation(
        gamma=gamma,
        fast_dim=fast_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seed=seed,
        warmup_steps=warmup_steps,
        calc_steps=calc_steps,
        reorth_interval=reorth_interval,
        early_stop_threshold=early_stop_threshold,
        early_stop_check_interval=early_stop_check_interval,
        enable_chaos=enable_chaos
    )


def run_parallel_gamma_search(
    gamma_values: List[float],
    fast_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    warmup_steps: int = 200,
    calc_steps: int = 200,
    reorth_interval: int = 10,
    num_processes: int = None,
    early_stop_threshold: Optional[float] = 5.0,
    early_stop_check_interval: int = 50,
    enable_chaos: bool = False
) -> List[FastValidationResult]:
    """
    并行网格搜索 gamma 参数

    Args:
        gamma_values: 要测试的 gamma 值列表
        fast_dim: 快速动力学维度
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
        warmup_steps: 热身步数
        calc_steps: 计算步数
        reorth_interval: 重正交化间隔
        num_processes: 并行进程数，默认使用 CPU 核心数（最多8个）
        early_stop_threshold: 早停阈值
        early_stop_check_interval: 早停检查间隔
        enable_chaos: 是否启用混沌注入

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
        (gamma, fast_dim, hidden_dim, num_layers, 42 + i, warmup_steps, calc_steps,
         reorth_interval, early_stop_threshold, early_stop_check_interval, enable_chaos)
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


def print_dimension_info(fast_dim: int, hidden_dim: int):
    """打印维度信息"""
    slow_dim = fast_dim // 4
    semantic_dim = slow_dim
    physical_dim = slow_dim
    fusion_dim = fast_dim
    meta_cognitive_dim = slow_dim // 4

    print("📐 维度配置:")
    print(f"   fast_dim:           {fast_dim}")
    print(f"   slow_dim:           {slow_dim}")
    print(f"   semantic_dim:       {semantic_dim}")
    print(f"   physical_dim:       {physical_dim}")
    print(f"   fusion_dim:         {fusion_dim}")
    print(f"   meta_cognitive_dim: {meta_cognitive_dim}")
    print(f"   hidden_dim:         {hidden_dim}")
    print()


def print_lyapunov_spectrum(spectrum: List[float], top_n: int = 10):
    """打印 Lyapunov 谱"""
    n = min(top_n, len(spectrum))
    print(f"   Lyapunov 谱 (前{n}个):")
    for i in range(n):
        marker = "  > 0" if spectrum[i] > 0 else "  ≤ 0"
        print(f"     λ_{i+1:<2d} = {spectrum[i]:>8.4f}{marker}")
    print()


def analyze_dimension_gamma_relation(results: List[FastValidationResult], fast_dim: int):
    """分析维度-gamma关系"""
    if not results:
        return

    print("📊 维度-Gamma 关系分析:")
    print(f"   维度: fast_dim={fast_dim}")

    positive_lambda = [r for r in results if r.lambda_max > 0]
    negative_lambda = [r for r in results if r.lambda_max <= 0]

    if positive_lambda and negative_lambda:
        # 找到临界 gamma（从负到正的过渡点）
        all_sorted = sorted(results, key=lambda r: r.gamma)
        transition_gamma = None
        for i in range(1, len(all_sorted)):
            if all_sorted[i-1].lambda_max <= 0 and all_sorted[i].lambda_max > 0:
                transition_gamma = (all_sorted[i-1].gamma + all_sorted[i].gamma) / 2
                break

        if transition_gamma is not None:
            print(f"   临界 gamma (λ_max=0): ~{transition_gamma:.4f}")

    if positive_lambda:
        chaos_gammas = [r.gamma for r in positive_lambda if 0 < r.lambda_max < 1.0]
        if chaos_gammas:
            print(f"   混沌区间 gamma: [{min(chaos_gammas):.4f}, {max(chaos_gammas):.4f}]")
            print(f"   混沌区间宽度: {max(chaos_gammas) - min(chaos_gammas):.4f}")

    # 估算维度与混沌阈值的关系
    passed_results = [r for r in results if r.passed]
    if passed_results:
        avg_gamma = np.mean([r.gamma for r in passed_results])
        print(f"   有效 gamma 平均值: {avg_gamma:.4f}")

    print()


def get_coarse_grid() -> List[float]:
    """获取粗搜索网格"""
    return [0.2, 0.5, 0.8, 1.0, 1.5, 2.0]


def get_fine_grid_from_coarse(coarse_results: List[FastValidationResult]) -> List[float]:
    """
    根据粗搜结果生成细搜网格

    找到有效区间（lambda_max 在 0~1 之间），然后在该区间内精细搜索
    """
    valid_results = [r for r in coarse_results if r.passed]

    if not valid_results:
        # 如果没有完全通过的，找 lambda_max 最接近 0.1 的两个点
        sorted_by_abs = sorted(coarse_results, key=lambda r: abs(r.lambda_max - 0.1))
        if len(sorted_by_abs) >= 2:
            gamma_min = min(sorted_by_abs[0].gamma, sorted_by_abs[1].gamma)
            gamma_max = max(sorted_by_abs[0].gamma, sorted_by_abs[1].gamma)
        else:
            gamma_min = 0.05
            gamma_max = 0.5
    else:
        gammas = [r.gamma for r in valid_results]
        gamma_min = min(gammas) * 0.8
        gamma_max = max(gammas) * 1.2

    # 生成精细网格，步长 0.01
    step = (gamma_max - gamma_min) / 20
    fine_grid = []
    g = gamma_min
    while g <= gamma_max:
        fine_grid.append(round(g, 4))
        g += step

    return fine_grid


def run_search(
    gamma_grid: List[float],
    fast_dim: int,
    hidden_dim: int,
    num_layers: int,
    warmup_steps: int,
    calc_steps: int,
    reorth_interval: int,
    parallel: bool,
    num_processes: Optional[int],
    early_stop_threshold: Optional[float],
    early_stop_check_interval: int,
    enable_chaos: bool
) -> List[FastValidationResult]:
    """执行搜索"""
    if parallel:
        results = run_parallel_gamma_search(
            gamma_values=gamma_grid,
            fast_dim=fast_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            warmup_steps=warmup_steps,
            calc_steps=calc_steps,
            reorth_interval=reorth_interval,
            num_processes=num_processes,
            early_stop_threshold=early_stop_threshold,
            early_stop_check_interval=early_stop_check_interval,
            enable_chaos=enable_chaos
        )
    else:
        results = []
        for gamma in gamma_grid:
            try:
                result = run_fast_gamma_validation(
                    gamma=gamma,
                    fast_dim=fast_dim,
                    hidden_dim=hidden_dim,
                    num_layers=num_layers,
                    warmup_steps=warmup_steps,
                    calc_steps=calc_steps,
                    reorth_interval=reorth_interval,
                    early_stop_threshold=early_stop_threshold,
                    early_stop_check_interval=early_stop_check_interval,
                    enable_chaos=enable_chaos
                )
                results.append(result)
            except Exception as e:
                print(f'{gamma:<10} {"ERROR":<12} {str(e)[:20]:<12}')

    return results


def print_results_table(results: List[FastValidationResult]):
    """打印结果表格"""
    print(f'{"gamma":<10} {"λ_max":<12} {"λ_sum":<12} {"正指数":<8} {"步数":<8} {"耗时(ms)":<10} {"状态":<15} {"结果"}')
    print("-" * 95)

    for result in results:
        status = "✅ PASS" if result.passed else "❌ FAIL"
        early_stop_status = f"({result.status})" if result.status == "rejected_early" else ""
        steps_str = str(result.actual_steps) if result.actual_steps > 0 else "-"
        print(
            f'{result.gamma:<10} {result.lambda_max:<12.4f} '
            f'{result.lambda_sum:<12.4f} {result.positive_count:<8} '
            f'{steps_str:<8} {result.runtime_ms:<10.0f} {status:<15} {early_stop_status}'
        )

    print("-" * 95)


def print_summary(results: List[FastValidationResult]):
    """打印结果摘要"""
    if not results:
        return

    valid_results = [r for r in results if r.passed]
    rejected_early_count = sum(1 for r in results if r.early_stopped)

    if valid_results:
        best = min(valid_results, key=lambda x: abs(x.lambda_max - 0.1))
        print(f"\n🎉 找到 {len(valid_results)} 个有效 gamma 值")
        print(f"   最佳 gamma={best.gamma}, λ_max={best.lambda_max:.4f}")

        # 打印最佳结果的 Lyapunov 谱
        if best.spectrum:
            print_lyapunov_spectrum(best.spectrum)
    else:
        best = min(results, key=lambda x: abs(x.lambda_max))
        print(f"\n⚠️  未找到完全通过的 gamma")
        print(f"   最接近 gamma={best.gamma}, λ_max={best.lambda_max:.4f}")

        if best.spectrum:
            print_lyapunov_spectrum(best.spectrum)

    avg_runtime = np.mean([r.runtime_ms for r in results])
    print(f"⏱️  平均单配置耗时: {avg_runtime:.0f}ms ({avg_runtime/1000:.2f}s)")

    if rejected_early_count > 0:
        print(f"   早停拒绝配置数: {rejected_early_count}/{len(results)}")


def main():
    """快速 gamma 网格搜索"""
    import argparse

    parser = argparse.ArgumentParser(description="快速 Gamma 网格搜索")

    # 维度参数
    parser.add_argument("--fast_dim", type=int, default=64,
                        help="快速动力学维度（默认：64，可选：64/128/256/512/2048）")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="隐藏层维度（默认：128）")

    # 计算步数参数
    parser.add_argument("--calc_steps", type=int, default=200,
                        help="Lyapunov 计算步数（默认：200）")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="热身步数（默认：200）")

    # 重正交化参数
    parser.add_argument("--reorth_interval", type=int, default=10,
                        help="重正交化间隔步数（默认：10，设为0禁用）")

    # 搜索模式
    parser.add_argument("--coarse", action="store_true",
                        help="粗搜模式：测试指定的粗网格点（0.2, 0.5, 0.8, 1.0, 1.5, 2.0）")
    parser.add_argument("--auto", action="store_true",
                        help="自动两阶段：先粗搜确定区间，再自动细搜")
    parser.add_argument("--fine-grid", action="store_true",
                        help="细搜模式：使用精细网格（默认模式）")

    # 并行和早停
    parser.add_argument("--parallel", action="store_true", help="使用并行模式")
    parser.add_argument("--num-processes", type=int, default=None,
                        help="并行进程数（默认：CPU核心数，最多8个）")
    parser.add_argument("--no-early-stop", action="store_true", help="禁用早停机制")
    parser.add_argument("--early-stop-threshold", type=float, default=5.0,
                        help="早停阈值（默认：5.0）")
    parser.add_argument("--enable-chaos", action="store_true", help="启用混沌注入（默认禁用）")

    # 自定义 gamma 范围
    parser.add_argument("--gamma-start", type=float, default=None,
                        help="自定义 gamma 搜索起始值")
    parser.add_argument("--gamma-end", type=float, default=None,
                        help="自定义 gamma 搜索结束值")
    parser.add_argument("--gamma-step", type=float, default=None,
                        help="自定义 gamma 搜索步长")

    args = parser.parse_args()

    early_stop_threshold = None if args.no_early_stop else args.early_stop_threshold
    early_stop_check_interval = 50  # 每50步检查一次

    print("=" * 70)
    print("快速 Gamma 网格搜索 (Jacobian + 低维代理)")
    print("=" * 70)

    # 确定搜索模式
    if args.coarse:
        search_mode = "粗搜"
    elif args.auto:
        search_mode = "自动两阶段"
    else:
        search_mode = "细搜"

    print(f"搜索模式: {search_mode}")
    print(f"执行模式: {'并行' if args.parallel else '串行'}")
    print(f"早停机制: {'禁用' if args.no_early_stop else f'启用 (阈值={args.early_stop_threshold}, 间隔={early_stop_check_interval}步)'}")
    print(f"混沌注入: {'启用' if args.enable_chaos else '禁用'}")
    print(f"重正交化: {'禁用' if args.reorth_interval <= 0 else f'每 {args.reorth_interval} 步'}")
    print(f"计算步数: 热身={args.warmup_steps}, 计算={args.calc_steps}")
    print()

    # 打印维度信息
    print_dimension_info(args.fast_dim, args.hidden_dim)

    # 阶段一：粗搜（如果是粗搜或自动模式）
    all_results = []

    if args.coarse or args.auto:
        print("=" * 70)
        print("阶段一：粗搜")
        print("=" * 70)

        coarse_grid = get_coarse_grid()
        print(f"粗搜网格: {coarse_grid}")
        print()

        coarse_results = run_search(
            gamma_grid=coarse_grid,
            fast_dim=args.fast_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            warmup_steps=args.warmup_steps,
            calc_steps=args.calc_steps,
            reorth_interval=args.reorth_interval,
            parallel=args.parallel,
            num_processes=args.num_processes,
            early_stop_threshold=early_stop_threshold,
            early_stop_check_interval=early_stop_check_interval,
            enable_chaos=args.enable_chaos
        )

        print_results_table(coarse_results)
        print_summary(coarse_results)

        all_results = coarse_results

        if args.coarse:
            # 粗搜模式到此结束
            analyze_dimension_gamma_relation(coarse_results, args.fast_dim)
            print()
            print("=" * 70)
            return

        # 自动模式：继续细搜
        print("\n" + "=" * 70)
        print("阶段二：细搜")
        print("=" * 70)

        fine_grid = get_fine_grid_from_coarse(coarse_results)
        print(f"细搜网格（基于粗搜结果）: {fine_grid}")
        print()

        fine_results = run_search(
            gamma_grid=fine_grid,
            fast_dim=args.fast_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            warmup_steps=args.warmup_steps,
            calc_steps=args.calc_steps,
            reorth_interval=args.reorth_interval,
            parallel=args.parallel,
            num_processes=args.num_processes,
            early_stop_threshold=early_stop_threshold,
            early_stop_check_interval=early_stop_check_interval,
            enable_chaos=args.enable_chaos
        )

        print_results_table(fine_results)
        print_summary(fine_results)

        all_results = fine_results

    else:
        # 细搜模式（默认）
        if args.gamma_start is not None and args.gamma_end is not None:
            # 自定义 gamma 范围
            step = args.gamma_step if args.gamma_step is not None else (args.gamma_end - args.gamma_start) / 20
            gamma_grid = []
            g = args.gamma_start
            while g <= args.gamma_end + 1e-9:
                gamma_grid.append(round(g, 6))
                g += step
            print(f"自定义搜索范围: gamma=[{args.gamma_start}, {args.gamma_end}], step={step}")
        elif args.fine_grid:
            gamma_grid = [0.100, 0.102, 0.104, 0.106, 0.108, 0.110, 0.112, 0.114, 0.116, 0.118, 0.120]
        else:
            gamma_grid = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]

        fine_results = run_search(
            gamma_grid=gamma_grid,
            fast_dim=args.fast_dim,
            hidden_dim=args.hidden_dim,
            num_layers=2,
            warmup_steps=args.warmup_steps,
            calc_steps=args.calc_steps,
            reorth_interval=args.reorth_interval,
            parallel=args.parallel,
            num_processes=args.num_processes,
            early_stop_threshold=early_stop_threshold,
            early_stop_check_interval=early_stop_check_interval,
            enable_chaos=args.enable_chaos
        )

        print_results_table(fine_results)
        print_summary(fine_results)

        all_results = fine_results

    # 维度-gamma关系分析
    analyze_dimension_gamma_relation(all_results, args.fast_dim)

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
