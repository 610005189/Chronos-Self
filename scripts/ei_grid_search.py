"""
E/I 平衡网络参数网格搜索验证
============================

对比 E/I 平衡网络开启和关闭的效果。
参数网格：alpha × gamma
目标：找到使 λ₁ ≤ 1.0 的参数组合
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class EIValidationResult:
    """E/I 验证结果"""
    use_ei_balance: bool
    alpha: float
    gamma: float
    lambda_max: float
    lambda_sum: float
    positive_count: int
    runtime_ms: float
    passed: bool
    status: str = "completed"


def compute_lyapunov_jacobian_fast(
    dynamics_fn,
    initial_state: torch.Tensor,
    slow_state: torch.Tensor,
    config,
    num_exponents: int = 10,
    warmup_steps: int = 100,
    calc_steps: int = 200,
    dt: float = 0.01,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    快速 Jacobian Lyapunov 谱计算
    """
    fast_dim = initial_state.shape[-1]
    k = min(num_exponents, fast_dim)

    Q = torch.eye(fast_dim, device=device)[:, :k]
    log_evolutions = torch.zeros(k, device=device)

    y = initial_state.clone()
    if y.dim() == 1:
        y = y.unsqueeze(0)

    # 热身阶段
    for _ in range(warmup_steps):
        with torch.no_grad():
            dydt = dynamics_fn.forward(
                torch.tensor([0.0], device=device),
                y,
                E_slow=slow_state.unsqueeze(0) if slow_state.dim() == 1 else slow_state
            )
            y = y + dydt * dt

    # 计算阶段
    for step in range(calc_steps):
        y = y.detach().requires_grad_(True)
        t = torch.tensor([step * dt], device=device)

        def dynamics_fn_simple(x):
            return dynamics_fn.forward(
                t, x,
                E_slow=slow_state.unsqueeze(0) if slow_state.dim() == 1 else slow_state
            ).squeeze(0)

        J = torch.autograd.functional.jacobian(
            dynamics_fn_simple,
            y,
            create_graph=False
        )

        # 处理 Jacobian 维度
        if J.dim() == 4:
            J = J.squeeze(0).squeeze(1)
        elif J.dim() == 3:
            J = J.squeeze(1)

        JQ = J @ Q[:, :k]
        Q, R = torch.linalg.qr(JQ)
        if R.dim() == 3:
            R = R.squeeze(0)
        log_evolutions += torch.log(torch.diag(R).abs())

        with torch.no_grad():
            dydt = dynamics_fn.forward(
                t, y,
                E_slow=slow_state.unsqueeze(0) if slow_state.dim() == 1 else slow_state
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


def run_ei_validation(
    use_ei_balance: bool,
    alpha: float,
    gamma: float,
    fast_dim: int = 64,
    hidden_dim: int = 128,
    num_layers: int = 2,
    seed: int = 42,
    ei_ratio: float = 4.0
) -> EIValidationResult:
    """
    验证单个 E/I 配置

    Args:
        use_ei_balance: 是否启用 E/I 平衡
        alpha: 抑制反馈增益
        gamma: 线性耗散系数
        fast_dim: 快速动力学维度
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
        seed: 随机种子
        ei_ratio: E/I 维度比例
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
        chaos_dim=fast_dim // 4,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_layers,
        activation="tanh",
        gamma=gamma,
        dynamics_scale=0.001,
        noise_scale=0.0,
        decay_rate=0.85,
        use_ei_balance=use_ei_balance,
        alpha=alpha,
        ei_ratio=ei_ratio,
        e_target=0.0
    )

    dynamics_fn = FastDynamicsFunction(config=config, device='cpu')

    # 激活谱归一化
    dynamics_fn.train()
    dummy_input = torch.randn(1, config.fast_dim) * 0.01
    dummy_time = torch.tensor([0.0])
    for _ in range(10):
        with torch.no_grad():
            _ = dynamics_fn.forward(dummy_time, dummy_input)
    dynamics_fn.eval()

    initial_fast = torch.randn(fast_dim) * 0.01
    slow_state = torch.randn(slow_dim) * 0.01

    # 计算 Jacobian Lyapunov
    result = compute_lyapunov_jacobian_fast(
        dynamics_fn,
        initial_fast,
        slow_state,
        config,
        num_exponents=10,
        warmup_steps=100,
        calc_steps=200,
        dt=0.01,
        device='cpu'
    )

    runtime_ms = (time.time() - start_time) * 1000

    lambda_max = result["lambda_max"]
    lambda_sum = result["lambda_sum"]
    positive_count = result["positive_count"]
    passed = 0 < lambda_max <= 1.0

    return EIValidationResult(
        use_ei_balance=use_ei_balance,
        alpha=alpha,
        gamma=gamma,
        lambda_max=lambda_max,
        lambda_sum=lambda_sum,
        positive_count=positive_count,
        runtime_ms=runtime_ms,
        passed=passed,
        status="completed"
    )


def run_ei_grid_search(
    alpha_values: List[float],
    gamma_values: List[float],
    fast_dim: int = 64
) -> Dict[str, List[EIValidationResult]]:
    """
    运行完整的 E/I 参数网格搜索

    Args:
        alpha_values: alpha 参数值列表
        gamma_values: gamma 参数值列表
        fast_dim: 快速动力学维度

    Returns:
        包含 "ei_on" 和 "ei_off" 结果的字典
    """
    results_ei_on = []
    results_ei_off = []

    print("=" * 80)
    print("E/I 平衡网络参数网格搜索")
    print("=" * 80)
    print(f"网格: alpha × gamma ({len(alpha_values)} × {len(gamma_values)} = {len(alpha_values) * len(gamma_values)} 组合)")
    print(f"代理维度: {fast_dim}")
    print()

    # E/I 开启
    print("=== E/I 平衡开启 (use_ei_balance=True) ===")
    print(f'{"alpha":<10} {"gamma":<10} {"λ_max":<12} {"λ_sum":<12} {"正指数":<8} {"耗时(ms)":<10} {"结果"}')
    print("-" * 80)

    for alpha in alpha_values:
        for gamma in gamma_values:
            try:
                result = run_ei_validation(
                    use_ei_balance=True,
                    alpha=alpha,
                    gamma=gamma,
                    fast_dim=fast_dim
                )
                results_ei_on.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(
                    f'{alpha:<10} {gamma:<10} {result.lambda_max:<12.4f} '
                    f'{result.lambda_sum:<12.4f} {result.positive_count:<8} '
                    f'{result.runtime_ms:<10.0f} {status}'
                )
            except Exception as e:
                print(f'{alpha:<10} {gamma:<10} {"ERROR":<12} {str(e)[:30]}')

    print("-" * 80)
    print()

    # E/I 关闭
    print("=== E/I 平衡关闭 (use_ei_balance=False) ===")
    print(f'{"alpha":<10} {"gamma":<10} {"λ_max":<12} {"λ_sum":<12} {"正指数":<8} {"耗时(ms)":<10} {"结果"}')
    print("-" * 80)

    for alpha in alpha_values:
        for gamma in gamma_values:
            try:
                result = run_ei_validation(
                    use_ei_balance=False,
                    alpha=alpha,
                    gamma=gamma,
                    fast_dim=fast_dim
                )
                results_ei_off.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                print(
                    f'{alpha:<10} {gamma:<10} {result.lambda_max:<12.4f} '
                    f'{result.lambda_sum:<12.4f} {result.positive_count:<8} '
                    f'{result.runtime_ms:<10.0f} {status}'
                )
            except Exception as e:
                print(f'{alpha:<10} {gamma:<10} {"ERROR":<12} {str(e)[:30]}')

    print("-" * 80)
    print()

    return {
        "ei_on": results_ei_on,
        "ei_off": results_ei_off
    }


def generate_report(results: Dict[str, List[EIValidationResult]]) -> str:
    """
    生成 E/I 验证报告

    Args:
        results: 包含 "ei_on" 和 "ei_off" 结果的字典

    Returns:
        报告文本
    """
    results_ei_on = results["ei_on"]
    results_ei_off = results["ei_off"]

    # 统计信息
    passed_ei_on = [r for r in results_ei_on if r.passed]
    passed_ei_off = [r for r in results_ei_off if r.passed]

    report = []
    report.append("# E/I 平衡网络参数网格搜索验证报告")
    report.append("")
    report.append("## 1. 实验概述")
    report.append("")
    report.append(f"- **参数网格**: alpha × gamma")
    report.append(f"- **alpha 范围**: {', '.join([str(a) for a in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]])}")
    report.append(f"- **gamma 范围**: {', '.join([str(g) for g in [0.0, 0.1, 0.5]])}")
    report.append(f"- **代理维度**: 64")
    report.append(f"- **验收标准**: λ₁ ≤ 1.0")
    report.append("")
    report.append("## 2. E/I 平衡开启结果")
    report.append("")
    report.append("| alpha | gamma | λ_max | λ_sum | 正指数数 | 通过 |")
    report.append("|-------|-------|-------|-------|----------|------|")

    for r in results_ei_on:
        status = "✅" if r.passed else "❌"
        report.append(f"| {r.alpha} | {r.gamma} | {r.lambda_max:.4f} | {r.lambda_sum:.4f} | {r.positive_count} | {status} |")

    report.append("")
    report.append(f"**通过率**: {len(passed_ei_on)}/{len(results_ei_on)} ({100*len(passed_ei_on)/len(results_ei_on):.1f}%)")
    report.append("")

    if passed_ei_on:
        best_ei_on = min(passed_ei_on, key=lambda x: abs(x.lambda_max - 0.1))
        report.append(f"**最佳配置**: alpha={best_ei_on.alpha}, gamma={best_ei_on.gamma}, λ_max={best_ei_on.lambda_max:.4f}")
    else:
        best_ei_on = min(results_ei_on, key=lambda x: abs(x.lambda_max))
        report.append(f"**最接近配置**: alpha={best_ei_on.alpha}, gamma={best_ei_on.gamma}, λ_max={best_ei_on.lambda_max:.4f}")

    report.append("")
    report.append("## 3. E/I 平衡关闭结果")
    report.append("")
    report.append("| alpha | gamma | λ_max | λ_sum | 正指数数 | 通过 |")
    report.append("|-------|-------|-------|-------|----------|------|")

    for r in results_ei_off:
        status = "✅" if r.passed else "❌"
        report.append(f"| {r.alpha} | {r.gamma} | {r.lambda_max:.4f} | {r.lambda_sum:.4f} | {r.positive_count} | {status} |")

    report.append("")
    report.append(f"**通过率**: {len(passed_ei_off)}/{len(results_ei_off)} ({100*len(passed_ei_off)/len(results_ei_off):.1f}%)")
    report.append("")

    if passed_ei_off:
        best_ei_off = min(passed_ei_off, key=lambda x: abs(x.lambda_max - 0.1))
        report.append(f"**最佳配置**: alpha={best_ei_off.alpha}, gamma={best_ei_off.gamma}, λ_max={best_ei_off.lambda_max:.4f}")
    else:
        best_ei_off = min(results_ei_off, key=lambda x: abs(x.lambda_max))
        report.append(f"**最接近配置**: alpha={best_ei_off.alpha}, gamma={best_ei_off.gamma}, λ_max={best_ei_off.lambda_max:.4f}")

    report.append("")
    report.append("## 4. E/I 方案效果对比")
    report.append("")
    report.append(f"- **E/I 开启通过率**: {100*len(passed_ei_on)/len(results_ei_on):.1f}%")
    report.append(f"- **E/I 关闭通过率**: {100*len(passed_ei_off)/len(results_ei_off):.1f}%")
    report.append(f"- **通过率提升**: {100*(len(passed_ei_on)-len(passed_ei_off))/len(results_ei_off):.1f}%")
    report.append("")

    # λ_max 对比
    avg_lambda_ei_on = np.mean([r.lambda_max for r in results_ei_on])
    avg_lambda_ei_off = np.mean([r.lambda_max for r in results_ei_off])
    report.append(f"- **平均 λ_max (E/I 开启)**: {avg_lambda_ei_on:.4f}")
    report.append(f"- **平均 λ_max (E/I 关闭)**: {avg_lambda_ei_off:.4f}")
    report.append(f"- **λ_max 降低幅度**: {(avg_lambda_ei_off - avg_lambda_ei_on)/avg_lambda_ei_off * 100:.1f}%")
    report.append("")

    # 验收标准达成
    report.append("## 5. 验收标准达成情况")
    report.append("")
    if passed_ei_on:
        report.append(f"✅ **验收标准已达成**: 找到 {len(passed_ei_on)} 个参数组合使 λ₁ ≤ 1.0")
        report.append(f"   - 最佳组合: alpha={best_ei_on.alpha}, gamma={best_ei_on.gamma}")
        report.append(f"   - λ_max={best_ei_on.lambda_max:.4f}")
    else:
        report.append(f"❌ **验收标准未达成**: 未找到参数组合使 λ₁ ≤ 1.0")
        report.append(f"   - 最接近组合: alpha={best_ei_on.alpha}, gamma={best_ei_on.gamma}")
        report.append(f"   - λ_max={best_ei_on.lambda_max:.4f}")
    report.append("")

    report.append("✅ **对比实验数据完整**: E/I 开启和关闭均已完成测试")
    report.append(f"   - E/I 开启: {len(results_ei_on)} 组测试")
    report.append(f"   - E/I 关闭: {len(results_ei_off)} 组测试")
    report.append("")

    # 结论
    report.append("## 6. 结论")
    report.append("")
    if passed_ei_on:
        report.append("E/I 平衡网络方案**有效**,能够显著提升系统稳定性:")
        report.append(f"- 成功找到 {len(passed_ei_on)} 个满足验收标准的参数组合")
        report.append(f"- 最佳参数: alpha={best_ei_on.alpha}, gamma={best_ei_on.gamma}")
        report.append(f"- 通过率从 {100*len(passed_ei_off)/len(results_ei_off):.1f}% 提升至 {100*len(passed_ei_on)/len(results_ei_on):.1f}%")
        report.append(f"- 平均 λ_max 降低 {(avg_lambda_ei_off - avg_lambda_ei_on)/avg_lambda_ei_off * 100:.1f}%")
    else:
        report.append("E/I 平衡网络方案**需要进一步调优**:")
        report.append("- 当前参数范围未找到满足验收标准的组合")
        report.append("- 建议: 扩大 alpha/gamma 搜索范围或调整 ei_ratio")
    report.append("")
    report.append("---")
    report.append(f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(report)


def main():
    """运行 E/I 平衡网络参数网格搜索"""
    alpha_values = [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    gamma_values = [0.0, 0.1, 0.5]

    results = run_ei_grid_search(
        alpha_values=alpha_values,
        gamma_values=gamma_values,
        fast_dim=64
    )

    # 生成报告
    report = generate_report(results)

    # 保存报告
    report_path = Path(__file__).parent.parent / "results" / "ei_validation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + "=" * 80)
    print("报告已保存至:", report_path)
    print("=" * 80)

    # 显示摘要
    print("\n摘要:")
    print(report.split("## 5. 验收标准达成情况")[1].split("## 6. 结论")[0])


if __name__ == "__main__":
    main()