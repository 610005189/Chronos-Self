"""
逐层谱约束验证脚本

验证 EvolutionFunctionMLP 的逐层谱约束是否有效：
1. 检查每层权重矩阵的谱范数是否 ≤ target_spectral_norm
2. 测试长时间运行后状态范数是否稳定
3. 测量 Lyapunov 指数（λ_max 和 λ_sum）
"""

import torch
import torch.nn as nn
import numpy as np
import math
import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.core.fast_dynamics import (
    EvolutionFunctionMLP,
    FastDynamicsSystem,
    FastDynamicsConfig
)


def power_iteration_spectral_norm(weight: torch.Tensor, num_iterations: int = 10) -> float:
    """
    使用幂迭代法估计权重矩阵的谱范数
    """
    v = torch.randn(weight.shape[1], device=weight.device)
    v = v / torch.norm(v)
    
    for _ in range(num_iterations):
        u = weight @ v
        u_norm = torch.norm(u)
        if u_norm < 1e-10:
            return 0.0
        u = u / u_norm
        
        v = weight.t() @ u
        v_norm = torch.norm(v)
        if v_norm < 1e-10:
            return 0.0
        v = v / v_norm
    
    spectral_norm = torch.norm(weight @ v).item()
    return spectral_norm


def validate_layer_spectral_norms(mlp: EvolutionFunctionMLP, target: float = 1.0):
    """
    验证每层权重矩阵的谱范数是否 ≤ target
    """
    print("\n=== 逐层谱范数验证 ===")
    print(f"目标谱范数: {target}")
    
    all_within_limit = True
    spectral_norms = []
    
    for i, layer in enumerate(mlp.linear_layers):
        if isinstance(layer, nn.Linear):
            spectral_norm = power_iteration_spectral_norm(layer.weight.data)
            spectral_norms.append(spectral_norm)
            
            status = "OK" if spectral_norm <= target + 1e-4 else "NO"
            if spectral_norm > target + 1e-4:
                all_within_limit = False
            
            print(f"  Layer {i}: spectral_norm = {spectral_norm:.6f} [{status}]")
    
    # 计算总 Lipschitz 上限
    total_lipschitz = 1.0
    for sn in spectral_norms:
        total_lipschitz *= sn
    
    print(f"\n总 Lipschitz 上限: {total_lipschitz:.6f}")
    print(f"验收状态: {'PASS' if all_within_limit else 'FAIL'}")
    
    return all_within_limit, spectral_norms


def test_long_term_norm_stability(mlp: EvolutionFunctionMLP, num_steps: int = 2000):
    """
    测试长时间运行后状态范数是否稳定
    
    注意：MLP 的输入和输出维度不同，所以不能直接迭代。
    这里测试的是 MLP 作为演化函数在网络中的稳定性。
    """
    print(f"\n=== 长时间范数稳定性测试 ({num_steps} 步) ===")
    
    # 创建一个输入输出维度相同的简化 MLP 用于稳定性测试
    device = next(mlp.parameters()).device
    test_mlp = EvolutionFunctionMLP(
        input_dim=64,
        output_dim=64,  # 输入输出维度相同
        hidden_dim=32,
        num_hidden_layers=2,
        activation="tanh",
        target_spectral_norm=mlp.target_spectral_norm
    ).to(device)
    test_mlp.gamma = mlp.gamma
    
    # 初始状态
    x = torch.randn(1, 64, device=device) * 0.01
    
    norms = []
    for step in range(num_steps):
        with torch.no_grad():
            dx = test_mlp(x)
            # 模拟 Euler 积分：x = x + dt * dx
            x = x + 0.01 * dx
        
        norm = torch.norm(x).item()
        norms.append(norm)
        
        if step % 500 == 0 or step == num_steps - 1:
            print(f"  Step {step}: norm = {norm:.6f}")
    
    # 检查范数是否稳定
    initial_norm = norms[0]
    final_norm = norms[-1]
    max_norm = max(norms)
    
    growth_ratio = final_norm / initial_norm if initial_norm > 1e-10 else 0
    max_growth_ratio = max_norm / initial_norm if initial_norm > 1e-10 else 0
    
    # 验收标准：最终范数 ≤ 初始范数的 2 倍
    passed = growth_ratio <= 2.0
    
    print(f"\n初始范数: {initial_norm:.6f}")
    print(f"最终范数: {final_norm:.6f}")
    print(f"最大范数: {max_norm:.6f}")
    print(f"增长比例: {growth_ratio:.2f}x")
    print(f"最大增长比例: {max_growth_ratio:.2f}x")
    print(f"验收状态: {'PASS' if passed else 'FAIL'} (目标: <= 2x)")
    
    return passed, norms


def compute_lyapunov_spectrum(mlp: EvolutionFunctionMLP, num_steps: int = 100):
    """
    计算 Lyapunov 指数谱（λ_max 和 λ_sum）
    
    λ_sum = 所有 Lyapunov 指数之和 = 相空间体积变化率
    """
    print(f"\n=== Lyapunov 指数谱计算 ({num_steps} 步) ===")
    
    device = next(mlp.parameters()).device
    input_dim = mlp.input_dim
    
    # 初始化状态和 QR 分解矩阵
    x = torch.randn(1, input_dim, device=device) * 0.01
    Q = torch.eye(input_dim, device=device)  # 初始正交矩阵
    
    lyapunov_sum = torch.zeros(input_dim, device=device)
    
    for step in range(num_steps):
        # 计算 Jacobian
        x_req = x.requires_grad_(True)
        dx = mlp(x_req)
        
        # Jacobian = d(dx)/d(x)
        jacobian = torch.zeros(input_dim, input_dim, device=device)
        for i in range(input_dim):
            grad = torch.autograd.grad(dx[0, i], x_req, retain_graph=True)[0]
            jacobian[i] = grad[0]
        
        x = x.detach()  # 移除梯度追踪
        
        # QR 分解：J @ Q = Q_new @ R
        JQ = jacobian @ Q
        Q_new, R = torch.linalg.qr(JQ)
        
        # Gram-Schmidt 重正交化（防止数值误差）
        for i in range(input_dim):
            for j in range(i):
                Q_new[:, i] = Q_new[:, i] - torch.dot(Q_new[:, j], Q_new[:, i]) * Q_new[:, j]
            norm = torch.norm(Q_new[:, i])
            if norm > 1e-10:
                Q_new[:, i] = Q_new[:, i] / norm
        
        Q = Q_new
        
        # 累积 Lyapunov 指数（log|R|的对角线）
        lyapunov_sum += torch.log(torch.abs(R.diagonal()))
        
        # 更新状态
        with torch.no_grad():
            x = x + 0.01 * mlp(x)
    
    # 平均 Lyapunov 指数
    lyapunov_exponents = lyapunov_sum / num_steps
    
    lambda_max = lyapunov_exponents.max().item()
    lambda_sum = lyapunov_exponents.sum().item()
    
    # 排序 Lyapunov 指数（降序）
    sorted_exponents = torch.sort(lyapunov_exponents, descending=True).values
    
    print(f"λ_max (最大 Lyapunov 指数): {lambda_max:.6f}")
    print(f"λ_sum (所有指数之和): {lambda_sum:.6f}")
    print(f"正指数数量: {(lyapunov_exponents > 0).sum().item()}")
    print(f"前10个 Lyapunov 指数:")
    for i in range(min(10, input_dim)):
        print(f"  λ_{i+1} = {sorted_exponents[i].item():.6f}")
    
    # 验收标准
    lambda_max_ok = 0 < lambda_max < 0.2  # 边缘混沌
    lambda_sum_ok = lambda_sum < 0  # 全局收缩
    
    print(f"\n验收状态:")
    print(f"  lambda_max in (0, 0.2): {'OK' if lambda_max_ok else 'NO'} ({lambda_max:.6f})")
    print(f"  lambda_sum < 0 (global shrink): {'OK' if lambda_sum_ok else 'NO'} ({lambda_sum:.6f})")
    
    return lambda_max, lambda_sum, lyapunov_exponents


def test_norm_decay(mlp: EvolutionFunctionMLP, decay_rate: float, num_steps: int = 500):
    """
    测试添加显式范数衰减后的稳定性
    
    正确实现：使用指数衰减（范数依赖衰减）
    演化方程：dx = MLP(x)，然后应用 x *= exp(-dt * decay_rate)
    """
    print(f"\n=== 范数衰减测试 (decay_rate={decay_rate}, {num_steps} 步) ===")
    
    device = next(mlp.parameters()).device
    input_dim = mlp.input_dim
    
    # 初始状态
    x = torch.randn(1, input_dim, device=device) * 0.01
    
    norms = []
    dt = 0.01
    
    for step in range(num_steps):
        with torch.no_grad():
            # 1. MLP 演化
            dx = mlp(x)
            # 2. Euler 积分
            x = x + dt * dx
            # 3. 指数衰减（正确实现）
            x = x * math.exp(-dt * decay_rate)
        
        norm = torch.norm(x).item()
        norms.append(norm)
        
        if step % 100 == 0 or step == num_steps - 1:
            print(f"  Step {step}: norm = {norm:.6f}")
    
    initial_norm = norms[0]
    final_norm = norms[-1]
    max_norm = max(norms)
    
    growth_ratio = final_norm / initial_norm if initial_norm > 1e-10 else 0
    
    passed = growth_ratio <= 2.0
    
    print(f"\n初始范数: {initial_norm:.6f}")
    print(f"最终范数: {final_norm:.6f}")
    print(f"最大范数: {max_norm:.6f}")
    print(f"增长比例: {growth_ratio:.2f}x")
    print(f"验收状态: {'PASS' if passed else 'FAIL'} (目标: <= 2x)")
    
    return passed, norms


def test_norm_clipping(mlp: EvolutionFunctionMLP, target_norm: float, num_steps: int = 500):
    """
    测试范数截断方法
    
    每步将状态范数截断到目标范围
    """
    print(f"\n=== 范数截断测试 (target_norm={target_norm}, {num_steps} 步) ===")
    
    device = next(mlp.parameters()).device
    input_dim = mlp.input_dim
    
    # 初始状态
    x = torch.randn(1, input_dim, device=device) * 0.01
    
    norms = []
    dt = 0.01
    
    for step in range(num_steps):
        with torch.no_grad():
            # 1. MLP 演化
            dx = mlp(x)
            # 2. Euler 积分
            x = x + dt * dx
            # 3. 范数截断
            norm = torch.norm(x)
            if norm > target_norm:
                x = x * (target_norm / norm)
        
        norm = torch.norm(x).item()
        norms.append(norm)
        
        if step % 100 == 0 or step == num_steps - 1:
            print(f"  Step {step}: norm = {norm:.6f}")
    
    initial_norm = norms[0]
    final_norm = norms[-1]
    max_norm = max(norms)
    
    growth_ratio = final_norm / initial_norm if initial_norm > 1e-10 else 0
    
    passed = growth_ratio <= 2.0
    
    print(f"\n初始范数: {initial_norm:.6f}")
    print(f"最终范数: {final_norm:.6f}")
    print(f"最大范数: {max_norm:.6f}")
    print(f"增长比例: {growth_ratio:.2f}x")
    print(f"验收状态: {'PASS' if passed else 'FAIL'} (目标: <= 2x)")
    
    return passed, norms


def main():
    print("=" * 60)
    print("最终验证：找到理想配置")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 测试不同的 target_spectral_norm，使用范数截断保持稳定性
    targets = [1.8, 1.9, 2.0, 2.1, 2.2]
    clip_norm = 0.1
    
    print(f"\n配置: 范数截断={clip_norm}, gamma=0")
    print(f"测试 target_spectral_norm 范围={targets}")
    
    results = []
    for target in targets:
        print(f"\n{'='*60}")
        print(f"测试 target_spectral_norm = {target}")
        print(f"{'='*60}")
        
        # 创建 MLP
        mlp = EvolutionFunctionMLP(
            input_dim=64,
            output_dim=64,
            hidden_dim=32,
            num_hidden_layers=2,
            activation="tanh",
            target_spectral_norm=target
        ).to(device)
        mlp.gamma = 0.0
        
        # 验证 Lyapunov 指数谱
        lambda_max, lambda_sum, _ = compute_lyapunov_spectrum(mlp, num_steps=100)
        
        # 验证范数截断稳定性
        passed, norms = test_norm_clipping(mlp, target_norm=clip_norm, num_steps=500)
        
        results.append({
            'target': target,
            'lambda_max': lambda_max,
            'lambda_sum': lambda_sum,
            'growth_ratio': norms[-1] / norms[0] if norms[0] > 1e-10 else 0,
            'stability_ok': passed
        })
    
    # 总结
    print("\n" + "=" * 60)
    print("最终验证总结")
    print("=" * 60)
    print(f"范数截断: target_norm={clip_norm}")
    print(f"\n{'target':>8} | {'λ_max':>10} | {'λ_sum':>10} | {'growth':>8} | {'状态':>12}")
    print("-" * 60)
    
    for r in results:
        # 理想状态：lambda_max in (0, 0.2)，lambda_sum < 0，stability_ok
        ideal = 0 < r['lambda_max'] < 0.2 and r['lambda_sum'] < 0 and r['stability_ok']
        close = -0.1 < r['lambda_max'] < 0.3 and r['lambda_sum'] < 0 and r['stability_ok']
        status = "IDEAL" if ideal else ("CLOSE" if close else "OPTIMIZE")
        
        print(f"{r['target']:>8.2f} | {r['lambda_max']:>10.4f} | {r['lambda_sum']:>10.4f} | "
              f"{r['growth_ratio']:>8.2f}x | {status:>12}")
    
    # 找到最优配置
    best = None
    for r in results:
        if r['stability_ok'] and r['lambda_sum'] < 0:
            if best is None or abs(r['lambda_max'] - 0.1) < abs(best['lambda_max'] - 0.1):
                best = r
    
    if best:
        ideal_status = "IDEAL" if 0 < best['lambda_max'] < 0.2 else "CLOSE"
        print(f"\n[{ideal_status}] Recommend: target_spectral_norm={best['target']}, "
              f"clip_norm={clip_norm}, lambda_max={best['lambda_max']:.4f}, lambda_sum={best['lambda_sum']:.4f}")
        
        # 输出实现建议
        if 0 < best['lambda_max'] < 0.2:
            print("\nImplementation suggestions:")
            print(f"1. EvolutionFunctionMLP: target_spectral_norm={best['target']}")
            print(f"2. FastDynamicsSystem: decay_rate=0, add norm clipping threshold={clip_norm}")
            print(f"3. gamma=0 (disable activation slope limit)")


if __name__ == "__main__":
    main()