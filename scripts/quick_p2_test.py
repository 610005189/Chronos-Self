"""
快速P2测试 - 只测前两个
"""

import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.core.fast_dynamics import FastDynamicsSystem, FastDynamicsConfig
from chronos_core.core.state_controller import StateMode

import warnings
warnings.filterwarnings("ignore")


def test_signal_response():
    print("=" * 70)
    print("测试1: 信号响应特性")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256, slow_dim=64, semantic_dim=128, physical_dim=64,
        fusion_dim=256, meta_cognitive_dim=64, chaos_dim=0,
        hidden_dim=128, num_hidden_layers=2, activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.WORK, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    for i in range(500):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
    
    pre_pulse_norm = torch.norm(E_fast).item()
    
    pulse_strength = 5.0
    X_sem_pulse = torch.randn(1, 128, device=device) * pulse_strength
    
    norms = []
    for i in range(500):
        t = float(500 + i) * 0.01
        inputs = {}
        if 100 <= i < 200:
            inputs['X_sem'] = X_sem_pulse
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, inputs=inputs, dt=0.01, t=t)
        norms.append(torch.norm(E_fast).item())
    
    norms = np.array(norms)
    pre = norms[:100].mean()
    during = norms[100:200].mean()
    post = norms[300:].mean()
    
    response = abs(during - pre) / pre
    recovery = abs(post - pre) / pre
    
    print(f"基线: {pre:.2f}, 脉冲期: {during:.2f}, 恢复后: {post:.2f}")
    print(f"响应强度: {response:.4f}, 恢复偏差: {recovery:.4f}")
    
    passed = response > 0.01 and recovery < 0.2
    print(f"信号响应测试: {'PASS' if passed else 'FAIL'}")
    return passed


def test_pca():
    print()
    print("=" * 70)
    print("测试2: 状态空间有效维数（PCA）")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256, slow_dim=64, semantic_dim=128, physical_dim=64,
        fusion_dim=256, meta_cognitive_dim=64, chaos_dim=0,
        hidden_dim=128, num_hidden_layers=2, activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.WORK, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    for i in range(500):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
    
    print("收集状态样本...")
    states = []
    for i in range(2000):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(500+i)*0.01)
        states.append(E_fast.squeeze(0).cpu().numpy())
    
    states = np.array(states)
    states_centered = states - states.mean(axis=0)
    cov = states_centered.T @ states_centered / 2000
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    
    total_var = eigenvalues.sum()
    cum_var = np.cumsum(eigenvalues) / total_var
    d_90 = np.searchsorted(cum_var, 0.9) + 1
    d_95 = np.searchsorted(cum_var, 0.95) + 1
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    
    print(f"d_90 = {d_90}, d_95 = {d_95}, PR = {pr:.2f}")
    
    passed = d_90 > 10 and d_95 < 256
    print(f"状态空间维数测试: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    r1 = test_signal_response()
    r2 = test_pca()
    print()
    print("=" * 70)
    print(f"结果: 2/2 通过" if (r1 and r2) else f"结果: 部分失败")
    print("=" * 70)
