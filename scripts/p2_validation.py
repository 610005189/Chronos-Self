"""
P2级验证 - 动力学特性验证
"""

import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.core.fast_dynamics import FastDynamicsSystem, FastDynamicsConfig
from chronos_core.core.state_controller import StateMode


def test_signal_response():
    """测试系统对输入信号的响应"""
    print("=" * 70)
    print("测试1: 信号响应特性")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256,
        slow_dim=64,
        semantic_dim=128,
        physical_dim=64,
        fusion_dim=256,
        meta_cognitive_dim=64,
        chaos_dim=0,
        hidden_dim=128,
        num_hidden_layers=2,
        activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.WORK, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    # 先跑500步达到稳态
    for i in range(500):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
    
    # 记录基线
    baseline_norm = torch.norm(E_fast).item()
    
    # 施加脉冲输入（语义流）
    pulse_strength = 5.0
    X_sem_pulse = torch.randn(1, 128, device=device) * pulse_strength
    
    # 记录响应
    norms = []
    for i in range(500):
        t = float(500 + i) * 0.01
        inputs = {}
        if 100 <= i < 200:  # 第100-200步施加输入
            inputs['X_sem'] = X_sem_pulse
        
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, inputs=inputs, dt=0.01, t=t)
        norms.append(torch.norm(E_fast).item())
    
    norms = np.array(norms)
    
    # 分析响应
    pre_pulse = norms[:100].mean()
    during_pulse = norms[100:200].mean()
    post_pulse = norms[300:].mean()
    
    response_strength = abs(during_pulse - pre_pulse) / pre_pulse
    recovery = abs(post_pulse - pre_pulse) / pre_pulse
    
    print(f"基线范数: {pre_pulse:.2f}")
    print(f"脉冲期间范数: {during_pulse:.2f}")
    print(f"恢复后范数: {post_pulse:.2f}")
    print(f"响应强度: {response_strength:.4f}")
    print(f"恢复偏差: {recovery:.4f}")
    
    # 验收：有响应（响应强度 > 1%）且能恢复（恢复偏差 < 20%）
    has_response = response_strength > 0.01
    can_recover = recovery < 0.2
    
    print()
    print(f"有信号响应: {'PASS' if has_response else 'FAIL'}")
    print(f"脉冲后可恢复: {'PASS' if can_recover else 'FAIL'}")
    
    passed = has_response and can_recover
    print()
    print(f"信号响应测试: {'PASS' if passed else 'FAIL'}")
    return passed


def test_state_space_dimensionality():
    """测试状态空间有效维数（PCA分析）"""
    print()
    print("=" * 70)
    print("测试2: 状态空间有效维数（PCA）")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256,
        slow_dim=64,
        semantic_dim=128,
        physical_dim=64,
        fusion_dim=256,
        meta_cognitive_dim=64,
        chaos_dim=0,
        hidden_dim=128,
        num_hidden_layers=2,
        activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.WORK, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    # 先跑500步暂态
    for i in range(500):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
    
    # 收集2000个状态样本
    n_samples = 2000
    states = []
    for i in range(n_samples):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(500+i)*0.01)
        states.append(E_fast.squeeze(0).cpu().numpy())
    
    states = np.array(states)
    
    # PCA
    states_centered = states - states.mean(axis=0)
    cov = states_centered.T @ states_centered / n_samples
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]  # 降序
    
    # 计算有效维数（参与比）
    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var
    
    # 90% 方差对应的维数
    d_90 = np.searchsorted(cumulative_var, 0.9) + 1
    # 95% 方差对应的维数
    d_95 = np.searchsorted(cumulative_var, 0.95) + 1
    
    # 参与比（Participation Ratio）
    pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()
    
    print(f"状态维度: 256")
    print(f"有效维数 (90%方差): {d_90}")
    print(f"有效维数 (95%方差): {d_95}")
    print(f"参与比 (Participation Ratio): {pr:.2f}")
    
    # 验证：有效维数应该 > 10（说明有丰富的动力学），且 < 256（说明有低维结构）
    has_rich_dynamics = d_90 > 10
    has_low_dim_structure = d_95 < 256
    
    print()
    print(f"丰富动力学 (d_90 > 10): {'PASS' if has_rich_dynamics else 'FAIL'}")
    print(f"低维结构 (d_95 < 256): {'PASS' if has_low_dim_structure else 'FAIL'}")
    
    passed = has_rich_dynamics and has_low_dim_structure
    print()
    print(f"状态空间维数测试: {'PASS' if passed else 'FAIL'}")
    return passed


def test_frequency_spectrum():
    """测试频率特性（功率谱分析）"""
    print()
    print("=" * 70)
    print("测试3: 频率特性分析")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256,
        slow_dim=64,
        semantic_dim=128,
        physical_dim=64,
        fusion_dim=256,
        meta_cognitive_dim=64,
        chaos_dim=0,
        hidden_dim=128,
        num_hidden_layers=2,
        activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.WORK, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    # 先跑1000步暂态
    for i in range(1000):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
    
    # 收集5000步的状态范数时间序列
    n_samples = 5000
    norms = []
    for i in range(n_samples):
        with torch.no_grad():
            E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(1000+i)*0.01)
        norms.append(torch.norm(E_fast).item())
    
    norms = np.array(norms)
    
    # FFT
    dt = 0.01
    freqs = np.fft.rfftfreq(n_samples, dt)
    spectrum = np.abs(np.fft.rfft(norms - norms.mean())) ** 2
    
    # 找主峰
    peak_idx = np.argmax(spectrum[1:]) + 1  # 跳过DC分量
    peak_freq = freqs[peak_idx]
    peak_power = spectrum[peak_idx]
    
    # 计算谱熵（衡量频率分布的复杂性）
    power_norm = spectrum / spectrum.sum()
    power_norm = power_norm[power_norm > 0]
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm))
    max_entropy = np.log2(len(power_norm))
    normalized_entropy = spectral_entropy / max_entropy
    
    print(f"采样率: {1/dt} Hz")
    print(f"主频率: {peak_freq:.3f} Hz")
    print(f"主周期: {1/peak_freq:.3f} s")
    print(f"谱熵(归一化): {normalized_entropy:.4f}")
    
    # 验证：有非零主频（不是纯噪声），且谱熵 > 0.3（不是纯单频）
    has_peak = peak_freq > 0.01
    has_rich_spectrum = normalized_entropy > 0.3
    
    print()
    print(f"有主频率: {'PASS' if has_peak else 'FAIL'}")
    print(f"丰富频谱 (熵>0.3): {'PASS' if has_rich_spectrum else 'FAIL'}")
    
    passed = has_peak and has_rich_spectrum
    print()
    print(f"频率特性测试: {'PASS' if passed else 'FAIL'}")
    return passed


def test_multi_state_cycle():
    """测试多状态切换循环"""
    print()
    print("=" * 70)
    print("测试4: 多状态切换循环")
    print("=" * 70)
    
    device = torch.device('cpu')
    config = FastDynamicsConfig(
        fast_dim=256,
        slow_dim=64,
        semantic_dim=128,
        physical_dim=64,
        fusion_dim=256,
        meta_cognitive_dim=64,
        chaos_dim=0,
        hidden_dim=128,
        num_hidden_layers=2,
        activation="tanh",
    )
    
    system = FastDynamicsSystem(config=config, device=str(device))
    system.initialize()
    system.switch_state(StateMode.REST, force=True)
    
    E_slow = torch.randn(1, 64, device=device) * 0.1
    E_fast = torch.randn(1, 256, device=device)
    E_fast = E_fast / torch.norm(E_fast) * 0.5
    
    # 循环：REST -> WORK -> EXPLORE -> REST -> WORK -> EXPLORE
    cycle = [
        (StateMode.REST, 200),
        (StateMode.WORK, 300),
        (StateMode.EXPLORE, 200),
        (StateMode.REST, 200),
        (StateMode.WORK, 300),
        (StateMode.EXPLORE, 200),
    ]
    
    norms = []
    modes = []
    step_count = 0
    
    for state, duration in cycle:
        system.switch_state(state, force=False)  # 平滑切换
        for i in range(duration):
            with torch.no_grad():
                E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(step_count)*0.01)
            norms.append(torch.norm(E_fast).item())
            modes.append(state.name)
            step_count += 1
    
    norms = np.array(norms)
    
    # 分析每个阶段的特征
    print(f"{'阶段':>10} | {'状态':>10} | {'均值':>8} | {'标准差':>8} | {'CV':>8}")
    print("-" * 55)
    
    idx = 0
    rest_cvs = []
    work_cvs = []
    explore_cvs = []
    
    for state, duration in cycle:
        segment = norms[idx:idx+duration]
        mean_val = segment.mean()
        std_val = segment.std()
        cv_val = std_val / max(mean_val, 1e-6)
        
        print(f"阶段{idx//200 + 1:>4} | {state.name:>10} | {mean_val:>8.2f} | {std_val:>8.2f} | {cv_val:>8.4f}")
        
        if state == StateMode.REST:
            rest_cvs.append(cv_val)
        elif state == StateMode.WORK:
            work_cvs.append(cv_val)
        elif state == StateMode.EXPLORE:
            explore_cvs.append(cv_val)
        
        idx += duration
    
    # 验证：同状态的CV一致性
    rest_consistent = np.std(rest_cvs) / np.mean(rest_cvs) < 0.5 if len(rest_cvs) > 1 else True
    work_consistent = np.std(work_cvs) / np.mean(work_cvs) < 0.5 if len(work_cvs) > 1 else True
    explore_consistent = np.std(explore_cvs) / np.mean(explore_cvs) < 0.5 if len(explore_cvs) > 1 else True
    
    # 验证：状态间有差异
    work_gt_rest = np.mean(work_cvs) > np.mean(rest_cvs) * 1.5
    explore_gt_work = np.mean(explore_cvs) > np.mean(work_cvs) * 1.1
    
    print()
    print(f"REST状态一致性: {'PASS' if rest_consistent else 'FAIL'}")
    print(f"WORK状态一致性: {'PASS' if work_consistent else 'FAIL'}")
    print(f"EXPLORE状态一致性: {'PASS' if explore_consistent else 'FAIL'}")
    print(f"WORK > REST (1.5x): {'PASS' if work_gt_rest else 'FAIL'}")
    print(f"EXPLORE > WORK (1.1x): {'PASS' if explore_gt_work else 'FAIL'}")
    
    passed = (rest_consistent and work_consistent and explore_consistent and 
              work_gt_rest and explore_gt_work)
    print()
    print(f"多状态循环测试: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("P2级验证 - 动力学特性")
    print("=" * 70)
    
    all_passed = True
    results = {}
    
    # 测试1: 信号响应
    try:
        passed = test_signal_response()
        results['signal_response'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['signal_response'] = False
        all_passed = False
    
    # 测试2: 状态空间维数
    try:
        passed = test_state_space_dimensionality()
        results['state_dimensionality'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['state_dimensionality'] = False
        all_passed = False
    
    # 测试3: 频率特性
    try:
        passed = test_frequency_spectrum()
        results['frequency_spectrum'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['frequency_spectrum'] = False
        all_passed = False
    
    # 测试4: 多状态切换循环
    try:
        passed = test_multi_state_cycle()
        results['multi_state_cycle'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['multi_state_cycle'] = False
        all_passed = False
    
    print()
    print("=" * 70)
    print("P2 验证总结")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name:<25}: {'PASS' if passed else 'FAIL'}")
    print()
    print(f"整体结果: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    main()
