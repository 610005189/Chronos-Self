"""
P1级验证 - 核心指标验证
"""

import torch
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from chronos_core.core.fast_dynamics import FastDynamicsSystem, FastDynamicsConfig
from chronos_core.core.state_controller import StateMode


def test_initial_condition_robustness():
    """测试不同初始条件下的鲁棒性"""
    print("=" * 70)
    print("测试1: 初始条件鲁棒性")
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
    
    print("测试 5 个不同初始状态...")
    print(f"{'Seed':>6} | {'mean_norm':>10} | {'std_norm':>10} | {'CV':>8} | {'max_norm':>10} | 状态")
    print("-" * 70)
    
    all_bounded = True
    cvs = []
    
    for seed in range(5):
        torch.manual_seed(seed)
        E_fast = torch.randn(1, 256, device=device)
        E_fast = E_fast / torch.norm(E_fast) * (0.1 + seed * 0.2)  # 不同初始范数
        
        norms = []
        for i in range(1000):
            with torch.no_grad():
                E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
            norms.append(torch.norm(E_fast).item())
        
        norms = np.array(norms)
        late_norms = norms[500:]
        mean_norm = late_norms.mean()
        std_norm = late_norms.std()
        cv_norm = std_norm / max(mean_norm, 1e-6)
        max_norm = norms.max()
        
        bounded = max_norm < 200
        if not bounded:
            all_bounded = False
        
        cvs.append(cv_norm)
        status = "PASS" if bounded else "FAIL"
        print(f"{seed:>6} | {mean_norm:>10.2f} | {std_norm:>10.2f} | {cv_norm:>8.4f} | {max_norm:>10.2f} | {status}")
    
    cv_mean = np.mean(cvs)
    cv_std = np.std(cvs)
    print()
    print(f"平均CV: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"所有初始条件有界: {'PASS' if all_bounded else 'FAIL'}")
    print(f"CV一致性(CV的CV < 0.5): {'PASS' if cv_std/max(cv_mean,1e-6) < 0.5 else 'FAIL'}")
    
    passed = all_bounded and cv_std/max(cv_mean,1e-6) < 0.5
    print()
    print(f"初始条件鲁棒性: {'PASS' if passed else 'FAIL'}")
    return passed


def test_spectral_constraint_effectiveness():
    """测试逐层谱约束的有效性"""
    print()
    print("=" * 70)
    print("测试2: 逐层谱约束有效性")
    print("=" * 70)
    
    device = torch.device('cpu')
    
    print("测试不同 target_spectral_norm 下的权重谱范数...")
    print(f"{'tSN':>6} | {'Layer1 σ_max':>12} | {'Layer2 σ_max':>12} | {'out σ_max':>12}")
    print("-" * 55)
    
    tSN_values = [1.0, 1.5, 2.0, 3.0]
    all_close = True
    
    for tSN in tSN_values:
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
            target_spectral_norm=tSN,
        )
        
        system = FastDynamicsSystem(config=config, device=str(device))
        system.initialize()
        
        # 计算各层权重的谱范数（最大奇异值）
        evo_fn = system.dynamics_fn.evolution_fn
        layers = []
        if hasattr(evo_fn, 'linear_layers'):
            for layer in evo_fn.linear_layers:
                if hasattr(layer, 'weight'):
                    layers.append(layer.weight.data)
        elif hasattr(evo_fn, 'layers'):
            for layer in evo_fn.layers:
                if hasattr(layer, 'weight'):
                    layers.append(layer.weight.data)
        
        sigma_maxes = []
        for i, W in enumerate(layers[:3]):  # 只看前3层
            s = torch.linalg.svdvals(W)
            sigma_max = s.max().item()
            sigma_maxes.append(sigma_max)
        
        # 检查是否接近目标值
        for sigma in sigma_maxes:
            if abs(sigma - tSN) / tSN > 0.3:  # 允许30%误差
                all_close = False
        
        while len(sigma_maxes) < 3:
            sigma_maxes.append(float('nan'))
        
        print(f"{tSN:>6.1f} | {sigma_maxes[0]:>12.4f} | {sigma_maxes[1]:>12.4f} | {sigma_maxes[2]:>12.4f}")
    
    print()
    print(f"谱约束有效(±30%): {'PASS' if all_close else 'FAIL'}")
    return all_close


def test_decay_controllability():
    """测试衰减率的可控性（状态范数随decay_rate变化）"""
    print()
    print("=" * 70)
    print("测试3: 衰减率可控性")
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
    
    decay_rates = [0.1, 0.3, 0.5, 0.8, 1.0]
    print(f"测试不同 decay_rate 下的状态范数...")
    print(f"{'decay':>6} | {'mean_norm':>10} | {'max_norm':>10} | 状态")
    print("-" * 45)
    
    norms_list = []
    all_bounded = True
    
    for decay in decay_rates:
        # 手动设置参数
        from chronos_core.core.state_controller import StateParameters
        params = StateParameters(
            decay_rate=decay,
            gamma=0.0,
            dynamics_scale=7.0,
            noise_scale=0.00001,
            ei_ratio=4.0,
            alpha=0.0,
            e_target=0.0,
            state_norm_threshold=200.0,
            state_norm_clip=0.0,
            max_gradient_norm=200.0,
            target_spectral_norm=1.5,
        )
        system.state_controller._transition_state.current_params = params
        system.state_controller._transition_state.target_params = params
        system._apply_state_params(params)
        
        torch.manual_seed(42)
        E_fast = torch.randn(1, 256, device=device)
        E_fast = E_fast / torch.norm(E_fast) * 0.5
        
        norms = []
        for i in range(1000):
            with torch.no_grad():
                E_fast = system.step(E_fast, E_slow, dt=0.01, t=float(i)*0.01)
            norms.append(torch.norm(E_fast).item())
        
        norms = np.array(norms)
        late_norms = norms[500:]
        mean_norm = late_norms.mean()
        max_norm = norms.max()
        
        norms_list.append(mean_norm)
        bounded = max_norm < 200
        if not bounded:
            all_bounded = False
        
        print(f"{decay:>6.1f} | {mean_norm:>10.2f} | {max_norm:>10.2f} | {'PASS' if bounded else 'FAIL'}")
    
    # 验证：decay越大，norm应该越小（单调性）
    is_monotonic = all(norms_list[i] >= norms_list[i+1] for i in range(len(norms_list)-1))
    print()
    print(f"衰减率-范数单调性: {'PASS' if is_monotonic else 'FAIL'} (decay越大，norm越小)")
    print(f"所有配置有界: {'PASS' if all_bounded else 'FAIL'}")
    
    passed = is_monotonic and all_bounded
    print()
    print(f"衰减率可控性: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    print("\n" + "=" * 70)
    print("P1级验证 - 核心指标")
    print("=" * 70)
    
    all_passed = True
    results = {}
    
    # 测试1: 初始条件鲁棒性
    try:
        passed = test_initial_condition_robustness()
        results['initial_condition'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['initial_condition'] = False
        all_passed = False
    
    # 测试2: 逐层谱约束有效性
    try:
        passed = test_spectral_constraint_effectiveness()
        results['spectral_constraint'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['spectral_constraint'] = False
        all_passed = False
    
    # 测试3: 衰减率可控性
    try:
        passed = test_decay_controllability()
        results['decay_controllability'] = passed
        all_passed = all_passed and passed
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        results['decay_controllability'] = False
        all_passed = False
    
    print()
    print("=" * 70)
    print("P1 验证总结")
    print("=" * 70)
    for name, passed in results.items():
        print(f"  {name:<25}: {'PASS' if passed else 'FAIL'}")
    print()
    print(f"整体结果: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    main()
