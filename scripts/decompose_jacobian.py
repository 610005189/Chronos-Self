import sys
sys.path.insert(0, 'd:/Projects/Chronos-Self')

import torch
from chronos_core.core.fast_dynamics import FastDynamicsFunction, FastDynamicsConfig

torch.manual_seed(42)

fast_dim = 64
slow_dim = 16

config = FastDynamicsConfig(
    fast_dim=fast_dim,
    slow_dim=slow_dim,
    semantic_dim=slow_dim,
    physical_dim=slow_dim,
    fusion_dim=fast_dim,
    meta_cognitive_dim=4,
    chaos_dim=16,
    hidden_dim=128,
    num_hidden_layers=2,
    activation="tanh",
    gamma=0.5,
    decay_rate=0.85,
    dynamics_scale=0.1,
)

dyn_fn = FastDynamicsFunction(config=config)
dyn_fn.eval()

y = torch.randn(1, fast_dim) * 0.01
y.requires_grad_(True)
t = torch.tensor([0.0])

print("=== 分解 Jacobian 的组成部分 ===")

# 1. 演化函数网络对 Jacobian 的贡献
def evolution_fn_only(x):
    """只计算演化函数网络的输出"""
    # 构建输入向量（与 forward 方法相同）
    input_parts = [
        x,  # 快变量状态
        torch.zeros(1, config.slow_dim),  # 慢变量（零填充）
        torch.zeros(1, config.semantic_dim),  # 语义流（零填充）
        torch.zeros(1, config.physical_dim),  # 物理流（零填充）
        torch.zeros(1, config.fusion_dim if not config.fourier_enabled else config.fourier_n_features * 2),  # 融合表征
        torch.zeros(1, config.meta_cognitive_dim),  # 元认知
        torch.zeros(1, config.chaos_dim),  # 混沌注入
        torch.tensor([[0.0]]).expand(1, -1),  # 时间
    ]
    input_vector = torch.cat(input_parts, dim=-1)
    F_output = dyn_fn.evolution_fn(input_vector) * config.dynamics_scale
    return F_output.squeeze(0)

J_evolution = torch.autograd.functional.jacobian(evolution_fn_only, y, create_graph=False)
if J_evolution.dim() == 4:
    J_evolution = J_evolution.squeeze(0).squeeze(1)
elif J_evolution.dim() == 3:
    J_evolution = J_evolution.squeeze(1)

print(f"\n1. 演化函数网络的 Jacobian:")
print(f"   Shape: {J_evolution.shape}")
print(f"   范数: {torch.norm(J_evolution).item():.6f}")
print(f"   对角线（前10个）: {torch.diag(J_evolution)[:10]}")

# 2. 衰减层对 Jacobian 的贡献（常数矩阵 W）
W_decay = dyn_fn.decay_layer.weight
print(f"\n2. 衰减层的权重矩阵 W:")
print(f"   Shape: {W_decay.shape}")
print(f"   范数: {torch.norm(W_decay).item():.6f}")
print(f"   对角线（前10个）: {W_decay.diag()[:10]}")

# 3. gamma 耗散项的贡献（常数矩阵 -gamma * I）
gamma_matrix = -config.gamma * torch.eye(fast_dim)
print(f"\n3. gamma 耗散项的贡献:")
print(f"   -gamma * I 的对角线: {-config.gamma}")

# 4. 完整 Jacobian
def full_dynamics(x):
    return dyn_fn.forward(t, x).squeeze(0)

J_full = torch.autograd.functional.jacobian(full_dynamics, y, create_graph=False)
if J_full.dim() == 4:
    J_full = J_full.squeeze(0).squeeze(1)
elif J_full.dim() == 3:
    J_full = J_full.squeeze(1)

print(f"\n4. 完整 Jacobian:")
print(f"   Shape: {J_full.shape}")
print(f"   范数: {torch.norm(J_full).item():.6f}")
print(f"   对角线（前10个）: {torch.diag(J_full)[:10]}")

# 5. 验证组成部分的加和
# 完整 Jacobian 应该 ≈ J_evolution + W_decay + gamma_matrix
J_expected = J_evolution + W_decay + gamma_matrix
print(f"\n5. 验证: J_evolution + W_decay + gamma_matrix")
print(f"   范数: {torch.norm(J_expected).item():.6f}")
print(f"   对角线（前10个）: {torch.diag(J_expected)[:10]}")

# 6. 检查额外的范数依赖衰减项
# 在 forward 方法中，第 597-605 行有额外的范数依赖衰减
# 这也会影响 Jacobian
print(f"\n6. 检查额外的范数依赖衰减项:")
y_test = y.clone().detach()
norm = torch.norm(y_test, dim=-1, keepdim=True).item()
threshold_half = config.state_norm_threshold * 0.5
extra_decay_factor = max(0, min(2, (norm - threshold_half) / threshold_half if norm > threshold_half else 0))
print(f"   当前状态范数: {norm:.6f}")
print(f"   阈值的一半: {threshold_half:.6f}")
print(f"   extra_decay_factor: {extra_decay_factor:.6f}")
if extra_decay_factor > 0:
    print(f"   范数依赖衰减项对 Jacobian 的贡献: -{extra_decay_factor:.6f} * I")

# 7. 检查梯度裁剪的影响
print(f"\n7. 检查梯度裁剪的影响:")
dydt = dyn_fn.forward(t, y)
dydt_norm = torch.norm(dydt).item()
print(f"   dydt 范数: {dydt_norm:.6f}")
print(f"   max_gradient_norm: {config.max_gradient_norm:.6f}")
if dydt_norm > config.max_gradient_norm:
    print(f"   梯度被裁剪！")
else:
    print(f"   梯度未被裁剪")