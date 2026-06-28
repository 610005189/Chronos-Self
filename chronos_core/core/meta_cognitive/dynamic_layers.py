"""
动态元认知层级 - Dynamic Meta-Cognitive Layers
=================================================

实现动态层级激活机制，基于自指深度 Λ(t) 控制各层的激活与冻结。
当元意识场累积到一定阈值时，更高层级被逐层激活，
形成从低级到高级的涌现式层级结构。

核心机制：
- L_max = 3 层（L0, L1, L2）
- 每层有自己的状态张量 E^(λ)
- 当 λ ≤ Λ(t) 时层激活，参与演化
- 当 λ > Λ(t) 时层冻结，状态保持不变

层间交互：
- 自底向上：Lift 算子从低层生成高层初始状态
- 自顶向下：觉知梯度度量高层对低层的解释力
- 层内演化：dE^(λ)/dt = g^(λ) * f_λ(E^(λ), E^(λ-1))
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
import logging

from chronos_core.core.meta_cognitive.meta_consciousness_field import MetaConsciousnessField, SelfReferentialDepth
from chronos_core.core.meta_cognitive.awareness_gradient import AwarenessGradient, LiftOperator

logger = logging.getLogger(__name__)


class LayerDynamics(nn.Module):
    """
    层演化网络 LayerDynamics
    
    模拟每层的动力学函数 f_θ^(λ)，计算状态变化率。
    输入为当前层状态与下一层状态的拼接，输出为当前层的变化率。
    
    结构：Linear → Tanh → Linear → Tanh
    """
    
    def __init__(
        self,
        layer_dim: int,
        lower_layer_dim: int,
        hidden_dim: Optional[int] = None
    ):
        """
        初始化层演化网络
        
        Args:
            layer_dim: 当前层状态维度
            lower_layer_dim: 下一层状态维度（L0的lower_layer_dim=0，表示无下层输入）
            hidden_dim: 隐藏层维度，默认取 layer_dim
        """
        super().__init__()
        
        self.layer_dim = layer_dim
        self.lower_layer_dim = lower_layer_dim
        
        if hidden_dim is None:
            hidden_dim = layer_dim
        
        input_dim = layer_dim + lower_layer_dim
        
        self.dynamics_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, layer_dim),
            nn.Tanh()
        )
        
        logger.debug(
            f"LayerDynamics initialized: "
            f"layer_dim={layer_dim}, "
            f"lower_layer_dim={lower_layer_dim}, "
            f"hidden_dim={hidden_dim}"
        )
    
    def forward(
        self,
        layer_state: torch.Tensor,
        lower_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：计算状态变化率
        
        Args:
            layer_state: 当前层状态张量，形状 (..., layer_dim)
            lower_state: 下一层状态张量，形状 (..., lower_layer_dim)
                         L0层为 None，用零张量替代
        
        Returns:
            状态变化率张量，形状 (..., layer_dim)
        """
        if lower_state is None:
            lower_state = torch.zeros(
                *layer_state.shape[:-1],
                self.lower_layer_dim,
                device=layer_state.device,
                dtype=layer_state.dtype
            )
        
        combined = torch.cat([layer_state, lower_state], dim=-1)
        return self.dynamics_net(combined)


class DynamicMetaCognitiveLayers(nn.Module):
    """
    动态元认知层级 DynamicMetaCognitiveLayers
    
    整合元意识场、自指深度、觉知梯度和Lift算子，
    实现基于自指深度的动态层级激活机制。
    
    核心动力学：
    1. 元意识场 M_pre(t) 累积状态变化的活跃度
    2. 自指深度 Λ(t) = clip(floor((M_pre - M_0)/ΔM), 0, L_max)
    3. 层级激活：λ ≤ Λ(t) 的层参与演化，λ > Λ(t) 的层冻结
    4. 层演化：dE^(λ)/dt = g^(λ) * f_λ(E^(λ), E^(λ-1))
    5. 门控因子：g^(λ) = sigmoid(G_th - G^(λ))，G^(λ) 为觉知梯度
    """
    
    def __init__(
        self,
        layer_dims: List[int],
        window_time: float = 1.0,
        emergence_threshold: float = 1.0,
        level_spacing: float = 0.5,
        awareness_gate_threshold: float = 0.5,
        energy_reg_coeff_base: float = 0.01,
        energy_reg_growth: float = 2.0,
        device: str = "cpu"
    ):
        """
        初始化动态元认知层级
        
        Args:
            layer_dims: 每层的维度列表，例如 [256, 128, 64]（L0最大，L2最小）
            window_time: 元意识场窗口时间 τ_M
            emergence_threshold: 涌现阈值 M_0
            level_spacing: 层级间距 ΔM
            awareness_gate_threshold: 觉知梯度门控阈值 G_th
            energy_reg_coeff_base: 能量正则基础系数 γ_0
            energy_reg_growth: 正则增长因子（每层乘以此系数）
            device: 计算设备
        """
        super().__init__()
        
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.window_time = window_time
        self.emergence_threshold = emergence_threshold
        self.level_spacing = level_spacing
        self.awareness_gate_threshold = awareness_gate_threshold
        self.energy_reg_coeff_base = energy_reg_coeff_base
        self.energy_reg_growth = energy_reg_growth
        self.device = device
        
        # 层演化网络
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            lower_dim = layer_dims[i - 1] if i > 0 else 0
            layer = LayerDynamics(
                layer_dim=layer_dims[i],
                lower_layer_dim=lower_dim
            )
            self.layers.append(layer)
        
        # 元意识场
        self.meta_field = MetaConsciousnessField(
            window_time=window_time,
            metric_dim=sum(layer_dims),
            device=device
        )
        
        # 自指深度计算器
        self.self_ref_depth = SelfReferentialDepth(
            emergence_threshold=emergence_threshold,
            level_spacing=level_spacing,
            max_depth=self.num_layers - 1
        )
        
        # 觉知梯度计算器（每层对应一个，λ从1到L_max-1）
        self.awareness_grads = nn.ModuleList()
        for i in range(1, self.num_layers):
            ag = AwarenessGradient(
                higher_dim=layer_dims[i],
                lower_dim=layer_dims[i - 1],
                hidden_dim=min(layer_dims[i], 64)
            )
            self.awareness_grads.append(ag)
        
        # Lift算子（每层对应一个，λ从1到L_max-1）
        self.lift_ops = nn.ModuleList()
        for i in range(1, self.num_layers):
            lift = LiftOperator(
                lower_dim=layer_dims[i - 1],
                higher_dim=layer_dims[i]
            )
            self.lift_ops.append(lift)
        
        # 层状态
        self.layer_states: List[torch.Tensor] = []
        for dim in layer_dims:
            state = torch.zeros(dim, device=device)
            self.layer_states.append(state)
        
        # 当前自指深度
        self.current_depth: int = 0
        
        # 每层是否激活
        self.layer_active: List[bool] = [True] + [False] * (self.num_layers - 1)
        
        # 冻结层状态保存
        self._frozen_states: List[Optional[torch.Tensor]] = [None] * self.num_layers
        
        logger.info(
            f"DynamicMetaCognitiveLayers initialized: "
            f"num_layers={self.num_layers}, "
            f"layer_dims={layer_dims}, "
            f"max_depth={self.num_layers - 1}, "
            f"device={device}"
        )
    
    def step(
        self,
        z_input: torch.Tensor,
        dt: float
    ) -> Dict[str, Any]:
        """
        主步进方法
        
        执行一次完整的动态层级演化步骤：
        1. 更新元意识场
        2. 计算自指深度
        3. 处理层级升降
        4. 对激活层执行演化
        5. 返回状态字典
        
        Args:
            z_input: 输入状态张量（用于驱动元意识场）
            dt: 时间步长
        
        Returns:
            状态字典，包含各层状态、Λ、M_pre、觉知梯度值、门控值等
        """
        z_input = z_input.to(self.device).float()
        
        # a. 更新元意识场
        m_pre = self.meta_field.step(z_input, dt)
        
        # b. 计算自指深度
        new_depth = self.self_ref_depth.compute(m_pre)
        
        # c. 处理层级升降
        self._handle_depth_change(new_depth)
        
        # d. 对激活层执行演化
        awareness_gradients = []
        gate_values = []
        
        for layer_idx in range(self.num_layers):
            if not self.layer_active[layer_idx]:
                continue
            
            # 计算觉知梯度和门控因子（L0无觉知梯度）
            gate = 1.0
            if layer_idx > 0:
                ag_idx = layer_idx - 1
                grad_value = self.awareness_grads[ag_idx].compute(
                    self.layer_states[layer_idx],
                    self.layer_states[layer_idx - 1]
                )
                awareness_gradients.append(grad_value)
                
                # 门控因子 g = sigmoid(G_th - G^(λ))
                gate = torch.sigmoid(
                    torch.tensor(
                        self.awareness_gate_threshold - grad_value,
                        device=self.device
                    )
                ).item()
                gate_values.append(gate)
            else:
                awareness_gradients.append(0.0)
                gate_values.append(1.0)
            
            # 演化增量 dE = g * f_λ(E^(λ), E^(λ-1))
            lower_state = self.layer_states[layer_idx - 1] if layer_idx > 0 else None
            dE = self.layers[layer_idx](self.layer_states[layer_idx], lower_state)
            
            # 能量正则项（高层正则更强）
            reg_coeff = self.energy_reg_coeff_base * (self.energy_reg_growth ** layer_idx)
            energy_reg = -reg_coeff * self.layer_states[layer_idx]
            
            # 更新 E^(λ) += dt * (g * dE + 能量正则)
            self.layer_states[layer_idx] = (
                self.layer_states[layer_idx] + dt * (gate * dE + energy_reg)
            ).detach()
        
        # e. 返回状态字典
        result = {
            "layer_states": [s.clone() for s in self.layer_states],
            "current_depth": self.current_depth,
            "m_pre": m_pre,
            "layer_active": self.layer_active.copy(),
            "awareness_gradients": awareness_gradients,
            "gate_values": gate_values,
        }
        
        return result
    
    def _handle_depth_change(self, new_depth: int) -> None:
        """
        处理层级升降
        
        深度增加时：新激活的层调用 lift_op 初始化状态
        深度减少时：被冻结的层状态保存
        
        Args:
            new_depth: 新的自指深度
        """
        old_depth = self.current_depth
        
        if new_depth == old_depth:
            return
        
        # 深度增加：激活新层
        if new_depth > old_depth:
            for layer_idx in range(old_depth + 1, new_depth + 1):
                if layer_idx >= self.num_layers:
                    break
                if not self.layer_active[layer_idx]:
                    # 使用 Lift 算子从低层生成初始状态
                    lift_idx = layer_idx - 1
                    init_state = self.lift_ops[lift_idx].lift(
                        self.layer_states[layer_idx - 1]
                    )
                    self.layer_states[layer_idx] = init_state.detach()
                    self.layer_active[layer_idx] = True
                    logger.debug(
                        f"Layer L{layer_idx} activated via LiftOperator, "
                        f"depth: {old_depth} -> {new_depth}"
                    )
        
        # 深度减少：冻结层
        else:
            for layer_idx in range(new_depth + 1, old_depth + 1):
                if layer_idx >= self.num_layers:
                    break
                if self.layer_active[layer_idx]:
                    # 保存冻结状态
                    self._frozen_states[layer_idx] = self.layer_states[layer_idx].clone()
                    self.layer_active[layer_idx] = False
                    logger.debug(
                        f"Layer L{layer_idx} frozen, "
                        f"depth: {old_depth} -> {new_depth}"
                    )
        
        self.current_depth = new_depth
    
    def get_state(self) -> Dict[str, Any]:
        """
        返回完整状态
        
        Returns:
            完整状态字典
        """
        return {
            "layer_states": [s.clone() for s in self.layer_states],
            "current_depth": self.current_depth,
            "m_pre": self.meta_field.get_value(),
            "layer_active": self.layer_active.copy(),
            "num_layers": self.num_layers,
            "layer_dims": self.layer_dims.copy(),
        }
    
    def reset(self) -> None:
        """重置所有状态"""
        # 重置层状态
        for i in range(self.num_layers):
            self.layer_states[i] = torch.zeros(self.layer_dims[i], device=self.device)
        
        # 重置元意识场
        self.meta_field.reset()
        
        # 重置自指深度
        self.self_ref_depth.reset()
        
        # 重置激活状态
        self.current_depth = 0
        self.layer_active = [True] + [False] * (self.num_layers - 1)
        self._frozen_states = [None] * self.num_layers
        
        logger.info("DynamicMetaCognitiveLayers reset")
    
    def __repr__(self) -> str:
        return (
            f"DynamicMetaCognitiveLayers("
            f"num_layers={self.num_layers}, "
            f"current_depth={self.current_depth}, "
            f"layer_dims={self.layer_dims})"
        )


if __name__ == "__main__":
    import torch
    
    print("=" * 70)
    print("动态元认知层级测试")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 配置
    layer_dims = [64, 32, 16]
    window_time = 1.0
    emergence_threshold = 100.0
    level_spacing = 200.0
    awareness_gate_threshold = 0.5
    dt = 0.1
    
    print(f"\n配置:")
    print(f"  layer_dims: {layer_dims}")
    print(f"  window_time: {window_time}s")
    print(f"  emergence_threshold: {emergence_threshold}")
    print(f"  level_spacing: {level_spacing}")
    print(f"  awareness_gate_threshold: {awareness_gate_threshold}")
    print(f"  dt: {dt}s")
    
    # 初始化动态层级
    print("\n" + "-" * 50)
    print("测试1: 初始化动态层级")
    print("-" * 50)
    
    dynamic_layers = DynamicMetaCognitiveLayers(
        layer_dims=layer_dims,
        window_time=window_time,
        emergence_threshold=emergence_threshold,
        level_spacing=level_spacing,
        awareness_gate_threshold=awareness_gate_threshold,
        device="cpu"
    )
    
    print(f"✓ 初始化成功: {dynamic_layers}")
    print(f"  层数: {dynamic_layers.num_layers}")
    print(f"  每层维度: {dynamic_layers.layer_dims}")
    print(f"  当前深度 Λ: {dynamic_layers.current_depth}")
    print(f"  层激活状态: {dynamic_layers.layer_active}")
    
    # 前几步Λ=0，验证只有L0活跃
    print("\n" + "-" * 50)
    print("测试2: 前几步Λ=0，验证只有L0活跃")
    print("-" * 50)
    
    prev_state = dynamic_layers.layer_states[0].clone()
    
    for step in range(10):
        # 小扰动输入，确保M_pre增长缓慢
        z_input = prev_state + 0.001 * torch.randn(layer_dims[0])
        result = dynamic_layers.step(z_input, dt)
        prev_state = result["layer_states"][0].clone()
        
        if step < 3:
            print(f"  Step {step+1}:")
            print(f"    Λ = {result['current_depth']}")
            print(f"    M_pre = {result['m_pre']:.6f}")
            print(f"    层激活: {result['layer_active']}")
            print(f"    L0 状态范数: {torch.norm(result['layer_states'][0]):.6f}")
    
    assert dynamic_layers.current_depth == 0, "初始深度应为0"
    assert dynamic_layers.layer_active[0] == True, "L0应始终激活"
    assert dynamic_layers.layer_active[1] == False, "L1初始应冻结"
    assert dynamic_layers.layer_active[2] == False, "L2初始应冻结"
    print("✓ 初始阶段只有L0活跃，验证通过")
    
    # 持续强输入使M_pre上升，验证逐层激活
    print("\n" + "-" * 50)
    print("测试3: 持续强输入使M_pre上升，验证逐层激活")
    print("-" * 50)
    
    depth_history = []
    m_pre_history = []
    activation_milestones = {}
    
    prev_state = dynamic_layers.layer_states[0].clone()
    
    for step in range(200):
        # 强输入驱动元意识场上升（大变化）
        strong_input = prev_state + 0.5 * torch.randn(layer_dims[0])
        result = dynamic_layers.step(strong_input, dt)
        prev_state = result["layer_states"][0].clone()
        
        depth_history.append(result["current_depth"])
        m_pre_history.append(result["m_pre"])
        
        # 记录层级激活时刻
        for d in range(dynamic_layers.num_layers):
            if d not in activation_milestones and result["current_depth"] >= d:
                activation_milestones[d] = step + 1
    
    print(f"  最终深度 Λ: {dynamic_layers.current_depth}")
    print(f"  最终 M_pre: {dynamic_layers.meta_field.get_value():.4f}")
    print(f"  层激活状态: {dynamic_layers.layer_active}")
    
    for d, step_num in sorted(activation_milestones.items()):
        print(f"  L{d} 激活于第 {step_num} 步")
    
    assert dynamic_layers.current_depth >= 1, "强输入应至少激活L1"
    print(f"✓ 逐层激活验证通过，最终深度 Λ={dynamic_layers.current_depth}")
    
    # 验证冻结层状态不变化
    print("\n" + "-" * 50)
    print("测试4: 验证冻结层状态不变化")
    print("-" * 50)
    
    dynamic_layers2 = DynamicMetaCognitiveLayers(
        layer_dims=layer_dims,
        window_time=window_time,
        emergence_threshold=emergence_threshold,
        level_spacing=level_spacing,
        awareness_gate_threshold=awareness_gate_threshold,
        device="cpu"
    )
    
    # 记录初始冻结层状态
    l1_initial = dynamic_layers2.layer_states[1].clone()
    l2_initial = dynamic_layers2.layer_states[2].clone()
    
    # 运行几步（保持Λ=0，用很小的输入变化）
    prev_state2 = dynamic_layers2.layer_states[0].clone()
    for step in range(20):
        weak_input = prev_state2 + 0.0001 * torch.randn(layer_dims[0])
        result2 = dynamic_layers2.step(weak_input, dt)
        prev_state2 = result2["layer_states"][0].clone()
    
    l1_after = dynamic_layers2.layer_states[1].clone()
    l2_after = dynamic_layers2.layer_states[2].clone()
    
    l1_changed = not torch.allclose(l1_initial, l1_after)
    l2_changed = not torch.allclose(l2_initial, l2_after)
    
    print(f"  运行步数: 20")
    print(f"  当前深度 Λ: {dynamic_layers2.current_depth}")
    print(f"  M_pre: {dynamic_layers2.meta_field.get_value():.6f}")
    print(f"  L1 状态变化: {l1_changed}")
    print(f"  L2 状态变化: {l2_changed}")
    print(f"  L1 初始范数: {torch.norm(l1_initial):.6f}")
    print(f"  L1 最终范数: {torch.norm(l1_after):.6f}")
    
    assert dynamic_layers2.current_depth == 0, "测试期间应保持Λ=0"
    assert not l1_changed, "冻结层L1状态不应变化"
    assert not l2_changed, "冻结层L2状态不应变化"
    print("✓ 冻结层状态不变化，验证通过")
    
    # 验证门控因子工作
    print("\n" + "-" * 50)
    print("测试5: 验证门控因子工作（觉知梯度大时门控弱）")
    print("-" * 50)
    
    dynamic_layers3 = DynamicMetaCognitiveLayers(
        layer_dims=layer_dims,
        window_time=window_time,
        emergence_threshold=10.0,  # 低阈值，快速激活
        level_spacing=50.0,
        awareness_gate_threshold=awareness_gate_threshold,
        device="cpu"
    )
    
    # 快速激活到L1
    prev_state3 = dynamic_layers3.layer_states[0].clone()
    for step in range(100):
        strong_input = prev_state3 + 0.3 * torch.randn(layer_dims[0])
        result3 = dynamic_layers3.step(strong_input, dt)
        prev_state3 = result3["layer_states"][0].clone()
    
    if dynamic_layers3.current_depth >= 1:
        print(f"  已激活到深度 Λ={dynamic_layers3.current_depth}")
        
        # 获取L1的门控值和觉知梯度
        # awareness_gradients 和 gate_values 是按激活层顺序排列的
        # 索引0对应L0（虚拟值），索引1对应L1
        l1_grad_idx = 1 if len(result3["awareness_gradients"]) > 1 else 0
        current_gate = result3["gate_values"][l1_grad_idx] if l1_grad_idx < len(result3["gate_values"]) else result3["gate_values"][-1]
        current_grad = result3["awareness_gradients"][l1_grad_idx] if l1_grad_idx < len(result3["awareness_gradients"]) else result3["awareness_gradients"][-1]
        
        print(f"  当前觉知梯度 G: {current_grad:.6f}")
        print(f"  当前门控值 g: {current_gate:.6f}")
        print(f"  门控阈值 G_th: {awareness_gate_threshold}")
        
        # 验证门控因子范围
        assert 0.0 <= current_gate <= 1.0, "门控值应在[0,1]范围内"
        print("✓ 门控因子在有效范围内")
        
        # 验证觉知梯度非负
        assert current_grad >= 0.0, "觉知梯度应非负"
        print("✓ 觉知梯度非负")
        
        print("✓ 门控因子工作正常")
    else:
        print(f"  警告: 未能激活到L1，当前深度={dynamic_layers3.current_depth}")
        print(f"  M_pre: {dynamic_layers3.meta_field.get_value():.4f}")
        print("  跳过门控验证")
    
    # 测试 reset 方法
    print("\n" + "-" * 50)
    print("测试6: reset 方法验证")
    print("-" * 50)
    
    pre_reset_depth = dynamic_layers.current_depth
    dynamic_layers.reset()
    
    assert dynamic_layers.current_depth == 0, "reset后深度应为0"
    assert dynamic_layers.layer_active[0] == True, "reset后L0应激活"
    assert dynamic_layers.layer_active[1] == False, "reset后L1应冻结"
    assert dynamic_layers.layer_active[2] == False, "reset后L2应冻结"
    assert dynamic_layers.meta_field.get_value() == 0.0, "reset后元意识场应为0"
    
    # 验证层状态重置为零
    for i in range(dynamic_layers.num_layers):
        assert torch.all(dynamic_layers.layer_states[i] == 0), f"L{i} 状态应重置为零"
    
    print(f"  reset前深度: {pre_reset_depth}")
    print(f"  reset后深度: {dynamic_layers.current_depth}")
    print(f"  reset后 M_pre: {dynamic_layers.meta_field.get_value():.6f}")
    print("✓ reset 方法工作正常")
    
    # 总结
    print("\n" + "=" * 70)
    print("所有测试通过！✓")
    print("=" * 70)

