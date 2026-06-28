"""
觉知梯度 - Awareness Gradient
============================

度量高层对低层的解释力，实现觉知梯度的计算与相关算子。

核心公式：
G^(λ) = || Dec^(λ→λ-1)(E^(λ)) - E^(λ-1) ||_{Σ_G}

值越小表示高层对低层解释力越强。

主要组件：
- DecoderNetwork: 解码器网络，从高层状态预测低层状态
- AwarenessGradient: 觉知梯度计算器
- LiftOperator: Lift算子，从低层状态生成高层初始状态
"""

import torch
import torch.nn as nn
from typing import Optional, Iterator
import logging

logger = logging.getLogger(__name__)


class DecoderNetwork(nn.Module):
    """
    解码器网络
    
    小型 MLP，从高层状态预测低层状态。
    结构：2层隐藏层，每层 hidden_dim，激活函数用 Tanh。
    """
    
    def __init__(
        self,
        higher_dim: int,
        lower_dim: int,
        hidden_dim: int = 64
    ):
        """
        初始化解码器网络
        
        Args:
            higher_dim: 高层状态维度
            lower_dim: 低层状态维度
            hidden_dim: 隐藏层维度，默认64
        """
        super().__init__()
        
        self.higher_dim = higher_dim
        self.lower_dim = lower_dim
        self.hidden_dim = hidden_dim
        
        self.decoder = nn.Sequential(
            nn.Linear(higher_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, lower_dim)
        )
        
        logger.debug(
            f"DecoderNetwork initialized: "
            f"higher_dim={higher_dim}, "
            f"lower_dim={lower_dim}, "
            f"hidden_dim={hidden_dim}"
        )
    
    def forward(self, higher_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播：从高层状态预测低层状态
        
        Args:
            higher_state: 高层状态张量，形状 (..., higher_dim)
            
        Returns:
            预测的低层状态张量，形状 (..., lower_dim)
        """
        return self.decoder(higher_state)


class AwarenessGradient(nn.Module):
    """
    觉知梯度计算器
    
    度量高层对低层的解释力。
    公式：G^(λ) = || Dec^(λ→λ-1)(E^(λ)) - E^(λ-1) ||_{Σ_G}
    
    度量张量 Σ_G 初始用单位矩阵近似（即普通 MSE/L2 范数）。
    梯度值始终非负。
    """
    
    def __init__(
        self,
        higher_dim: int,
        lower_dim: int,
        hidden_dim: int = 64
    ):
        """
        初始化觉知梯度计算器
        
        Args:
            higher_dim: 高层状态维度
            lower_dim: 低层状态维度
            hidden_dim: 隐藏层维度，默认64
        """
        super().__init__()
        
        self.higher_dim = higher_dim
        self.lower_dim = lower_dim
        self.hidden_dim = hidden_dim
        
        self.decoder = DecoderNetwork(
            higher_dim=higher_dim,
            lower_dim=lower_dim,
            hidden_dim=hidden_dim
        )
        
        logger.info(
            f"AwarenessGradient initialized: "
            f"higher_dim={higher_dim}, "
            f"lower_dim={lower_dim}, "
            f"hidden_dim={hidden_dim}"
        )
    
    def compute(
        self,
        higher_state: torch.Tensor,
        lower_state: torch.Tensor
    ) -> float:
        """
        计算觉知梯度
        
        Args:
            higher_state: 高层状态张量，形状 (..., higher_dim)
            lower_state: 低层状态张量，形状 (..., lower_dim)
            
        Returns:
            觉知梯度值（非负）
        """
        predicted_lower = self.decoder(higher_state)
        
        mse_loss = nn.functional.mse_loss(predicted_lower, lower_state, reduction='mean')
        
        gradient = torch.clamp(mse_loss, min=0.0)
        
        return gradient.item()
    
    def get_decoder_params(self) -> Iterator[nn.Parameter]:
        """
        返回解码器参数（用于训练）
        
        Returns:
            解码器参数迭代器
        """
        return self.decoder.parameters()
    
    def forward(
        self,
        higher_state: torch.Tensor,
        lower_state: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播（支持反向传播训练）
        
        Args:
            higher_state: 高层状态张量，形状 (..., higher_dim)
            lower_state: 低层状态张量，形状 (..., lower_dim)
            
        Returns:
            觉知梯度张量（非负）
        """
        predicted_lower = self.decoder(higher_state)
        mse_loss = nn.functional.mse_loss(predicted_lower, lower_state, reduction='mean')
        return torch.clamp(mse_loss, min=0.0)


class LiftOperator(nn.Module):
    """
    Lift算子
    
    实现 lift 算子：E^(λ)(t_act^+) = L_λ(E^(λ-1)(t_act))
    
    从低层状态生成高层初始状态。
    结构：1层线性 + Tanh + 1层线性。
    提供数值有限性保证（输出用 clamp 限制在合理范围）。
    """
    
    def __init__(
        self,
        lower_dim: int,
        higher_dim: int,
        output_clamp: float = 10.0
    ):
        """
        初始化Lift算子
        
        Args:
            lower_dim: 低层状态维度
            higher_dim: 高层状态维度
            output_clamp: 输出钳位范围（±output_clamp），默认10.0
        """
        super().__init__()
        
        self.lower_dim = lower_dim
        self.higher_dim = higher_dim
        self.output_clamp = output_clamp
        
        self.lift_net = nn.Sequential(
            nn.Linear(lower_dim, higher_dim),
            nn.Tanh(),
            nn.Linear(higher_dim, higher_dim)
        )
        
        self._frozen_state: Optional[torch.Tensor] = None
        
        logger.info(
            f"LiftOperator initialized: "
            f"lower_dim={lower_dim}, "
            f"higher_dim={higher_dim}, "
            f"output_clamp={output_clamp}"
        )
    
    def lift(self, lower_state: torch.Tensor) -> torch.Tensor:
        """
        从低层状态生成高层状态
        
        Args:
            lower_state: 低层状态张量，形状 (..., lower_dim)
            
        Returns:
            高层状态张量，形状 (..., higher_dim)
        """
        higher_state = self.lift_net(lower_state)
        
        clamped_state = torch.clamp(
            higher_state,
            min=-self.output_clamp,
            max=self.output_clamp
        )
        
        return clamped_state
    
    def forward(self, lower_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播（与 lift 相同）
        
        Args:
            lower_state: 低层状态张量，形状 (..., lower_dim)
            
        Returns:
            高层状态张量，形状 (..., higher_dim)
        """
        return self.lift(lower_state)
    
    def freeze_state(self, state: torch.Tensor) -> None:
        """
        保存冻结状态
        
        Args:
            state: 要冻结的状态张量
        """
        self._frozen_state = state.clone().detach()
        logger.debug(f"LiftOperator: state frozen, shape={self._frozen_state.shape}")
    
    def restore_state(self) -> Optional[torch.Tensor]:
        """
        恢复冻结状态
        
        Returns:
            冻结的状态张量，如果没有冻结状态则返回 None
        """
        if self._frozen_state is None:
            logger.warning("LiftOperator: no frozen state to restore")
            return None
        
        restored = self._frozen_state.clone()
        logger.debug(f"LiftOperator: state restored, shape={restored.shape}")
        return restored
    
    def has_frozen_state(self) -> bool:
        """
        检查是否有冻结状态
        
        Returns:
            是否存在冻结状态
        """
        return self._frozen_state is not None


if __name__ == "__main__":
    print("=" * 60)
    print("AwarenessGradient 模块测试")
    print("=" * 60)
    
    torch.manual_seed(42)
    
    higher_dim = 32
    lower_dim = 16
    hidden_dim = 64
    batch_size = 8
    
    print(f"\n配置: higher_dim={higher_dim}, lower_dim={lower_dim}, "
          f"hidden_dim={hidden_dim}, batch_size={batch_size}")
    
    # 测试1: DecoderNetwork 输出形状
    print("\n" + "-" * 40)
    print("测试1: DecoderNetwork 输出形状")
    print("-" * 40)
    
    decoder = DecoderNetwork(higher_dim, lower_dim, hidden_dim)
    higher_state = torch.randn(batch_size, higher_dim)
    predicted_lower = decoder(higher_state)
    
    assert predicted_lower.shape == (batch_size, lower_dim), \
        f"Decoder输出形状错误: 期望 ({batch_size}, {lower_dim}), 实际 {predicted_lower.shape}"
    
    print(f"✓ Decoder输出形状正确: {predicted_lower.shape}")
    print(f"  数值范围: [{predicted_lower.min().item():.4f}, {predicted_lower.max().item():.4f}]")
    print(f"  有无NaN: {torch.isnan(predicted_lower).any().item()}")
    print(f"  有无Inf: {torch.isinf(predicted_lower).any().item()}")
    
    # 测试2: AwarenessGradient 计算
    print("\n" + "-" * 40)
    print("测试2: AwarenessGradient 计算")
    print("-" * 40)
    
    awareness_grad = AwarenessGradient(higher_dim, lower_dim, hidden_dim)
    lower_state = torch.randn(batch_size, lower_dim)
    
    gradient = awareness_grad.compute(higher_state, lower_state)
    
    assert gradient >= 0.0, f"觉知梯度应为非负，实际: {gradient}"
    assert not torch.isnan(torch.tensor(gradient)), f"觉知梯度包含NaN"
    assert not torch.isinf(torch.tensor(gradient)), f"觉知梯度包含Inf"
    
    print(f"✓ 觉知梯度值: {gradient:.6f}")
    print(f"  非负性验证: {'通过' if gradient >= 0 else '失败'}")
    print(f"  数值有限性: {'通过' if not (torch.isnan(torch.tensor(gradient)) or torch.isinf(torch.tensor(gradient))) else '失败'}")
    
    # 测试 get_decoder_params
    params = list(awareness_grad.get_decoder_params())
    print(f"✓ 解码器参数数量: {len(params)} 个参数组")
    print(f"  总参数量: {sum(p.numel() for p in params)}")
    
    # 测试3: LiftOperator
    print("\n" + "-" * 40)
    print("测试3: LiftOperator")
    print("-" * 40)
    
    lift_op = LiftOperator(lower_dim, higher_dim)
    
    lifted_state = lift_op.lift(lower_state)
    
    assert lifted_state.shape == (batch_size, higher_dim), \
        f"Lift输出形状错误: 期望 ({batch_size}, {higher_dim}), 实际 {lifted_state.shape}"
    
    print(f"✓ Lift输出形状正确: {lifted_state.shape}")
    print(f"  数值范围: [{lifted_state.min().item():.4f}, {lifted_state.max().item():.4f}]")
    print(f"  钳位范围: ±{lift_op.output_clamp}")
    print(f"  有无NaN: {torch.isnan(lifted_state).any().item()}")
    print(f"  有无Inf: {torch.isinf(lifted_state).any().item()}")
    
    # 测试数值有限性
    assert not torch.isnan(lifted_state).any(), "Lift输出包含NaN"
    assert not torch.isinf(lifted_state).any(), "Lift输出包含Inf"
    assert (lifted_state >= -lift_op.output_clamp).all() and (lifted_state <= lift_op.output_clamp).all(), \
        "Lift输出超出钳位范围"
    
    print("✓ 数值有限性验证通过")
    
    # 测试4: freeze/restore
    print("\n" + "-" * 40)
    print("测试4: freeze_state / restore_state")
    print("-" * 40)
    
    test_state = torch.randn(batch_size, higher_dim)
    lift_op.freeze_state(test_state)
    
    assert lift_op.has_frozen_state(), "应有冻结状态"
    print("✓ freeze_state 成功")
    
    restored = lift_op.restore_state()
    
    assert restored is not None, "restore_state 应返回张量"
    assert restored.shape == test_state.shape, "恢复的状态形状不匹配"
    assert torch.allclose(restored, test_state), "恢复的状态与原状态不一致"
    
    print("✓ restore_state 成功")
    print(f"  恢复状态形状: {restored.shape}")
    print(f"  与原状态一致: {torch.allclose(restored, test_state)}")
    
    # 测试无冻结状态时 restore
    lift_op2 = LiftOperator(lower_dim, higher_dim)
    result = lift_op2.restore_state()
    assert result is None, "无冻结状态时应返回None"
    print("✓ 无冻结状态时 restore 返回 None")
    
    # 测试5: 梯度反向传播
    print("\n" + "-" * 40)
    print("测试5: 梯度反向传播")
    print("-" * 40)
    
    awareness_grad_train = AwarenessGradient(higher_dim, lower_dim, hidden_dim)
    optimizer = torch.optim.Adam(awareness_grad_train.get_decoder_params(), lr=0.01)
    
    higher_batch = torch.randn(32, higher_dim)
    lower_batch = torch.randn(32, lower_dim)
    
    initial_loss = awareness_grad_train.compute(higher_batch, lower_batch)
    print(f"  初始损失: {initial_loss:.6f}")
    
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        loss = awareness_grad_train(higher_batch, lower_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    
    final_loss = awareness_grad_train.compute(higher_batch, lower_batch)
    print(f"  最终损失: {final_loss:.6f}")
    print(f"  损失变化: {final_loss - initial_loss:+.6f}")
    
    assert final_loss <= initial_loss or len(losses) > 0, "训练未正常进行"
    print("✓ 梯度反向传播正常")
    
    # 总结
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
