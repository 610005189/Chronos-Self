# Chronos-Self 训练指南

## 概述

本指南详细介绍 Chronos-Self 系统的训练流程、训练策略和最佳实践。

---

## 训练系统架构

**核心文件**: `chronos_core/training/training_system.py`

### 训练流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Chronos-Self 训练流程                               │
└─────────────────────────────────────────────────────────────────────────────┘

1. 数据准备
   └── 外部输入序列生成
       ├── 文本数据集
       ├── 物理状态数据
       └── 合成随机数据（测试模式）

2. 前向传播
   └── IntegrationEngine 演化
       ├── 快变量积分
       ├── 慢变量积分
       ├── 混沌注入
       └── 元认知调控

3. 损失计算
   └── LossFunctions 计算
       ├── 预测损失 L_pred
       ├── 抗寂灭损失 L_anti_quietus
       ├── 惬性正则损失 L_inertia
       └── 动力学对齐损失

4. 反向传播
   └── Neural ODE 伴随法
       ├── 计算图构建
       ├── 伴随变量计算
       ├── 梯度回传

5. 参数更新
   └── FreezingStrategy 应用
       ├── L0 编码器冻结
       ├── L1 积分引擎更新
       └── 固定投影冻结

6. 验证
   └── 周期性验证
       ├── P0 级验证
       ├── 动力学监测
       └── 早停判定
```

---

## 训练配置

### 训练系统配置

```python
from chronos_core.training.training_system import TrainingSystem, TrainingSystemConfig

config = TrainingSystemConfig(
    # 训练模式
    training_mode="standard",  # 'standard', 'alignment', 'p0_validation', 'long_sequence'
    
    # 训练参数
    num_epochs=100,
    batch_size=32,
    learning_rate=1e-4,
    weight_decay=1e-5,
    
    # 优化器配置
    optimizer_type="adam",  # 'adam', 'adamw', 'sgd'
    scheduler_type="step",  # 'step', 'cosine', 'none'
    
    # 梯度配置
    gradient_clip_threshold=1.0,
    use_gradient_accumulation=False,
    
    # 混合精度训练
    use_amp=True,
    
    # 验证配置
    validation_frequency=5,
    validation_duration_hours=1.0,
    
    # 早停配置
    early_stopping_patience=10,
    early_stopping_threshold=0.05,
)
```

### 全局配置

```python
from chronos_core.utils.config import ChronosConfig

global_config = ChronosConfig()

# 训练相关配置
global_config.training.num_epochs = 100
global_config.training.learning_rate = 1e-4
global_config.training.batch_size = 32
global_config.training.gradient_clip_threshold = 1.0
global_config.training.weight_decay = 1e-5
global_config.training.validation_frequency = 5
global_config.training.checkpoint_frequency = 10

# 动力学对齐配置
global_config.training.epochs_before_alignment = 20  # 对齐训练开始时机

# Annealing 配置
global_config.training.annealing_initial_temp = 10.0
global_config.training.annealing_rate = 1000  # 步数
```

---

## 损失函数详解

### 1. 预测损失 (Prediction Loss)

**目的**: 生存任务自预测能力

**公式**:
```
L_pred = ||E_fast(t+T) - E_fast_predicted(t+T)||²
```

**说明**:
- 计算系统对未来状态的预测能力
- 预测准确性反映系统的认知完整性

### 2. 抗寂灭损失 (Anti-Quietus Loss)

**目的**: 防止状态坍塌到零向量

**公式**:
```
L_anti_quietus = λ * (1 - ||E_fast|| / ||E_baseline||)
```

**参数**:
- `λ`: 抗寂灭权重 (默认 0.01)
- `E_baseline`: 基准状态范数

**说明**:
- 确保系统保持活跃状态
- 防止动力学的"寂灭"现象

### 3. 惯性正则损失 (Inertia Loss)

**目的**: 维持状态连续性

**公式**:
```
L_inertia = μ * ||dE/dt - v_prev||²
```

**参数**:
- `μ`: 惯性权重 (默认 0.001)
- `v_prev`: 前一步速度

**说明**:
- 状态演化不应过于剧烈
- 保持认知的连续性

### 4. 动力学对齐损失 (Dynamics Alignment Loss)

**目的**: 多步长积分一致性

**公式**:
```
L_align = ||E(h₁+h₂) - E(E(h₁), h₂)||²
```

**说明**:
- 验证半群性质：E(h₁+h₂) = E(E(h₁), h₂)
- 确保不同步长的积分结果一致

---

## 训练模式

### 1. 标准训练模式 (standard)

**适用场景**: 基础训练

**流程**:
```python
training_system = TrainingSystem(config=TrainingSystemConfig(training_mode="standard"))
training_system.initialize()
history = training_system.train(num_epochs=100)
```

**特点**:
- 使用组合损失函数
- 应用冻结策略
- 周期性验证

### 2. 动力学对齐模式 (alignment)

**适用场景**: 强化动力学一致性

**流程**:
```python
config = TrainingSystemConfig(training_mode="alignment")
training_system = TrainingSystem(config=config)
training_system.initialize()
history = training_system.train(num_epochs=50)
```

**特点**:
- 强化多步长一致性损失
- 验证半群性质
- 更严格的数值稳定性要求

### 3. P0 验证训练模式 (p0_validation)

**适用场景**: P0 级验证准备

**流程**:
```python
config = TrainingSystemConfig(training_mode="p0_validation")
training_system = TrainingSystem(config=config)
training_system.initialize()
history = training_system.train(num_epochs=30)
```

**特点**:
- 训练后系统需要满足 P0 级验证标准
- 72 小时开环稳定性训练
- 边缘混沌状态调节

### 4. 长时序训练模式 (long_sequence)

**适用场景**: 长时间演化训练

**流程**:
```python
config = TrainingSystemConfig(
    training_mode="long_sequence",
    sequence_length=10000,  # 长序列
)
training_system = TrainingSystem(config=config)
training_system.initialize()
history = training_system.train(num_epochs=20)
```

**特点**:
- 更长的积分序列
- 更强的稳定性训练
- 更大的计算开销

---

## 分阶段冻结策略

### 策略设计原理

**路径**: `chronos_core/training/freezing_strategy.py`

冻结策略确保训练的稳定性和正确性：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          冻结策略设计原理                                     │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: L0 编码器冻结
├── SemanticEncoder: 冻结
├── LogicalEncoder: 冻结
└── 原因: 编码器提供稳定的输入表征，不应在训练中变化

Phase 2: L1 积分引擎训练
├── IntegrationEngine: 从头训练
├── FastDynamics: 训练
├── SlowDynamics: 训练
├── CouplingSystem: 训练
└── 原因: 核心动力学需要学习自我指涉机制

Phase 3: 固定投影冻结
├── ChaosInjector 投影矩阵: 冻结
├── L2 元认知投影: 冻结
└── 原因: 随机投影是固定的，不应随训练变化
```

### 应用冻结策略

```python
from chronos_core.training.freezing_strategy import FreezingStrategy, FreezingStrategyConfig

# 创建冻结策略
freeze_config = FreezingStrategyConfig(
    freeze_semantic_encoder=True,
    freeze_logical_encoder=True,
    freeze_integration_engine=False,  # 核心组件训练
    freeze_chaos_projection=True,
    freeze_l2_projection=True,
)

freeze_strategy = FreezingStrategy(config=freeze_config)

# 应用冻结
freeze_strategy.apply_freezing(training_system)

# 获取可训练参数
trainable_params = freeze_strategy.get_trainable_parameters()
print(f"可训练参数数量: {sum(p.numel() for p in trainable_params)}")
```

---

## 训练执行

### 创建训练系统

```python
from chronos_core.training.training_system import TrainingSystem
from chronos_core.utils.config import ChronosConfig

# 创建配置
global_config = ChronosConfig()

# 创建训练系统
training_system = TrainingSystem(
    global_config=global_config,
    device="cuda"
)

# 初始化
training_system.initialize()
```

### 执行训练

```python
# 基础训练
history = training_system.train(
    num_epochs=100,
    callback=None  # 可选回调
)

# 查看训练历史
print(f"总轮数: {history.total_epochs}")
print(f"最佳损失: {history.best_loss}")
print(f"最佳轮数: {history.best_epoch}")
```

### 使用回调函数

```python
def training_callback(epoch: int, metrics: Dict[str, float]):
    """训练回调函数"""
    print(f"Epoch {epoch}: loss={metrics['total_loss']:.4f}")
    
    if metrics.get('validation_passed'):
        print("  ✓ 验证通过")

# 执行训练
history = training_system.train(
    num_epochs=100,
    callback=training_callback
)
```

### 检查点保存和加载

```python
# 保存检查点
training_system.save_checkpoint("checkpoints/model_epoch_50.pth")

# 加载检查点
training_system.load_checkpoint("checkpoints/model_epoch_50.pth")

# 继续训练
history = training_system.train(num_epochs=50)
```

---

## 训练监控

### 损失监控

```python
# 获取损失历史
loss_history = training_system.history.loss_history

# 分析损失趋势
for i, loss_dict in enumerate(loss_history[-10:]):
    print(f"Step {i}:")
    print(f"  预测损失: {loss_dict['prediction_loss']:.4f}")
    print(f"  抗寂灭损失: {loss_dict['anti_quietus_loss']:.4f}")
    print(f"  惯性损失: {loss_dict['inertia_loss']:.4f}")
    print(f"  总损失: {loss_dict['total_loss']:.4f}")
```

### 验证监控

```python
# 获取验证历史
validation_history = training_system.history.validation_history

# 检查验证结果
for result in validation_history:
    print(f"验证轮数: {result['epoch']}")
    print(f"  P0 通过: {result['p0_passed']}")
    print(f"  P1 通过: {result['p1_passed']}")
    print(f"  总体得分: {result['overall_score']:.4f}")
```

### 早停机制

```python
# 配置早停
config = TrainingSystemConfig(
    early_stopping_patience=10,
    early_stopping_threshold=0.05,
)

# 早停触发条件
# - 连续 10 轮损失未下降超过 5%
# - 自动停止训练并保存最佳模型
```

---

## 最佳实践

### 1. 渐进式训练

**推荐流程**:
```
Step 1: 最小配置测试 (D_f=256, D_s=128)
   ├── 验证基本功能
   └── 快速迭代调试

Step 2: 中等配置训练 (D_f=512, D_s=256)
   ├── 30 轮基础训练
   └── 动力学对齐验证

Step 3: 标准配置训练 (D_f=2048, D_s=512)
   ├── 50 轮标准训练
   └── P0 验证训练

Step 4: 长时序训练
   ├── 20 轮长序列训练
   └── 完整验证流程
```

### 2. 混合精度训练

**推荐**: 使用 AMP 提升训练效率

```python
config = TrainingSystemConfig(use_amp=True)

# GPU 训练时自动启用混合精度
# 减少内存占用，加速训练
```

### 3. 学习率调优

**推荐策略**:
```python
# 初始学习率
initial_lr = 1e-4

# 使用 Cosine Annealing
config = TrainingSystemConfig(
    scheduler_type="cosine",
    learning_rate=initial_lr,
)

# 或 Step Decay
config = TrainingSystemConfig(
    scheduler_type="step",
    scheduler_step_size=30,
    scheduler_gamma=0.1,
)
```

### 4. 梯度裁剪

**推荐**: 防止梯度爆炸

```python
config = TrainingSystemConfig(
    gradient_clip_threshold=1.0,
)

# Neural ODE 训练中梯度裁剪尤其重要
# 防止伴随法计算中的梯度爆炸
```

### 5. 定期验证

**推荐**: 周期性验证确保训练质量

```python
config = TrainingSystemConfig(
    validation_frequency=5,  # 每 5 轮验证
    validation_duration_hours=0.1,  # 快速验证
)

# 及时发现训练问题
# 避免"虚假"训练成功
```

---

## 训练脚本示例

### 完整训练脚本

```python
"""
Chronos-Self 训练脚本示例
"""

import torch
from chronos_core.training.training_system import TrainingSystem, TrainingSystemConfig
from chronos_core.utils.config import ChronosConfig

def main():
    # 创建配置
    global_config = ChronosConfig()
    global_config.dim.fast_variable_dim = 512  # 中等配置
    global_config.dim.slow_variable_dim = 256
    
    # 训练配置
    training_config = TrainingSystemConfig(
        training_mode="standard",
        num_epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        use_amp=True,
        validation_frequency=5,
        checkpoint_frequency=10,
        gradient_clip_threshold=1.0,
    )
    
    # 创建训练系统
    training_system = TrainingSystem(
        config=training_config,
        global_config=global_config,
        device="cuda"
    )
    
    # 初始化
    training_system.initialize()
    
    # 定义回调
    def callback(epoch, metrics):
        print(f"\nEpoch {epoch}:")
        print(f"  损失: {metrics['total_loss']:.4f}")
        if metrics.get('validation_passed'):
            print(f"  ✓ 验证通过")
    
    # 执行训练
    history = training_system.train(
        num_epochs=50,
        callback=callback
    )
    
    # 输出结果
    print("\n" + "=" * 60)
    print("训练完成")
    print(f"  总轮数: {history.total_epochs}")
    print(f"  最佳损失: {history.best_loss:.4f}")
    print(f"  最佳轮数: {history.best_epoch}")
    print("=" * 60)
    
    # 保存最终模型
    training_system.save_checkpoint("checkpoints/final_model.pth")

if __name__ == "__main__":
    main()
```

---

## 常见问题

### Q1: 训练过程中损失不下降？

**可能原因**:
1. 学习率过小或过大
2. 损失函数权重配置不当
3. 冻结策略未正确应用

**解决方案**:
```python
# 调整学习率
config.learning_rate = 5e-5  # 降低或提高

# 检查损失权重
global_config.coupling_stability.anti_quietus_weight = 0.01
global_config.coupling_stability.inertia_weight = 0.001

# 验证冻结策略
freeze_strategy.verify_freezing(training_system)
```

### Q2: 训练出现 NaN？

**可能原因**:
1. 数值不稳定
2. 梯度爆炸
3. 步长设置不当

**解决方案**:
```python
# 降低步长
global_config.neural_ode.dt = 0.001

# 加强梯度裁剪
config.gradient_clip_threshold = 0.5

# 检查稳定性
training_system.integration_engine.check_stability()
```

### Q3: 内存占用过高？

**可能原因**:
1. 序列长度过大
2. 批量过大
3. 计算图未清理

**解决方案**:
```python
# 降低序列长度
config.sequence_length = 100

# 降低批量大小
config.batch_size = 16

# 清理计算图
torch.cuda.empty_cache()
```

---

## 扩展阅读

- [验证指南](validation_guide.md) - 验证流程详解
- [架构文档](architecture.md) - 系统架构概览