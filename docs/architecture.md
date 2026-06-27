# Chronos-Self 系统架构文档

## 概述

Chronos-Self 是一个基于连续动力学的状态监控系统，采用 Neural ODE 框架实现多时间尺度状态演化。系统设计基于 **自我指涉连续动力学（SRCD）** 理论框架，旨在探索动力学系统的稳态机制。

## 核心架构

### 1. 双通道表征系统

系统采用双通道输入编码架构，将外部信息分解为**语义意图流**和**逻辑物理流**：

#### 1.1 语义意图编码器 (SemanticEncoder)

**路径**: `chronos_core/representation/semantic_encoder.py`

**功能**:
- 轻量级 Transformer 编码器
- 情感倾向提取（情绪强度、情感极性）
- 语用意图提取（意图类型、意图强度）
- 输出意图向量（维度 256-512）

**关键接口**:
```python
intent_vector = semantic_encoder.forward(text)
# IntentVector:
#   - combined_vector: 综合意图向量
#   - intent_type: 意图类型 (inform/request/question/greet等)
#   - intent_confidence: 意图置信度
#   - emotional_intensity: 情感强度
#   - emotional_polarity: 情感极性
```

#### 1.2 逻辑物理编码器 (LogicalEncoder)

**路径**: `chronos_core/representation/logical_encoder.py`

**功能**:
- 结构化状态空间模型 (SSM) 架构
- 本体感觉流编码（内部状态：姿态、能量、资源占用）
- 外部世界流编码（环境状态）
- 物理约束和因果链条编码

**子编码器**:
- `ProprioceptiveEncoder`: 本体感觉编码
- `WorldEncoder`: 外部世界状态编码
- `CausalEncoder`: 因果链条编码

#### 1.3 交叉注意力融合 (FusionModule)

**路径**: `chronos_core/representation/fusion.py`

**功能**:
- 语义→物理交叉注意力：意图查询物理约束
- 物理→语义交叉注意力：物理状态约束意图生成
- 双通道 enriched 表征拼接

**架构**:
```
X_sem → SemanticToPhysicalCrossAttention → X_sem_enriched
X_log → PhysicalToSemanticCrossAttention → X_log_enriched
[X_sem_enriched, X_log_enriched] → Concat → X_fused
```

---

### 2. 多时间尺度积分引擎

**路径**: `chronos_core/core/integration_engine.py`

这是系统的**核心组件**，实现自我状态的连续演化。

#### 2.1 快变量动力学 (FastDynamics)

**路径**: `chronos_core/core/fast_dynamics.py`

**维度**: D_f = 2048 (默认)

**功能**:
- 快速认知动态变化
- 集成语义流、物理流、监控调控信号
- 混沌注入耦合到核心子空间
- Neural ODE 自适应步长积分

**演化方程**:
```
dE_fast/dt = F(E_fast, E_slow, X_sem, X_log, ξ, η_meta)
```

其中：
- `X_sem`: 语义输入
- `X_log`: 逻辑输入
- `ξ`: DMN 混沌注入信号
- `η_meta`: 监控调控向量

#### 2.2 慢变量动力学 (SlowDynamics)

**路径**: `chronos_core/core/slow_dynamics.py`

**维度**: D_s = 512 (默认)

**功能**:
- 稳定人格/身份表征
- 快变量池化聚合
- 弹性恢复项（baseline 回归）
- 低频更新策略（每 100 快变量步更新 1 次）

**演化方程**:
```
dE_slow/dt = β * Pool(E_fast) + γ * (E_baseline - E_slow)
```

其中：
- `β`: 自适应耦合系数
- `γ`: 弹性恢复系数
- `E_baseline`: 基准状态

#### 2.3 非对称耦合系统

**路径**: `chronos_core/core/coupling.py`

**功能**:
- 自适应耦合系数计算（基于快变量方差）
- 耦合强度上限裁剪
- 稳定性监测（防止发散）

**耦合机制**:
- 快变量 → 慢变量：强耦合（β 动态调整）
- 慢变量 → 快变量：弱耦合（确保稳定性）

---

### 3. 内源性默认模式网络 (DMN)

**路径**: `chronos_core/core/dmn_system.py`

模拟无外部输入时的内源性动力学维持。

#### 3.1 混沌吸引子库

**路径**: `chronos_core/core/chaos/`

**吸引子类型**:

| 吸引子 | 文件 | 关键参数 |
|--------|------|----------|
| Lorenz | `lorenz_attractor.py` | σ=10, ρ=28, β=8/3 |
| Rössler | `rossler_attractor.py` | a=0.2, b=0.2, c=5.7 |
| Chua | `chua_attractor.py` | α=15.35, β=28 |

**功能**:
- 生成混沌轨迹
- 多吸引子随机切换（间隔 ΔT_switch）
- 切换平滑过渡插值

#### 3.2 高维注入机制 (ChaosInjector)

**路径**: `chronos_core/core/chaos_injector.py`

**功能**:
- 固定随机正交投影矩阵（64 维核心子空间）
- 混沌信号投影注入
- 自适应增益控制（基于核心子空间方差）

**注入方程**:
```
ξ = g * P * chaos_attractor(t)
```

其中：
- `g`: 自适应增益
- `P`: 正交投影矩阵
- `chaos_attractor(t)`: 混沌轨迹

---

### 4. 递归状态监控

**路径**: `chronos_core/core/meta_cognitive/meta_cognitive_system.py`

实现三层状态监控架构。

#### 4.1 L0 感知层

**路径**: `chronos_core/core/meta_cognitive/perception_layer.py`

**功能**:
- 实时滤波与编码接口
- 外部知识库接入（RAG 调用）
- **无自指能力**（仅感知层）

**特点**:
- 不包含 SelfState 信息
- 仅处理外部输入

#### 4.2 L1 自我状态层

**路径**: `chronos_core/core/meta_cognitive/self_state_layer.py`

**功能**:
- 完整认知积分（包含 SelfState）
- 引用 L0 数据的接口
- 内部工作记忆和注意焦点管理

**特点**:
- 包含完整自我状态信息
- 可以引用 L0 的外部数据

#### 4.3 L2 监控层

**路径**: `chronos_core/core/meta_cognitive/meta_cognitive_layer.py`

**功能**:
- 固定稀疏随机投影（Johnson-Lindenstrauss）
- 高阶统计特征提取：
  - 置信度评估
  - 情绪方差
  - 演化曲率
- 元参数调控向量输出：
  - 积分步长调整
  - 衰减率调整
  - 偏移量调整

**物理隔离设计**:
- L2 **从未见过** L0 原始数据
- 仅通过 L1 的压缩投影获取信息
- 确保自指机制的纯粹性

---

### 5. 工作记忆机制

**路径**: `chronos_core/memory/work_memory.py`

实现米勒定律（7±2）容量约束。

#### 5.1 组块 (Chunk)

**数据结构**:
```python
class Chunk:
    chunk_id: str
    content: torch.Tensor  # 组块内容
    chunk_type: str        # 类型标记
    activation: float      # 激活强度
    created_at: float      # 创建时间
    last_accessed: float   # 最后访问时间
    metadata: Dict         # 元数据
```

#### 5.2 激活强度管理

**机制**:
- 注意力绑定的动态生成
- 激活强度向量维护
- 指数衰减更新

#### 5.3 容量约束

**策略**:
- Top-N 组块选择（N=7）
- 低激活组块信息保留（a_min=ε）
- 组块快速恢复机制

---

### 6. 反思与睡眠重放系统

**路径**: `chronos_core/core/reflection/reflection_system.py`

#### 6.1 实时反思机制

**路径**: `chronos_core/core/reflection/realtime_reflection.py`

**功能**:
- 最近 T 步计算图维护（T=1000）
- 有限截断伴随法梯度回传
- 实时轨迹修正

**反思条件**:
- 每 N 步触发一次反思
- 计算自预测误差
- 应用梯度修正

#### 6.2 睡眠重放系统

**路径**: `chronos_core/core/reflection/sleep_replay.py`

**功能**:
- 24 小时触发机制（强制进入睡眠期）
- 关键帧向量数据库存储（ChromaDB）
- 重放一致性损失计算
- 预测改善损失计算

**睡眠流程**:
1. 识别关键帧（时间戳标记）
2. 从关键帧向前积分重放
3. 计算重放与实际轨迹的一致性损失
4. 计算预测改善损失
5. 仅更新积分引擎参数（不修改关键帧）

---

### 7. 训练系统

**路径**: `chronos_core/training/training_system.py`

#### 7.1 损失函数系统

**路径**: `chronos_core/training/loss_functions.py`

**损失类型**:

| 损失 | 公式 | 作用 |
|------|------|------|
| 预测损失 | L_pred | 生存任务自预测 |
| 抗寂灭损失 | L_anti_quietus | 防止状态坍塌 |
| 惯性正则损失 | L_inertia | 状态连续性 |

**组合损失**:
```
L_total = L_pred + λ * L_anti_quietus + μ * L_inertia
```

#### 7.2 动力学对齐训练

**路径**: `chronos_core/training/dynamics_alignment.py`

**功能**:
- 多步长一致性损失
- 半群性质正则损失
- 长时序开环损失（72 小时）

**对齐验证**:
- 步长 h₁ 和 h₂ 的轨迹终点误差 < ε
- 多步积分满足半群性质

#### 7.3 分阶段冻结策略

**路径**: `chronos_core/training/freezing_strategy.py`

**策略**:
- **L0 编码器冻结**: 语义/逻辑编码器固定
- **L1 积分引擎训练**: 从头训练
- **固定投影矩阵冻结**: 核心子空间、L2 投影冻结

---

### 8. 验证系统

**路径**: `chronos_core/validation/validation_system.py`

#### 8.1 P0 级核心验证

**路径**: `chronos_core/validation/p0_validation.py`

**验证项**:
- 72 小时无输入开环运行
- 慢变量基线漂移率监测
- 最大李雅普诺夫指数计算
- 动力学对齐验证

**判定标准**:
```
- 基线漂移率 < 0.1
- λ_max ∈ (0, 0.1)  # 边缘混沌
- 轨迹终点误差 < 0.05
```

#### 8.2 动力学序参量监测

**路径**: `chronos_core/validation/dynamics_monitoring.py`

**监测指标**:
- 状态自相关系数 ρ(τ_mid)
- 最大李雅普诺夫指数 λ_max
- 自预测误差稳态值 ε_ssr

#### 8.3 行为学指标判定

**路径**: `chronos_core/validation/behavioral_metrics.py`

**判定指标**:
- 自发目标生成（意图熵）
- 跨场景知识迁移
- 干预后行为重组

---

### 9. 完整系统集成

**路径**: `chronos_core/integration/system_integration.py`

#### 9.1 ChronosSystem

**主系统类**，整合所有核心组件：

```python
class ChronosSystem:
    # 核心组件
    integration_engine: IntegrationEngine
    dmn: DefaultModeNetwork
    working_memory: WorkingMemory
    meta_cognitive_system: MetaCognitiveSystem
    reflection_system: ReflectionSystem
    semantic_encoder: SemanticEncoder
    logical_encoder: LogicalEncoder
    fusion_module: FusionModule
```

**主要方法**:
- `initialize()`: 初始化所有组件
- `process_input()`: 处理外部输入
- `get_current_state()`: 获取当前自我状态
- `run_validation()`: 运行验证
- `trigger_sleep()`: 触发睡眠重放

#### 9.2 ChronosSystemController

**系统控制器**，管理生命周期：

```python
class ChronosSystemController:
    def start()        # 启动系统
    def stop()         # 停止系统
    def process_input() # 处理输入
    def run_continuous() # 连续运行
```

---

## 系统流程图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Chronos-Self 系统完整流程                                │
└──────────────────────────────────────────────────────────────────────────────┘

┌────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│  文本输入   │───→│  SemanticEncoder     │───→│                      │
└────────────┘    │  (语义意图编码)        │    │                      │
                  └──────────────────────┘    │                      │
                                              │   FusionModule       │───→ X_fused
┌────────────┐    ┌──────────────────────┐    │   (交叉注意力融合)    │
│ 物理状态   │───→│  LogicalEncoder       │───→│                      │
└────────────┘    │  (逻辑物理编码)        │    │                      │
                  └──────────────────────┘    └──────────────────────┘
                                                       │
                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                           IntegrationEngine                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  FastDynamics    │    │  SlowDynamics    │    │  CouplingSystem     │   │
│  │  (D_f = 2048)    │←──→│  (D_s = 512)    │←──→│  (自适应耦合)        │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
│           ↑                      ↑                                         │
│           │                      │                                         │
│           └───────────────────────────────────────────────────────────────│
│                                │                                            │
│                                ▼                                            │
│                    ┌──────────────────────┐                                │
│                    │  DefaultModeNetwork  │                                │
│                    │  (混沌注入)           │                                │
│                    │  Lorenz/Rössler/Chua │                                │
│                    └──────────────────────┘                                │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                        MetaCognitiveSystem                                   │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────────┐   │
│  │  L0 感知层       │───→│  L1 状态层       │───→│  L2 监控层          │   │
│  │  (无自指)        │    │  (包含SelfState)│    │  (物理隔离)         │   │
│  └──────────────────┘    └──────────────────┘    └──────────────────────┘   │
│                                                        │                    │
│                                                        ▼                    │
│                                              元参数调控向量输出             │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ReflectionSystem                                     │
│  ┌──────────────────────────┐    ┌──────────────────────────────┐           │
│  │  RealtimeReflection      │    │  SleepReplay                 │           │
│  │  (T=1000步计算图维护)    │    │  (24h触发/关键帧重放)        │           │
│  │  (截断伴随法梯度)        │    │  (一致性损失+改善损失)       │           │
│  └──────────────────────────┘    └──────────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         WorkingMemory                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Chunk管理 (7±2容量)  │  激活强度向量  │  指数衰减  │  快速恢复     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                         系统响应输出                                         │
│  - SelfState 更新                                                         │
│  - 行为输出                                                                │
│  - 状态报告                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 配置参数

**路径**: `chronos_core/utils/config.py`

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fast_variable_dim` | 2048 | 快变量维度 D_f |
| `slow_variable_dim` | 512 | 慢变量维度 D_s |
| `core_subspace_dim` | 64 | 核心子空间维度 k |
| `working_memory_chunks` | 7 | 工作记忆容量 |
| `reflection_window` | 1000 | 反思窗口 T |
| `sleep_replay_interval_hours` | 24.0 | 睡眠触发间隔 |
| `coupling_adaptation_coeff` | 1.0 | 耦合系数 β |
| `elastic_restoration_coeff` | 0.001 | 弹性恢复系数 γ |
| `anti_quietus_weight` | 0.01 | 抗寂灭权重 λ |
| `lyapunov_threshold` | 0.1 | 李雅普诺夫阈值 |

---

## 文件结构

```
chronos_core/
├── core/
│   ├── state.py              # SelfState 定义
│   ├── integration_engine.py # 积分引擎
│   ├── fast_dynamics.py      # 快变量动力学
│   ├── slow_dynamics.py      # 慢变量动力学
│   ├── coupling.py           # 耦合系统
│   ├── dmn_system.py         # 默认模式网络
│   ├── chaos/
│   │   ├── lorenz_attractor.py
│   │   ├── rossler_attractor.py
│   │   ├── chua_attractor.py
│   │   └── attractor_manager.py
│   ├── meta_cognitive/
│   │   ├── meta_cognitive_system.py
│   │   ├── perception_layer.py      # L0
│   │   ├── self_state_layer.py      # L1
│   │   └── meta_cognitive_layer.py  # L2
│   └── reflection/
│       ├── reflection_system.py
│       ├── realtime_reflection.py
│       └── sleep_replay.py
├── representation/
│   ├── semantic_encoder.py    # 语义编码器
│   ├── logical_encoder.py     # 逻辑编码器
│   ├── fusion.py              # 融合模块
│   └── ssm.py                 # 状态空间模型
├── memory/
│   └── work_memory.py         # 工作记忆
├── training/
│   ├── training_system.py     # 训练系统
│   ├── loss_functions.py      # 损失函数
│   ├── dynamics_alignment.py  # 动力学对齐
│   └── freezing_strategy.py   # 冻结策略
├── validation/
│   ├── validation_system.py   # 验证系统
│   ├── p0_validation.py       # P0 级验证
│   ├── dynamics_monitoring.py # 动力学监测
│   └── behavioral_metrics.py  # 行为学指标
├── integration/
│   └── system_integration.py  # 完整系统集成
└── utils/
    ├── config.py              # 配置管理
    └── logger.py              # 日志系统
```

---

## 使用示例

```python
from chronos_core.integration.system_integration import (
    ChronosSystem,
    ChronosSystemConfig,
    ChronosSystemController
)
from chronos_core.utils.config import ChronosConfig

# 创建配置
config = ChronosConfig()
config.dim.fast_variable_dim = 2048
config.dim.slow_variable_dim = 512

# 创建系统
system_config = ChronosSystemConfig(
    enable_semantic_encoder=True,
    enable_logical_encoder=True,
    enable_meta_cognitive=True,
    enable_reflection=True,
)

system = ChronosSystem(
    config=system_config,
    global_config=config,
    device="cuda"
)
system.initialize()

# 创建控制器
controller = ChronosSystemController(system)
controller.start()

# 处理输入
response = controller.process_input(text="你好，请介绍一下你自己")

# 获取状态
state = system.get_current_state()
print(f"状态范数: {state.get_fast_norm():.4f}")

# 停止系统
controller.stop()
```

---

## 扩展阅读

- [训练指南](training_guide.md) - 训练流程详解
- [验证指南](validation_guide.md) - P0-P2 级验证步骤
- [使用示例](../examples/simple_text_interaction.py) - 简单文本交互示例