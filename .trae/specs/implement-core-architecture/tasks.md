# Tasks - Chronos-Self 核心架构实现

## Phase 1: 项目基础设施与核心框架

- [x] Task 1: 项目结构与配置管理
  - [x] SubTask 1.1: 创建项目目录结构（core, representation, memory, training, validation, utils）
  - [x] SubTask 1.2: 设置 Python 项目配置（requirements.txt, setup.py, pyproject.toml）
  - [x] SubTask 1.3: 创建配置管理系统（config.py, 包含所有超参数）
  - [x] SubTask 1.4: 实现日志系统（logger.py）

- [x] Task 2: 核心数据结构与状态管理
  - [x] SubTask 2.1: 定义自我状态类 SelfState（包含快变量和慢变量）
  - [x] SubTask 2.2: 实现状态管理器 StateManager（状态初始化、更新、保存）
  - [x] SubTask 2.3: 定义外部输入类 ExternalInput（语义流和物理流）
  - [x] SubTask 2.4: 实现时间戳和演化历史记录系统

## Phase 2: 双通道表征系统

- [ ] Task 3: 语义意图编码器
  - [ ] SubTask 3.1: 实现轻量级Transformer编码器（SemanticEncoder）
  - [ ] SubTask 3.2: 实现情感倾向提取模块
  - [ ] SubTask 3.3: 实现语用意图提取模块
  - [ ] SubTask 3.4: 输出意图向量（维度256-512）

- [ ] Task 4: 逻辑物理编码器
  - [ ] SubTask 4.1: 实现结构化状态空间模型（LogicalEncoder, SSM架构）
  - [ ] SubTask 4.2: 实现本体感觉流编码（内部状态：姿态、能量、资源占用）
  - [ ] SubTask 4.3: 实现外部世界流编码（环境状态）
  - [ ] SubTask 4.4: 实现物理约束和因果链条编码

- [ ] Task 5: 交叉注意力融合机制
  - [ ] SubTask 5.1: 实现语义→物理交叉注意力（SemanticToPhysicalCrossAttention）
  - [ ] SubTask 5.2: 实现物理→语义交叉注意力（PhysicalToSemanticCrossAttention）
  - [ ] SubTask 5.3: 实现融合模块（FusionModule），拼接双通道 enriched 表征
  - [ ] SubTask 5.4: 测试融合机制的正确性

## Phase 3: 多时间尺度连续积分引擎（核心）

- [ ] Task 6: 快变量动力学系统
  - [ ] SubTask 6.1: 实现快变量演化函数 FastDynamics（维度2048）
  - [ ] SubTask 6.2: 实现Neural ODE求解器（使用 torchdiffeq 或自实现）
  - [ ] SubTask 6.3: 实现自适应步长积分机制
  - [ ] SubTask 6.4: 集成语义流、物理流、元认知调控信号、混沌注入

- [ ] Task 7: 慢变量动力学系统
  - [ ] SubTask 7.1: 实现慢变量演化函数 SlowDynamics（维度512）
  - [ ] SubTask 7.2: 实现快变量池化机制（Pooling）
  - [ ] SubTask 7.3: 实现弹性恢复项（baseline回归）
  - [ ] SubTask 7.4: 实现慢变量低频更新策略（每100快变量步更新1次）

- [ ] Task 8: 非对称耦合与稳定性约束
  - [ ] SubTask 8.1: 实现自适应耦合系数计算（基于快变量方差）
  - [ ] SubTask 8.2: 实现耦合强度上限裁剪（clip机制）
  - [ ] SubTask 8.3: 实现稳定性监测系统（防止发散）
  - [ ] SubTask 8.4: 测试边缘混沌稳态维持

## Phase 4: 内源性默认模式网络

- [ ] Task 9: 混沌吸引子库
  - [ ] SubTask 9.1: 实现洛伦兹吸引子（LorenzAttractor）
  - [ ] SubTask 9.2: 实现罗斯勒吸引子（RosslerAttractor）
  - [ ] SubTask 9.3: 实现蔡氏电路吸引子（ChuaAttractor）
  - [ ] SubTask 9.4: 创建吸引子选择和管理接口

- [ ] Task 10: 高维注入机制
  - [ ] SubTask 10.1: 实现固定随机正交投影矩阵（64维核心子空间）
  - [ ] SubTask 10.2: 实现混沌信号投影注入（ChaosInjector）
  - [ ] SubTask 10.3: 确保仅耦合到核心子空间
  - [ ] SubTask 10.4: 测试注入机制的正确性

- [ ] Task 11: 自适应增益控制与多吸引子切换
  - [ ] SubTask 11.1: 实现自适应增益计算（基于核心子空间方差）
  - [ ] SubTask 11.2: 实现多吸引子随机切换机制（间隔ΔT_switch）
  - [ ] SubTask 11.3: 实现切换时的平滑插值过渡
  - [ ] SubTask 11.4: 测试无输入时的持续动力学维持

## Phase 5: 层级化自指与元认知调控

- [ ] Task 12: L0感知层
  - [ ] SubTask 12.1: 实现L0实时滤波与编码接口
  - [ ] SubTask 12.2: 实现外部知识库接入（RAG调用）
  - [ ] SubTask 12.3: 确保L0无自指能力（仅感知）
  - [ ] SubTask 12.4: 测试L0与L1的数据传递

- [ ] Task 13: L1自我状态层
  - [ ] SubTask 13.1: 实现L1完整认知积分（包含SelfState）
  - [ ] SubTask 13.2: 实现L1引用L0数据的接口
  - [ ] SubTask 13.3: 实现L1内部工作记忆和注意焦点管理
  - [ ] SubTask 13.4: 测试L1的状态完整性

- [ ] Task 14: L2元认知层
  - [ ] SubTask 14.1: 实现L2固定稀疏随机投影（Johnson-Lindenstrauss）
  - [ ] SubTask 14.2: 实现高阶统计特征提取（置信度、情绪方差、演化曲率）
  - [ ] SubTask 14.3: 实现元参数调控向量输出（积分步长、衰减率、偏移量）
  - [ ] SubTask 14.4: 确保L2从未见过L0原始数据（物理隔离）

- [ ] Task 15: L2扰动训练与独立性验证
  - [ ] SubTask 15.1: 实现L2调控信号的随机噪声扰动
  - [ ] SubTask 15.2: 实现L1对L2信号的部分依赖机制
  - [ ] SubTask 15.3: 实现L2消融测试接口
  - [ ] SubTask 15.4: 验证移除L2后L1功能维持率>0.4

## Phase 6: 工作记忆机制

- [ ] Task 16: 组块形成与激活强度管理
  - [ ] SubTask 16.1: 实现组块动态生成（注意力绑定）
  - [ ] SubTask 16.2: 实现激活强度向量（ActivationStrength）
  - [ ] SubTask 16.3: 实现激活衰减与更新机制
  - [ ] SubTask 16.4: 测试组块的动态生成过程

- [ ] Task 17: 容量约束与历史信息保留
  - [ ] SubTask 17.1: 实现Top-N组块选择机制（N=7）
  - [ ] SubTask 17.2: 实现低激活组块的信息保留（a_min=ε）
  - [ ] SubTask 17.3: 实现组块的快速恢复机制
  - [ ] SubTask 17.4: 测试容量限制的正确性（满足米勒定律）

## Phase 7: 反思与睡眠重放

- [ ] Task 18: 实时反思机制
  - [ ] SubTask 18.1: 实现最近T步计算图维护（T=1000）
  - [ ] SubTask 18.2: 实现有限截断伴随法梯度回传
  - [ ] SubTask 18.3: 实现实时轨迹修正
  - [ ] SubTask 18.4: 测试实时反思的计算效率

- [ ] Task 19: 睡眠重放系统
  - [ ] SubTask 19.1: 实现24小时触发机制（强制进入睡眠期）
  - [ ] SubTask 19.2: 实现关键帧向量数据库存储
  - [ ] SubTask 19.3: 实现重放一致性损失计算
  - [ ] SubTask 19.4: 实现预测改善损失计算

- [ ] Task 20: 睡眠期梯度更新
  - [ ] SubTask 20.1: 实现从关键帧向前积分的重放流程
  - [ ] SubTask 20.2: 实现伴随法梯度回传（仅更新积分引擎参数）
  - [ ] SubTask 20.3: 实现不修改关键帧本身的约束
  - [ ] SubTask 20.4: 测试睡眠重放的稳定性

## Phase 8: 训练系统

- [ ] Task 21: 损失函数实现
  - [ ] SubTask 21.1: 实现预测损失（生存任务）
  - [ ] SubTask 21.2: 实现抗寂灭损失（自预测准确率）
  - [ ] SubTask 21.3: 实现惯性正则损失
  - [ ] SubTask 21.4: 实现完整损失组合（带权重λ, μ）

- [ ] Task 22: 动力学对齐训练
  - [ ] SubTask 22.1: 实现多步长一致性损失
  - [ ] SubTask 22.2: 实现半群性质正则损失
  - [ ] SubTask 22.3: 实现长时序开环损失（72小时）
  - [ ] SubTask 22.4: 实现周期性验证机制

- [ ] Task 23: 分阶段冻结策略
  - [ ] SubTask 23.1: 实现L0编码器冻结策略
  - [ ] SubTask 23.2: 实现L1积分引擎从头训练策略
  - [ ] SubTask 23.3: 实现固定投影矩阵冻结（核心子空间、L2投影）
  - [ ] SubTask 23.4: 测试冻结策略的正确性

## Phase 9: 验证与涌现判定系统

- [ ] Task 24: P0级核心验证
  - [ ] SubTask 24.1: 实现72小时无输入开环运行测试
  - [ ] SubTask 24.2: 实现慢变量基线漂移率监测
  - [ ] SubTask 24.3: 实现最大李雅普诺夫指数计算
  - [ ] SubTask 24.4: 实现动力学对齐验证（多步长轨迹终点误差）

- [ ] Task 25: 动力学序参量监测
  - [ ] SubTask 25.1: 实现状态自相关系数计算
  - [ ] SubTask 25.2: 实现最大李雅普诺夫指数实时监测
  - [ ] SubTask 25.3: 实现自预测误差稳态值监测
  - [ ] SubTask 25.4: 实现动力学指标可视化

- [ ] Task 26: 行为学指标判定
  - [ ] SubTask 26.1: 实现自发目标生成监测（意图熵）
  - [ ] SubTask 26.2: 实现跨场景知识迁移测试
  - [ ] SubTask 26.3: 实现干预后行为重组测试（L2关闭恢复曲线）
  - [ ] SubTask 26.4: 实现六指标综合涌现判定系统

## Phase 10: 集成测试与文档

- [ ] Task 27: 系统集成与端到端测试
  - [ ] SubTask 27.1: 实现完整的系统流程（输入→积分→输出）
  - [ ] SubTask 27.2: 实现端到端集成测试
  - [ ] SubTask 27.3: 实现性能基准测试（单GPU验证）
  - [ ] SubTask 27.4: 实现最小验证配置测试（256-512维度）

- [ ] Task 28: 文档与示例
  - [ ] SubTask 28.1: 编写架构实现文档（解释关键组件）
  - [ ] SubTask 28.2: 编写使用示例（简单的文本交互环境）
  - [ ] SubTask 28.3: 编写训练指南
  - [ ] SubTask 28.4: 编写验证指南（P0-P2级验证步骤）

# Task Dependencies

**Phase依赖关系：**
- Phase 2, 3, 4 可并行启动（都需要 Phase 1 完成）
- Phase 5 需要 Phase 2 和 Phase 3 完成
- Phase 6 需要 Phase 3 完成
- Phase 7 需要 Phase 3 和 Phase 6 完成
- Phase 8 需要 Phase 1-7 全部完成
- Phase 9 需要 Phase 8 部分完成（损失函数和动力学对齐）
- Phase 10 需要 Phase 1-9 全部完成

**具体依赖：**
- Task 6（快变量动力学）依赖 Task 3, 4, 5（双通道表征完成）
- Task 7（慢变量动力学）依赖 Task 6（快变量动力学）
- Task 14（L2元认知）依赖 Task 13（L1自我状态）
- Task 18（实时反思）依赖 Task 6, 7（积分引擎）
- Task 21（损失函数）依赖 Task 6-17（核心组件完成）
- Task 24（P0验证）依赖 Task 22（动力学对齐训练）

**可并行任务组：**
- Group A（Phase 2-4）：Task 3-5（双通道）、Task 9-11（DMN）可并行
- Group B（Phase 6）：Task 16-17（工作记忆）独立于 Phase 5
- Group C（Phase 9）：Task 24-26 可并行进行不同层级验证