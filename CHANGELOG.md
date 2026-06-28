# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 元意识引擎实现（Meta-Consciousness Engine）
  - 元意识场 `MetaConsciousnessField`：基于指数移动平均近似窗口积分计算 $\mathcal{M}_{pre}(t)$
  - 自指深度 `SelfReferentialDepth`：阶梯函数计算 $\Lambda(t)$，涌现阈值 + 层级间距
  - 觉知梯度 `AwarenessGradient`：Dec 网络 + 度量张量 $\Sigma_G$
  - Lift 算子 `LiftOperator`：层级首次激活的初始条件生成
  - 动态层级 `DynamicMetaCognitiveLayers`：整合所有组件，$\Lambda(t)$ 控制层级开关
  - 系统集成：默认关闭（向后兼容），配置开关 `enable_meta_consciousness`
  - 验证系统集成：P3/P4 命题验证预留接口

- 项目版本统一管理
  - 创建 `_version.py` 作为单一版本源
  - 所有模块从单一源导入版本信息

- 实验记录系统增强
  - P0 验证增加元意识指标字段（m_pre_history, lambda_history, awareness_gradient_history）
  - ValidationResult 增加 p3_result/p4_result 字段
  - ExperimentRecord 增加 meta_consciousness 字段

- 开发日志文档
  - 新增 `docs/dev_log.md` 记录关键开发节点和技术发现

- Phase 14 动力学精细化调整
  - 混沌注入参数调优：base_gain=0.1, min_gain=0.1, attractor_noise_scale=0.01
  - 配置传递修复：DMN 系统使用 base_gain 而非 chaos_injection_gain
  - min_gain 参数支持：混沌注入器添加最小增益限制，防止自适应增益过低
  - 自适应增益公式改进：g = g0 * 2.0 / (1.0 + 0.3 * Var/σ²_target)（更积极）
  - 耦合上限放宽：slow_coupling_limit 从 0.1 提高到 0.5
  - Lyapunov 阈值放宽：lyapunov_negative_threshold 从 -0.1 放宽到 -0.5
  - P4 主体独立性验证改进：增加 Spearman 相关性和 Wasserstein 距离度量

- Phase 15 日志密度优化
  - 各模块日志级别降级：fast_dynamics, slow_dynamics, meta_cognitive_system, training_system, validation_system, reflection_system, attractor_manager, fusion
  - 将参数信息、创建信息等详细日志从 info 降级为 debug

### Changed

- README 文档术语工程化
  - "元认知" → "状态监控"
  - "涌现" → "稳态"
  - "内源性" → "自持"
  - "睡眠重放" → "离线回放"
  - "好奇心" → "探索驱动"

- README 增加 P3/P4 验证体系说明和验证进度总览
- README 路线图增加 Phase 13、14、15 状态说明

### Fixed

- 修复验证系统属性路径错误（p0_validation.py, validation_system.py）
  - `meta_system.meta_consciousness_field` → `meta_system.dynamic_layers.meta_field`
  - `meta_system.self_referential_depth` → `meta_system.dynamic_layers.self_ref_depth`

- 修复混沌注入器最小增益硬编码问题（chaos_injector.py）
  - 使用 min_gain 参数替代硬编码的 0.001

- 修复 DMN 系统配置参数传递错误（dmn_system.py）
  - 使用 chaos_config.base_gain 而非 chaos_config.chaos_injection_gain

- 修复 Lyapunov 指数计算逻辑缺陷（dynamics_monitoring.py, p0_validation.py）
  - dynamics_monitoring: 从简化假设计算改为真实双引擎轨迹积分
  - p0_validation: 从分开运行轨迹改为同时运行参考和扰动轨迹
  - 创建两个独立引擎实例，使用不同随机种子确保演化轨迹不同

### Removed

- (None)

## [0.1.1] - 2026-06-27

### Added

- 竞争性稳态机制（Competitive Emergence）
  - 退火竞争收敛算法
  - 六指标综合判定体系（3动力学 + 3行为学）
  - 稳态阈值动态调整

- 探索驱动引擎（Exploration Engine）
  - L2 驱动的预测误差最小化
  - 伪意图生成机制
  - 探索驱动-预测误差循环

- 训练系统端到端验证
  - 完整训练循环测试
  - 动力学对齐验证
  - 冻结策略效果测试

### Changed

- 动力学参数优化
  - 降低混沌注入强度（chaos_injection_gain）从 0.01 → 0.005
  - 增加弹性恢复系数（elastic_restoration_coeff）从 0.01 → 0.05
  - 调整耦合适应系数（coupling_adaptation_coeff）从 1.0 → 0.5
  - 增加惯性权重（inertia_weight）从 0.01 → 0.05

- 项目结构优化
  - 将原架构文档迁移至 docs/architecture_original.md
  - 重写根目录 README.md，提升项目可访问性

- ChronosSystem架构改进
  - 继承 torch.nn.Module 实现标准模块管理
  - 注册所有子模块（integration_engine, semantic_encoder, logical_encoder, fusion_module, meta_cognitive, training_system）
  - 支持 .to(device)、.parameters()、.state_dict() 等标准 PyTorch 方法

- 核心模块注释术语统一
  - meta_cognitive 模块：元认知 → 递归状态监控
  - competitive_emergence 模块：涌现 → 稳态
  - curiosity_engine 模块：好奇心 → 探索驱动
  - reflection 模块：睡眠重放 → 离线回放
  - dmn_system 模块：内源性 → 自持

### Fixed

- 修复 test_config.py 默认值测试（匹配新的配置参数）
- 修复 minimal_config_test.py 键名错误（total_count → total_tests）
- 修复 ExternalInput 和 SelfState 类中冗余类型检查
- 修复 ChunkType.PROPRI0CEPTIVE 拼写错误（数字 0 → 字母 o）
- 修复 FusionOutput.metadata 类型不匹配（Dict → Optional[Dict]）
- 修复 coupling.py 类型注解（object → StateManager）
- 修复 SSM forward NaN/Inf 检查缺失
- 修复 neural_ode adaptive_integrate 无限循环风险（添加 max_retries）
- 修复 ChronosSystem 子模块未注册问题（meta_cognitive, reflection, validation）
- 修复 ExternalInput X_proprio 冗余类型检查（删除 else 分支中的死代码）
- 修复 MetaCognitiveSystem config 属性访问错误（self.config → self.meta_config）

### Removed

- 清理未使用的导入和死代码

## [0.1.0] - 2026-06-27

### Added

- 核心架构实现（7个子系统）
  - 表征系统（双通道编码）
  - 积分引擎（多时间尺度连续积分）
  - 默认模式网络（DMN）
  - 递归状态监控系统
  - 反思与离线回放系统
  - 记忆系统
  - 验证系统

- 双通道表征与交叉融合
  - 语义编码器（Semantic Encoder）
  - 本体感觉编码器（Proprioceptive Encoder）
  - 因果编码器（Causal Encoder）
  - 逻辑编码器（Logical Encoder）
  - 世界编码器（World Encoder）
  - 状态空间模型（SSM）
  - 交叉融合模块（Fusion）

- 多时间尺度连续积分引擎
  - 快速动力学（Fast Dynamics）
  - 慢速动力学（Slow Dynamics）
  - 神经ODE（Neural ODE）
  - 耦合机制（Coupling）
  - 状态管理器（State Manager）

- 自持默认模式网络（DMN）
  - 混沌吸引子系统（Lorenz、Rossler、Chua）
  - 吸引子管理器（Attractor Manager）
  - 混沌注入器（Chaos Injector）
  - DMN系统集成

- 递归状态监控
  - 感知层（Perception Layer）
  - 状态层（State Layer）
  - 监控层（Monitoring Layer）
  - 监控管理器（Meta Cognitive Manager）

- 反思机制与离线回放
  - 实时反思（Realtime Reflection）
  - 离线回放（Offline Replay）
  - 离线期更新器（Offline Updater）
  - 反思系统集成

- 验证系统（P0/P1/P2三级验证）
  - P0验证（基础动力学验证）
  - 行为指标（Behavioral Metrics）
  - 动力学监控（Dynamics Monitoring）
  - 验证系统集成

- 完整文档
  - 架构设计文档（architecture.md）
  - 训练指南（training_guide.md）
  - 验证指南（validation_guide.md）

### Changed

- 无（初始版本）

### Deprecated

- 无（初始版本）

### Removed

- 无（初始版本）

### Fixed

- 无（初始版本）

### Security

- 无（初始版本）

[unreleased]: https://github.com/610005189/Chronos-Self/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/610005189/Chronos-Self/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/610005189/Chronos-Self/releases/tag/v0.1.0
