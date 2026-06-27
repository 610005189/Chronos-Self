# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 标准开源项目文件
  - LICENSE（MIT License）
  - CONTRIBUTING.md（贡献指南）
  - docs/README.md（文档索引）

- 竞争性稳态机制（Competitive Emergence）
  - 退火竞争收敛算法
  - 六指标综合判定体系（3动力学 + 3行为学）
  - 稳态阈值动态调整

- 探索驱动引擎（Curiosity Engine）
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

### Removed

- 清理未使用的导入和死代码

## [0.1.0] - 2026-06-27

### Added

- 核心架构实现（7个子系统）
  - 表征系统（双通道编码）
  - 积分引擎（多时间尺度连续积分）
  - 默认模式网络（DMN）
  - 元认知调控系统
  - 反思与睡眠重放系统
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
  - 睡眠更新器（Sleep Updater）
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

[unreleased]: https://github.com/610005189/Chronos-Self/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/610005189/Chronos-Self/releases/tag/v0.1.0
