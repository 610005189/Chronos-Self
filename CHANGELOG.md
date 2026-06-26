# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- 内源性默认模式网络（DMN）
  - 混沌吸引子系统（Lorenz、Rossler、Chua）
  - 吸引子管理器（Attractor Manager）
  - 混沌注入器（Chaos Injector）
  - DMN系统集成

- 层级化自指与元认知调控
  - 感知层（Perception Layer）
  - 自我状态层（Self State Layer）
  - 元认知层（Meta Cognitive Layer）
  - 元认知管理器（Meta Cognitive Manager）

- 反思机制与睡眠重放
  - 实时反思（Realtime Reflection）
  - 睡眠重放（Sleep Replay）
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

[unreleased]: https://github.com/Chronos-Self/Chronos-Self/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Chronos-Self/Chronos-Self/releases/tag/v0.1.0
