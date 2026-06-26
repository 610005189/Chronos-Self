# Chronos-Self 项目结构整理与 README 改造 - The Implementation Plan

## [ ] Task 1: 迁移原 README 到 docs 目录
- **Priority**: high
- **Depends On**: None
- **Description**:
  - 将根目录 readme.md 移动到 docs/ 目录，重命名为 architecture_original.md
  - 保留原文件内容不变，仅移动位置
- **Acceptance Criteria Addressed**: AC-1
- **Test Requirements**:
  - `programmatic` TR-1.1: 根目录不再存在 readme.md
  - `programmatic` TR-1.2: docs/architecture_original.md 存在且内容与原文件一致
- **Notes**: 原文件是557行的架构设计文档，作为历史文档保留

## [ ] Task 2: 编写吸引人的新 README.md
- **Priority**: high
- **Depends On**: Task 1
- **Description**:
  - 首屏：项目名称 + 英文副标题 + 一句话价值主张 + 核心亮点徽章
  - 项目简介：2-3段话讲清楚 Chronos-Self 是什么、为什么重要
  - 核心特性：6-8个bullet points，每个有emoji和简短描述
  - 架构概览：ASCII架构图 + 七大子系统简介
  - 快速开始：安装 + 最小运行示例（5行代码以内）
  - 文档索引：链接到 docs/ 下的各文档
  - 路线图：当前状态和未来规划
  - 贡献指南：简要说明如何参与
  - 许可证：MIT
- **Acceptance Criteria Addressed**: AC-2, AC-6
- **Test Requirements**:
  - `human-judgement` TR-2.1: 首屏3秒内抓住注意力（价值主张清晰、独特性突出）
  - `human-judgement` TR-2.2: 整体视觉层次分明，阅读体验流畅
  - `programmatic` TR-2.3: README 包含所有必需章节（简介、特性、架构、快速开始、文档、贡献、许可）
- **Notes**: 中文为主，配英文副标题和关键词；使用emoji增强视觉效果

## [ ] Task 3: 补全标准开源项目文件
- **Priority**: medium
- **Depends On**: None
- **Description**:
  - 创建 LICENSE 文件（MIT License）
  - 创建 CHANGELOG.md（初始化 v0.1.0 记录）
  - 创建 CONTRIBUTING.md（贡献指南：代码规范、提交流程、issue模板）
- **Acceptance Criteria Addressed**: AC-3
- **Test Requirements**:
  - `programmatic` TR-3.1: LICENSE 文件存在且为标准 MIT 协议
  - `programmatic` TR-3.2: CHANGELOG.md 存在且包含 v0.1.0 版本记录
  - `programmatic` TR-3.3: CONTRIBUTING.md 存在且包含开发环境搭建、代码规范、PR流程
- **Notes**: MIT License 与 pyproject.toml 中声明一致

## [ ] Task 4: 更新 .gitignore 和清理临时目录
- **Priority**: medium
- **Depends On**: None
- **Description**:
  - 在 .gitignore 中添加 validation_results* 模式
  - 添加 run_validation.py 到忽略（如为临时脚本）
  - 检查是否有其他应忽略的文件/目录
- **Acceptance Criteria Addressed**: AC-4, AC-6
- **Test Requirements**:
  - `programmatic` TR-4.1: .gitignore 包含 validation_results* 规则
  - `programmatic` TR-4.2: 现有已追踪文件不受影响
- **Notes**: 不要删除任何文件，只更新忽略规则

## [x] Task 5: 创建 docs/README.md 文档索引
- **Priority**: medium
- **Depends On**: Task 1
- **Description**:
  - 在 docs/ 目录下创建 README.md 作为文档入口
  - 列出所有文档及其简介
  - 按类别组织：架构文档、使用指南、开发文档
- **Acceptance Criteria Addressed**: AC-5
- **Test Requirements**:
  - `programmatic` TR-5.1: docs/README.md 存在
  - `programmatic` TR-5.2: 包含 architecture.md, training_guide.md, validation_guide.md, architecture_original.md 的链接和简介
- **Notes**: 每个文档条目包含：标题、一句话简介、链接

# Task Dependencies
- Task 2 depends on Task 1
- Task 5 depends on Task 1
- Task 3, Task 4 可与其他任务并行执行
