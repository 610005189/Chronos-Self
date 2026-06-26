# Chronos-Self 项目结构整理与 README 改造 - Product Requirement Document

## Overview
- **Summary**: 整理 Chronos-Self 项目结构，将原架构文档移入 docs 目录，重新编写吸引人的根 README，并补全标准 Git 项目所需的配置文件和文档。
- **Purpose**: 提升项目可访问性和专业度，让访客一眼就能理解项目价值，同时建立标准的开源项目结构。
- **Target Users**: AI 研究者、工程师、对意识/自我建模感兴趣的开发者、开源贡献者

## Goals
- 将根目录的架构设计文档移入 docs/ 目录，保持项目根目录简洁
- 编写高质量、吸引人的 README.md，突出项目独特性和核心价值
- 补全标准开源项目文件（LICENSE, CHANGELOG, CONTRIBUTING, CODE_OF_CONDUCT 等）
- 清理 .gitignore，将散落的验证结果目录纳入忽略
- 优化项目目录结构，提升可维护性

## Non-Goals (Out of Scope)
- 修改任何核心代码逻辑
- 修改 docs/ 下已有的架构文档、训练指南、验证指南内容
- 添加新功能或修复 bug
- 重新组织 chronos_core 内部模块结构

## Background & Context
当前项目根目录存在以下问题：
1. `readme.md` 内容是 557 行的架构设计文档，过于技术化，不适合作为项目入口
2. 多个 `validation_results_*` 目录散落在根目录，应被 .gitignore 忽略或移入统一位置
3. 缺少标准开源项目文件（LICENSE 虽在 pyproject.toml 声明为 MIT 但无实际文件）
4. docs/ 目录已有 architecture.md，但与根目录 readme.md 内容重复
5. 项目整体缺乏专业的第一印象，访客难以快速理解项目价值

## Functional Requirements
- **FR-1**: 原 readme.md 移入 docs/ 并重命名为 architecture_original.md
- **FR-2**: 新 README.md 包含：项目简介、核心特性、架构亮点、快速开始、安装指南、文档索引、贡献指南、许可证
- **FR-3**: 补全 LICENSE 文件（MIT License）
- **FR-4**: 创建 CHANGELOG.md 初始化版本记录
- **FR-5**: 创建 CONTRIBUTING.md 贡献指南
- **FR-6**: 更新 .gitignore 忽略 validation_results_* 等临时目录
- **FR-7**: 优化 docs/README.md 作为文档索引页

## Non-Functional Requirements
- **NFR-1**: README 必须在 3 秒内抓住读者注意力（首屏包含项目定位和核心价值）
- **NFR-2**: README 使用中英文双语或纯英文（考虑开源社区），本次采用中文为主配英文副标题
- **NFR-3**: README 包含视觉化的架构示意（ASCII 图或 Mermaid 图）
- **NFR-4**: 所有 Markdown 文件格式规范，标题层级清晰
- **NFR-5**: .gitignore 更新后不影响现有开发流程

## Constraints
- **Technical**: Python 项目，基于 PyTorch
- **Business**: 开源项目，MIT 协议
- **Dependencies**: 不引入新依赖，仅修改文档和配置文件

## Assumptions
- 原 readme.md 的架构内容已在 docs/architecture.md 中有类似版本（需确认是否完全重复）
- pyproject.toml 中声明的 MIT License 是项目的真实意图
- 验证结果目录是临时产物，不应纳入版本控制

## Acceptance Criteria

### AC-1: 原 README 迁移完成
- **Given**: 根目录存在 readme.md（架构设计文档）
- **When**: 执行迁移操作
- **Then**: readme.md 被移动到 docs/architecture_original.md，根目录不再有该文件
- **Verification**: `programmatic`

### AC-2: 新 README 具备吸引力
- **Given**: 新编写的 README.md
- **When**: 访客打开项目主页
- **Then**: 首屏能看到项目定位、核心价值主张、1-2个关键亮点
- **Verification**: `human-judgment`
- **Notes**: 评估标准：是否有清晰的价值主张、是否突出独特性、是否引导用户继续阅读

### AC-3: 标准项目文件齐全
- **Given**: 项目根目录
- **When**: 检查标准文件
- **Then**: 存在 LICENSE, CHANGELOG.md, CONTRIBUTING.md, .gitignore（更新版）
- **Verification**: `programmatic`

### AC-4: .gitignore 覆盖临时目录
- **Given**: 更新后的 .gitignore
- **When**: 存在 validation_results* 目录
- **Then**: 这些目录被正确忽略
- **Verification**: `programmatic`

### AC-5: 文档索引完整
- **Given**: docs/ 目录
- **When**: 查看文档结构
- **Then**: 有 docs/README.md 作为索引，列出所有文档及其简介
- **Verification**: `programmatic`

### AC-6: 不破坏现有功能
- **Given**: 所有修改完成后
- **When**: 运行现有测试
- **Then**: 测试通过率与修改前一致
- **Verification**: `programmatic`

## Decisions (已确认)
- [x] README 主语言：中文为主 + 英文副标题（兼顾国内社区和国际视野）
- [x] 添加项目徽章（许可证、Python版本、PyTorch版本、开发状态等）
- [x] validation_results_* 目录：直接加入 .gitignore 忽略，不移动文件
