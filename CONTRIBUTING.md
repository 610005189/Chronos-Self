# 贡献指南

感谢您对 Chronos-Self 项目的关注和贡献！本文档将帮助您了解如何参与项目开发。

## 目录

- [项目简介](#项目简介)
- [如何开始](#如何开始)
- [代码规范](#代码规范)
- [提交流程](#提交流程)
- [Issue 报告指南](#issue-报告指南)
- [社区行为准则](#社区行为准则)

## 项目简介

Chronos-Self 是一个基于 PyTorch 的连续时间动力学架构，旨在构建具有稳定状态表征的人工智能系统。项目核心特性包括：

- **7 大核心子系统**：表征系统、积分引擎、默认模式网络、递归监控、反思系统、记忆系统、验证系统
- **双通道表征与交叉融合**：语义与本体感觉双通道编码，多层级交叉融合
- **多时间尺度连续积分**：基于神经 ODE 的快慢动力学耦合
- **自持 DMN**：混沌吸引子驱动的默认模式网络
- **递归状态监控**：感知-状态-监控三层调控
- **反思与离线回放**：实时反思与离线巩固机制
- **三级验证体系**：P0/P1/P2 逐级验证

## 如何开始

### 环境要求

- Python 3.9+
- PyTorch 2.0+
- Git

### 开发环境搭建

1. **Fork 并克隆仓库**

   ```bash
   git clone https://github.com/your-username/Chronos-Self.git
   cd Chronos-Self
   ```

2. **创建虚拟环境**

   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/macOS
   source venv/bin/activate
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **安装开发依赖**

   ```bash
   pip install black pytest mypy
   ```

5. **验证安装**

   ```bash
   python -c "import chronos_core; print('Chronos-Self 导入成功')"
   python run_validation.py --quick
   ```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_fusion.py -v

# 运行快速验证
python run_validation.py --quick
```

## 代码规范

### Python 代码风格

- 使用 **Black** 进行代码格式化
- 行宽：88 字符（Black 默认）
- 使用 4 空格缩进

```bash
# 格式化代码
black chronos_core/ tests/

# 检查格式
black --check chronos_core/ tests/
```

### 类型提示

所有公共 API 必须包含完整的类型注解：

```python
from typing import Optional, Tuple, List
import torch

def encode(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    编码输入张量。
    
    Args:
        x: 输入张量，形状 (batch_size, input_dim)
        context: 可选的上下文张量
        
    Returns:
        (encoded, hidden): 编码结果和隐藏状态
    """
    ...
```

### 导入顺序

1. 标准库导入
2. 第三方库导入（空行分隔）
3. 本地项目导入（空行分隔）

```python
import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from chronos_core.representation.semantic_encoder import SemanticEncoder
from chronos_core.utils.logger import get_logger
```

### 文档字符串

使用 Google 风格的 docstring：

```python
class IntegrationEngine:
    """
    多时间尺度连续积分引擎。
    
    负责快速动力学与慢速动力学的耦合积分，
    支持自适应步长的神经 ODE 求解。
    
    Attributes:
        fast_dynamics: 快速动力学模块
        slow_dynamics: 慢速动力学模块
        coupling: 耦合机制模块
    """
    
    def step(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """
        执行一步积分。
        
        Args:
            state: 当前状态张量
            dt: 时间步长
            
        Returns:
            更新后的状态张量
            
        Raises:
            ValueError: 当状态张量形状不匹配时
        """
        ...
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块 | 小写+下划线 | `semantic_encoder.py` |
| 类 | 大驼峰 | `SemanticEncoder` |
| 函数/方法 | 小写+下划线 | `encode_input()` |
| 变量 | 小写+下划线 | `hidden_state` |
| 常量 | 大写+下划线 | `MAX_STEPS` |
| 私有成员 | 前导下划线 | `_internal_state` |

## 提交流程

### 1. Fork 仓库

在 GitHub 上 Fork Chronos-Self 仓库到您的账号。

### 2. 创建分支

从 `main` 分支创建新的功能分支：

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

分支命名规范：
- `feature/xxx`：新功能
- `fix/xxx`：Bug 修复
- `docs/xxx`：文档更新
- `refactor/xxx`：代码重构
- `perf/xxx`：性能优化

### 3. 提交更改

```bash
git add .
git commit -m "feat: 添加语义编码器的注意力机制"
```

**提交信息规范**（Conventional Commits）：

```
<type>: <description>

[optional body]

[optional footer(s)]
```

类型（type）：
- `feat`：新功能
- `fix`：Bug 修复
- `docs`：文档变更
- `style`：代码格式（不影响功能）
- `refactor`：重构（既不新增功能也不修复 bug）
- `perf`：性能优化
- `test`：测试相关
- `chore`：构建/工具链变更

### 4. 推送并创建 PR

```bash
git push origin feature/your-feature-name
```

然后在 GitHub 上创建 Pull Request。

### PR 要求

- [ ] 代码通过 Black 格式化检查
- [ ] 所有现有测试通过
- [ ] 新增/修改的功能有相应的测试覆盖
- [ ] 更新了相关文档
- [ ] PR 描述清晰，关联相关 Issue
- [ ] 保持 PR 专注，一个 PR 只做一件事

### 5. 代码审查

- 核心开发者会对 PR 进行审查
- 请及时回应审查意见
- PR 通过所有检查后会被合并

## Issue 报告指南

### Bug 报告

提交 Bug 时请包含以下信息：

**标题**：简明扼要地描述问题

```
[Bug] 语义编码器在长序列输入时显存溢出
```

**内容**：

```markdown
## 环境信息
- 操作系统：Windows 11 / Linux Ubuntu 22.04
- Python 版本：3.10.12
- PyTorch 版本：2.1.0
- Chronos-Self 版本：v0.1.0

## 问题描述
清晰描述遇到的问题。

## 复现步骤
1. 配置：使用默认配置
2. 运行：`python examples/simple_text_interaction.py`
3. 输入：长度超过 1000 token 的文本
4. 现象：CUDA out of memory

## 预期行为
描述您期望的正确行为。

## 实际行为
描述实际发生的错误。

## 错误日志/截图
```
Traceback (most recent call last):
  ...
RuntimeError: CUDA out of memory
```

## 其他信息
任何可能有用的额外信息。
```

### 功能建议

提交功能建议时请说明：

- 您遇到的问题/场景
- 您期望的解决方案
- 可能的替代方案
- 额外的上下文信息

## 社区行为准则

### 我们的承诺

为了营造开放和友好的环境，我们承诺让每个人参与本项目和社区时都能获得无骚扰的体验。

### 我们的标准

**积极行为包括：**

- 使用友好和包容的语言
- 尊重不同的观点和经验
- 优雅地接受建设性批评
- 关注对社区最有利的事情
- 对其他社区成员表示同理心

**不可接受的行为包括：**

- 使用性化的语言或图像，以及不受欢迎的性关注或性骚扰
- 恶意评论、侮辱/贬损性评论以及人身或政治攻击
- 公开或私下的骚扰
- 未经明确许可，发布他人的私人信息，如物理地址或电子邮件地址
- 在专业场合中可能被合理认为不当的其他行为

### 我们的责任

项目维护者有责任阐明可接受行为的标准，并应对任何不可接受的行为采取适当和公平的纠正措施。

### 适用范围

本行为准则适用于所有项目空间，也适用于个人在公共空间代表项目或其社区时的情况。

### 执行

如遇到虐待、骚扰或其他不可接受的行为，请通过 [邮件地址] 联系项目团队。所有投诉都将被审查和调查，并将导致被认为必要且适合情况的回应。

---

再次感谢您对 Chronos-Self 项目的贡献！
