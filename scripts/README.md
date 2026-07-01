# 验证脚本索引

本文档列出所有验证脚本及其用途。

---

## 📋 快速开始

```bash
# 完整验证流程
python scripts/p0_validation_spectral.py   # P0 级
python scripts/p1_validation.py             # P1 级
python scripts/p2_validation.py             # P2 级
```

---

## 🔴 P0 级验证

### `p0_validation_spectral.py`
核心指标验证：Lyapunov 指数、长时间稳定性、漂移率、对齐误差

**运行时间**: ~2 分钟

**输出**:
```
P0 验证总结
  stability      : PASS (diverged=False)
  lyapunov       : PASS (λ_max=-0.3107)
  drift          : PASS (drift=0.000000)
  alignment      : PASS (error=0.016767)

整体结果: PASS
```

---

## 🟡 P1 级验证

### `p1_validation.py`
子系统有效性验证：初始条件鲁棒性、逐层谱约束、衰减率可控性

**运行时间**: ~5 分钟

**输出**:
```
P1 验证总结
  initial_condition    : PASS
  spectral_constraint  : PASS
  decay_controllability: PASS

整体结果: PASS
```

---

## 🟢 P2 级验证

### `p2_validation.py`
动力学特性验证：信号响应、状态空间维数、频率特性、多状态切换

**运行时间**: ~10 分钟

### `quick_p2_test.py`
快速 P2 测试（简化版）

**运行时间**: ~3 分钟

### `complete_p2_test.py`
完整 P2 测试（优化版）

**运行时间**: ~5 分钟

---

## 📊 验证结果汇总

| 脚本 | 状态 | 通过率 |
|------|------|--------|
| p0_validation_spectral.py | ✅ | 100% |
| p1_validation.py | ✅ | 100% |
| p2_validation.py | ⚠️ | 75% |

---

## 🎯 最佳配置（WORK 状态）

```python
decay_rate = 0.5          # 适度衰减
dynamics_scale = 7.0       # 中等动力学强度
target_spectral_norm = 1.5  # 逐层谱约束
max_gradient_norm = 200.0  # 不裁剪真实动力学
state_norm_clip = 0.0      # 关闭主动截断

# 验证结果
λ_max = -0.31              # 边缘混沌
λ_sum = -127.87            # 全局收缩
CV = 0.035                 # 丰富振荡
max_norm = 86.6            # 有界稳定
```

---

## 🔄 运行所有验证

```bash
cd scripts

# 方式1: 逐个运行
python p0_validation_spectral.py
python p1_validation.py
python p2_validation.py

# 方式2: Python 脚本
python -c "
import sys
sys.path.insert(0, '.')
from p0_validation_spectral import main as p0
from p1_validation import main as p1
print('='*60)
p0()
print('='*60)
p1()
"
```

---

## 📁 其他脚本

| 脚本 | 说明 |
|------|------|
| `demo_fusion.py` | 双通道交叉融合演示 |
| `example_semantic_encoder.py` | 语义编码器使用示例 |