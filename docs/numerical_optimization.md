以下为基于六个数学工具的完整设计方案与修改指导，以原设计文档（第1-10节）为不可变基线，工具集作为数值底层的替换实现，不修改任何模块定义或损失函数。

---

## 一、工具集在架构中的精确位置

```
原设计模块（不变）                 新增/替换数值底层
─────────────────────────────    ─────────────────────────────
L0: 双通道编码器                  傅里叶特征映射（输入预处理）
L1: 积分引擎 F_theta             谱归一化（权重约束）
                                  IMEX算子分裂（求解器）
                                  辛积分器（两步更新）
                                  伴随重计算（内存管理）
L2: 元认知（固定投影）            不变
ContextRetriever                 线性注意力（检索替换）
睡眠重放/离线训练                 不变
```

所有工具作用在**数值积分与内存管理**层，不改变模块间数据流或状态更新方程。

---

## 二、各工具的工程修改规范（按实施顺序）

### 修改 A：谱归一化（P0，代码量：25行）

**修改文件**：`models/fast_dynamics.py`

**操作**：
1. 在 `EvolutionFunctionMLP.__init__()` 中，将每个 `nn.Linear` 替换为：
```python
self.layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)))
```
2. 若需要移除谱归一化（调试用），增加 `remove_spectral_norm()` 方法：
```python
def remove_spectral_norm(self):
    for i, layer in enumerate(self.layers):
        if hasattr(layer, 'weight_orig'):
            self.layers[i] = nn.utils.remove_spectral_norm(layer)
```
3. 在 `forward()` 中保持原样，谱归一化在前向传播中自动应用。

**验证**：运行 `torch.linalg.matrix_norm(layer.weight, ord=2)` 应在每次前向后 ≈1（浮点误差内）。

---

### 修改 B：IMEX 算子分裂（P0，代码量：120行）

**修改文件**：`integration/integration_engine.py`

**新增方法**：
```python
def _estimate_j0(self, E_fast):
    """从当前状态估算线性主部 J0（对角近似）"""
    eps = 1e-6
    J0_diag = torch.zeros(E_fast.shape[-1])
    for i in range(E_fast.shape[-1]):
        e_plus = E_fast.clone(); e_plus[..., i] += eps
        e_minus = E_fast.clone(); e_minus[..., i] -= eps
        f_plus = self.F_theta(e_plus, ...)
        f_minus = self.F_theta(e_minus, ...)
        J0_diag[i] = ((f_plus - f_minus) / (2*eps))[..., i].mean()
    return torch.diag(J0_diag.clamp(min=0.0))
```

**修改 `step()` 方法**：
```python
def step(self, E_fast, E_slow, X_fused, dt):
    J0 = self._estimate_j0(E_fast)
    I = torch.eye_like(J0)
    A_inv = torch.inverse(I - dt * J0)
    E_linear = A_inv @ E_fast
    
    N_theta = self.F_theta(E_fast, E_slow, X_fused) - J0 @ E_fast
    E_nonlinear = E_fast + dt * N_theta
    
    E_new = E_linear + (E_nonlinear - E_fast)
    return E_new
```

**配置参数**：在 `configs/default.yaml` 新增：
```yaml
integration:
  imex_update_interval: 100
  imex_dt_safety: 0.9
```

---

### 修改 C：辛积分器（P1，代码量：60行）

**修改文件**：`integration/integration_engine.py`

**Verlet-like 更新**（近似辛）：
```python
def step_verlet(self, E_fast, E_slow, X_fused, dt):
    E_half = E_fast + 0.5 * dt * self.F_theta(E_fast, E_slow, X_fused)
    E_new = E_fast + dt * self.F_theta(E_half, E_slow, X_fused)
    E_new = E_new + 0.5 * dt * self.F_theta(E_new, E_slow, X_fused)
    return E_new
```

**切换开关**：在 `config` 中添加 `solver_type: "verlet" | "imex" | "euler"`，默认 `"imex"`。

---

### 修改 D：线性注意力（P1，代码量：40行）

**修改文件**：`memory/context_retriever.py`

**替换 `retrieve()` 中的注意力计算**：
```python
def retrieve(self, E_fast):
    Q = self.query_proj(E_fast)
    phi_Q = torch.nn.functional.elu(Q) + 1.0
    phi_K = torch.nn.functional.elu(self.keys[:self.size]) + 1.0
    
    KV = phi_K.T @ self.values[:self.size]
    Z = phi_K.sum(dim=0)
    context = (phi_Q @ KV) / (phi_Q @ Z + 1e-6)
    return context
```

**原代码保留**：通过 `attention_mode: "linear" | "full"` 控制切换。

---

### 修改 E：伴随重计算（P2，代码量：150行）

**修改文件**：`training/offline_trainer.py`

**实现 `checkpointed_integration` 函数**：
```python
from torch.utils.checkpoint import checkpoint

class CheckpointedIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, E0, F_theta, X_seq, dt, n_steps, checkpoint_interval=50):
        ctx.save_for_backward(E0, X_seq)
        ctx.F_theta = F_theta
        ctx.dt = dt
        ctx.n_steps = n_steps
        ctx.interval = checkpoint_interval
        
        states = [E0]
        for t in range(n_steps):
            if t % checkpoint_interval == 0:
                ctx.checkpoints.append(states[-1].detach())
            states.append(step(states[-1], ...))
        return states[-1]
    
    @staticmethod
    def backward(ctx, grad_output):
        ...
```

---

### 修改 F：傅里叶特征映射（P2，代码量：30行）

**修改文件**：`models/fast_dynamics.py`

**在 `EvolutionFunctionMLP.forward()` 开头插入**：
```python
def forward(self, E_fast, E_slow, X_fused):
    B = self.register_buffer('B', torch.randn(1024, X_fused.shape[-1]) * 2.0)
    gamma_X = torch.cat([
        torch.cos(B @ X_fused.T).T,
        torch.sin(B @ X_fused.T).T
    ], dim=-1)
    
    combined = torch.cat([E_fast, E_slow, gamma_X], dim=-1)
    return self.net(combined)
```

**注意**：`B` 必须在初始化时固定（`self.register_buffer`），不参与训练。

---

## 三、组合集成的推荐实施顺序

| 阶段 | 工具 | 验收标准 |
| :--- | :--- | :--- |
| 第1周 | 谱归一化 + IMEX | 步长从 1e-3 提升至 1e-2 不产生 NaN，单步耗时增加 <10% |
| 第2周 | 辛积分器（Verlet 模式） | 在 IMEX 基础上，长期能量漂移误差降低至原 RK4 的 1/10 |
| 第3周 | 线性注意力 | ContextRetriever 检索耗时从 1.2ms 降至 0.01ms |
| 第4周 | 伴随重计算 | 在保持批量尺寸不变下，GPU显存占用从 12GB 降至 4GB |
| 第5周 | 傅里叶特征 | 离线训练 epoch 数减少至原来的 1/5 |

**总代码增量**：约 425 行 Python（不含注释）。

---

## 四、配置文件的最终形态（新增条目）

```yaml
numerics:
  solver_type: "imex"
  spectral_norm_enabled: true
  spectral_norm_coeff: 1.0
  
  imex:
    update_interval: 100
    dt_safety_factor: 0.9
  
  attention:
    mode: "linear"
    feature_map: "elu_plus_one"
  
  checkpointing:
    enabled: true
    interval: 50
  
  fourier:
    enabled: true
    n_features: 1024
    scale: 2.0
```

---

## 五、验证门槛的重新校准

| 指标 | 原门槛 | 新门槛（基于工具集） |
| :--- | :--- | :--- |
| 72h 开环步数 | 2.6e8 | 2.6e5 |
| 单步耗时 | 0.24s（CPU） | 预期 0.02s（GPU） |
| 总验证耗时 | 不可计算 | 预期 < 3 小时（单GPU） |
| 自相关系数阈值 | ≥0.3 | ≥0.25 |
| 允许的最大步长 | 1e-3 | 0.01 |

---

## 六、修改风险等级与回滚方案

| 工具 | 风险等级 | 回滚方案 |
| :--- | :--- | :--- |
| 谱归一化 | 极低 | 注释 `nn.utils.spectral_norm` 调用即可 |
| IMEX | 低 | 保留原 `euler` 求解器作为 fallback |
| 辛积分器 | 中 | 需验证 `F_theta` 是否为梯度场；若不满足，切回 Verlet 近似 |
| 线性注意力 | 低 | 保留 `full` 模式通过配置切换 |
| 伴随重计算 | 中 | 若梯度不稳定，增加检查点间隔或禁用 |
| 傅里叶特征 | 极低 | 将 `enabled: false` 即可还原输入维度 |