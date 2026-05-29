# 🔬 Fidelity Metrics

**语义保真度度量工具** — 诊断神经网络各层之间的信息流是否断裂。

## 安装

```bash
pip install fidelity-metrics
```

或从源码安装：

```bash
git clone https://github.com/laodongX/Fidelity-metrics.git
cd Fidelity-metrics
pip install -e .
```

## 快速开始

### 1. 直接度量两层隐状态的保真度

```python
import torch
from fidelity_metrics import SemanticFidelityProbe

probe = SemanticFidelityProbe(dim=768)

# 模拟两层隐状态 [B, S, D]
z_layer4 = torch.randn(2, 512, 768)
z_layer8 = torch.randn(2, 512, 768)

report = probe.measure(z_layer4, z_layer8)
# {'structural': 0.72, 'distributional': 0.91, 'combined': 0.815}
```

### 2. 用 Hook 自动扫描模型各层

```python
import torch
from fidelity_metrics import SemanticFidelityProbe, LayerwiseFidelityHook

probe = SemanticFidelityProbe(dim=768)
hook = LayerwiseFidelityHook(model, probe)

# 指定要监控的层
hook.register_hooks(["blocks.0", "blocks.4", "blocks.8"])

with torch.no_grad():
    _ = model(val_ids)

report = hook.get_report()
# [{'layers': 'blocks.0->blocks.4', 'structural': 0.85, ...}, ...]

hook.remove_hooks()
```

## 两个核心指标

| 指标 | 含义 | 范围 |
|:---|:---|:---|
| **structural** | KNN 近邻保持率：源空间的邻居在目标空间还是邻居吗？ | 0~1 |
| **distributional** | 投影分布匹配度：均值和方差是否偏移？ | 0~1 |
| **combined** | 两者平均 | 0~1 |

**诊断规则：**
- `combined ≥ 0.8` → ✅ 信息健康传播
- `0.6 ≤ combined < 0.8` → ⚠️ 信息衰减，关注该层
- `combined < 0.6` → 🚨 信息断流，检查门控或路由

## API

### SemanticFidelityProbe(dim, num_projections=256, eps=1e-4)

| 参数 | 说明 |
|:---|:---|
| `dim` | 隐状态维度 |
| `num_projections` | 分布保真度的投影方向数 |
| `eps` | 数值稳定项 |

**方法：**

- `measure(z_source, z_target, k=5)` → `dict` 返回三个保真度得分
- `structural_fidelity(z_source, z_target, k=5)` → `float`
- `distributional_fidelity(z_source, z_target)` → `float`

### LayerwiseFidelityHook(model, probe)

| 参数 | 说明 |
|:---|:---|
| `model` | `nn.Module` 实例 |
| `probe` | `SemanticFidelityProbe` 实例 |

**方法：**

- `register_hooks(layer_names)` 注册 forward hook（自动清理旧 hook）
- `get_report()` → `List[Dict]` 返回相邻层对的保真度报告
- `remove_hooks()` 移除所有 hook

## 许可证

MIT
