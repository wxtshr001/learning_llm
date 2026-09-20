"""Build the interactive lesson 0009 notebook with standard-library JSON."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "0009-tiny-decoder-layer.ipynb"
cells: list[dict[str, object]] = []


def markdown(source: str) -> None:
    cells.append({"cell_type": "markdown", "id": f"md-{len(cells):02d}", "metadata": {}, "source": dedent(source).strip() + "\n"})


def code(source: str) -> None:
    cells.append({"cell_type": "code", "execution_count": None, "id": f"code-{len(cells):02d}", "metadata": {}, "outputs": [], "source": dedent(source).strip() + "\n"})


markdown("""
# 0009：用可执行代码组装 Tiny Decoder Layer

本课不重写已经通过的 RMSNorm、GQA、RoPE 与 gated FFN；只学习一个新能力：**按正确顺序组成两条 pre-norm residual 支路**。

使用方法：每个“先预测”先在纸上回答，再运行紧随其后的代码单元。最后独立完成 `../exercises/0009_tiny_decoder_layer.py`。
""")

markdown("""
## 0. 所有字母与固定例子

| 字母 | 含义 | 本课值 |
|---|---|---:|
| B | batch size，同时处理的序列数 | 1 |
| S | sequence length，每条序列 token 数 | 3 |
| H | hidden size，residual stream 宽度 | 4 |
| Nq | query head 数 | 2 |
| Nkv | key/value head 数 | 1 |
| D | head dimension | 2 |
| I | FFN intermediate size | 6 |

`H=Nq*D=4`。layer 的输入输出始终是 `[B,S,H]=[1,3,4]`。
""")

code("""
from pathlib import Path
import sys
import torch

root = Path.cwd()
if root.name == "notebooks":
    root = root.parent
exercise_dir = str(root / "exercises")
if exercise_dir not in sys.path:
    sys.path.insert(0, exercise_dir)

from decoder_components import GatedFFN, TinyGQAAttention, TinyRMSNorm, fill_deterministic_parameters

torch.set_printoptions(precision=4, sci_mode=False)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)
""")

markdown("""
## 1. residual 是“主干 + 支路修正量”

**先预测：** `x=[2,0,1,3]`，支路输出 `delta=[0.1,0.2,-0.1,0]`，相加结果是什么？为什么两者最后一维必须同为 H？
""")

code("""
x = torch.tensor([2.0, 0.0, 1.0, 3.0])
delta = torch.tensor([0.1, 0.2, -0.1, 0.0])
updated = x + delta
print("x:", x)
print("delta:", delta)
print("x + delta:", updated)
torch.testing.assert_close(updated, torch.tensor([2.1, 0.2, 0.9, 3.0]))
""")

markdown("""
## 2. 建立本课已有组件

两个 RMSNorm 必须是两个不同 module。Attention 内部仍执行 projection、split、RoPE、causal GQA、merge、o_proj；FFN 内部仍执行 gate/up/mixed/down。
""")

code("""
B, S, H, Nq, Nkv, D, I = 1, 3, 4, 2, 1, 2, 6
hidden_states = torch.tensor(
    [[[1.0, 2.0, 3.0, 4.0],
      [2.0, 0.0, 1.0, 3.0],
      [4.0, 1.0, 0.0, 2.0]]],
    device=device,
)
position_ids = torch.tensor([[0, 1, 2]], device=device)

input_norm = TinyRMSNorm(H).to(device)
self_attn = TinyGQAAttention(H, Nq, Nkv, D).to(device)
post_attention_norm = TinyRMSNorm(H).to(device)
mlp = GatedFFN(H, I).to(device)
for module in (input_norm, self_attn, post_attention_norm, mlp):
    fill_deterministic_parameters(module)

print("hidden_states:", hidden_states.shape)
print("two distinct norms:", input_norm is not post_attention_norm)
assert input_norm is not post_attention_norm
""")

markdown("""
## 3. 第一条 pre-norm 支路

```text
residual_0 = x0
attention_input = input_norm(x0)
attention_output = self_attn(attention_input, position_ids)
x1 = residual_0 + attention_output
```

**先预测：** norm 会不会改变 shape？Attention 内部哪些 tensor 有 head axis？`o_proj` 后为什么必须回到 H？
""")

code("""
residual_0 = hidden_states
attention_input = input_norm(hidden_states)
attention_trace = self_attn.trace(attention_input, position_ids)
attention_output = attention_trace["output"]
x1 = residual_0 + attention_output

for name in ("query", "key", "value", "weights", "head_output", "merged", "output"):
    print(f"{name:>12}:", tuple(attention_trace[name].shape))
print("x1:", tuple(x1.shape))
assert attention_input.shape == attention_output.shape == x1.shape == (B, S, H)
""")

markdown("""
## 4. 看一个 token 的第一次相加

注意 residual 左边是未归一化的 `x0`，不是 `attention_input`。
""")

code("""
token = 1
print("x0:", residual_0[0, token])
print("norm(x0):", attention_input[0, token])
print("attention branch:", attention_output[0, token])
print("x1 = x0 + branch:", x1[0, token])
torch.testing.assert_close(x1, residual_0 + attention_output)
""")

markdown("""
## 5. 第二条 pre-norm 支路

```text
residual_1 = x1
ffn_input = post_attention_norm(x1)
ffn_output = mlp(ffn_input)
x2 = residual_1 + ffn_output
```

**先预测：** 为什么 residual_1 必须是 x1，而不是最早的 x0？FFN 的 I=6 在哪一步出现、在哪一步消失？
""")

code("""
residual_1 = x1
ffn_input = post_attention_norm(x1)
ffn_trace = mlp.trace(ffn_input)
ffn_output = ffn_trace["output"]
x2 = residual_1 + ffn_output

for name in ("gate_raw", "gate", "up", "mixed", "output"):
    print(f"{name:>10}:", tuple(ffn_trace[name].shape))
print("x2:", tuple(x2.shape))
assert ffn_trace["mixed"].shape == (B, S, I)
assert ffn_output.shape == x2.shape == (B, S, H)
""")

markdown("""
## 6. 同一个 token 完成整个 layer

不要背小数，只检查数据依赖：`x0 → x1 → x2`，第二条支路不能绕过 x1。
""")

code("""
print("x0:              ", residual_0[0, token])
print("attention branch:", attention_output[0, token])
print("x1:              ", x1[0, token])
print("ffn branch:      ", ffn_output[0, token])
print("x2:              ", x2[0, token])
torch.testing.assert_close(x2, x1 + ffn_output)
""")

markdown("""
## 7. 哪一步跨 token

RMSNorm、Linear、RoPE、FFN 和 residual 都逐 token 工作。只有 causal Attention 汇集多个历史位置。

**先预测：** 只把未来 token 2 改大，位置 0 和 1 的最终输出会不会改变？
""")

code("""
changed = hidden_states.clone()
changed[:, 2, :] += 1000
changed_attention_input = input_norm(changed)
changed_attention_output = self_attn(changed_attention_input, position_ids)
changed_x1 = changed + changed_attention_output
changed_x2 = changed_x1 + mlp(post_attention_norm(changed_x1))

print("earlier outputs unchanged:", torch.allclose(x2[:, :2], changed_x2[:, :2]))
print("last output changed:", not torch.allclose(x2[:, 2], changed_x2[:, 2]))
torch.testing.assert_close(x2[:, :2], changed_x2[:, :2])
assert not torch.allclose(x2[:, 2], changed_x2[:, 2])
""")

markdown("""
## 8. identity 测试

若 Attention 的 `o_proj` 和 FFN 的 `down_proj` 全为零，两条支路最终都输出零。两次 residual 后，整个 layer 必须等于输入。
""")

code("""
zero_attn = TinyGQAAttention(H, Nq, Nkv, D).to(device)
zero_mlp = GatedFFN(H, I).to(device)
with torch.no_grad():
    zero_attn.o_proj.weight.zero_()
    zero_mlp.down_proj.weight.zero_()
zero_x1 = hidden_states + zero_attn(input_norm(hidden_states), position_ids)
zero_x2 = zero_x1 + zero_mlp(post_attention_norm(zero_x1))
torch.testing.assert_close(zero_x2, hidden_states)
print("zero branches -> identity:", True)
""")

markdown("""
## 9. Parameter、activation 与 state

- Parameter：两个 norm weight；q/k/v/o projection；gate/up/down projection。
- 运行输入：hidden_states、position_ids。
- Activation：所有 norm/branch/attention/FFN 中间 tensor，以及 x1/x2。
- 0009 请求 state：没有。0010 的历史 K_rope/V Cache 才跨 decode steps 保存。
""")

code("""
modules = {"input_norm": input_norm, "self_attn": self_attn, "post_attention_norm": post_attention_norm, "mlp": mlp}
for module_name, module in modules.items():
    print(f"[{module_name}]")
    for name, parameter in module.named_parameters():
        print(" ", name, tuple(parameter.shape))
""")

markdown("""
## 10. 完整 forward 骨架

```python
residual_0 = hidden_states
x1 = residual_0 + self_attn(input_norm(hidden_states), position_ids)

residual_1 = x1
x2 = residual_1 + mlp(post_attention_norm(x1))
return x2
```

现在运行引导脚本，再独立完成 TODO：

1. `../exercises/0009_explore_decoder_layer.py`
2. `../exercises/0009_tiny_decoder_layer.py`
3. [速查表](../reference/0009-tiny-decoder-layer-cheatsheet.html)
4. [闭卷题](../assessments/0009-tiny-decoder-layer.md)
5. [提交模板](../submissions/0009.md)

核实来源：[PyTorch nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)；[Qwen3 modeling source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)。
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
