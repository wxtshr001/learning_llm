# 0006R 反馈

## 结论

**95/100，通过。第 0006 课正式完成。**

0006R 约定复测的三个关键项全部通过：两个求和轴的公式语义、Attention 在 decoder block 中的完整路径，以及 positive-D/device 代码 contract。第 0006 课首次提交中已经通过的手算、shape、causal 行为、scale 与 PyTorch 主计算继续保留，不重复评分。

按照学习者此前“通过后不用生成新课程，只保存进度”的要求，本次只保存完成状态，不生成第 0007 课。

## 1. 两个逐元素公式：40/45，关键题通过

- scaled score 写出了 `Q[b,q,d]`、`K[b,k,d]`、求和、`sqrt(D)`，并明确说明沿 feature 轴 D 求和、留下 `b,q,k`。
- output 写出了 softmax weight 与 `V[b,k,d]` 的乘积、沿 k 求和，并明确留下 `b,q,d`。
- `sum 0->d-1` 的上界应规范写为 `sum d=0..D-1`；output 公式还缺一个右括号，`softmax(scores[b,q,:])[k]` 的下标也可写得更清楚。这些是局部符号书写问题；轴含义和计算对象均明确正确，因此不升级为关键概念失败。

推荐的规范记法：

```text
score[b,q,k] = sum_{d=0}^{D-1} Q[b,q,d] * K[b,k,d] / sqrt(D)
output[b,q,d] = sum_{k=0}^{S-1} weights[b,q,k] * V[b,k,d]
```

## 2. Decoder block 路径：40/40，关键题通过

`tmp/test` 中提交的路径完整包含：

```text
hidden_states
→ RMSNorm
→ Q/K/V projections
→ causal attention
→ merge heads / output projection
→ 第一次 residual add
→ RMSNorm
→ gated FFN
→ 第二次 residual add
```

并正确补充多个 decoder layers 后还需 final RMSNorm 与 LM Head，才得到词表 logits。

## 3. 代码 contract：15/15

- `validate_head_dim()` 使用 `head_dim <= 0`，正确拒绝 D=0。
- mask 使用 `device=query.device`，正确跟随输入设备。
- Agent 使用 PyTorch 2.13.0+cpu 复跑 `0006R_attention_contract.py`，全部测试通过。
- 正确解释 tensor shape 不允许 D=0，以及参与同一计算的 mask 与 query 必须位于同一 device。

## 最终裁决

第 0006 课已证明：Q/K/V 分工、scaled dot-product score、causal mask、softmax key axis、value 加权、两个求和轴、因果行为测试，以及 Attention 子层在 decoder residual stream 中的结构位置。
