# 0005R 反馈

## 结论

**90/100，通过。第 0005 课正式完成。**

RMSNorm 统计量、Parameter、输出 tensor 的分类，以及 Linear 构造参数、weight storage、运行 tensor 三种 shape 均已掌握。允许两处局部扣分：第 1 题最终 output 沿用了引导例子的 H=4；第 2 题漏写逐元素公式。两处都没有推翻跨题一致的 H=6/I=14 推理链，因此按课程评分一致性规则不升级为关键概念失败。

## 逐题评分

### 1. RMSNorm 三类 shape：35/40，关键题通过

- `mean_square=[3,2,1]`，正确。
- `weight=[6]`，正确。
- output 应为 `[3,2,6]`，答卷写成 `[3,2,4]`；结合 weight、索引公式和第 2 题均稳定使用 H=6，判为沿用旧例数字的单次抄写错误。
- `mean_square[b,s,0]` 与 `weight[h]` 索引正确。
- 正确区分每 token 统计量和跨 B/S 共享的训练 Parameter。

### 2. Gated FFN 三种写法：45/50，关键题通过

| 层 | 构造 | weight | 运行 tensor |
|---|---|---|---|
| gate | `Linear(6,14)` | `[14,6]` | `[4,5,6] -> [4,5,14]` |
| up | `Linear(6,14)` | `[14,6]` | `[4,5,6] -> [4,5,14]` |
| down | `Linear(14,6)` | `[6,14]` | `[4,5,14] -> [4,5,6]` |

逐元素解释公式未填写，扣 5 分；但全部数字 shape 已证明第一轴是 out、第二轴是 in 的应用能力。

### 3. 分类检查：10/10

- mean_square：统计量；
- norm.weight：Parameter；
- gate_raw：运行 tensor；
- gate_proj.weight：Parameter。

全部正确。

## 下一步

进入第 0006 课：单头 causal attention。第 0005 课已证明的 RMSNorm/FFN 内容不再重复教学，仅在 decoder 组合时复用。
