# 0005 闭卷测试：RMSNorm 与 Gated FFN

完成探索脚本和独立作业后，关闭课程、源码和速查表作答。

## 符号表

- `B`：Batch size，序列批次数。
- `S`：Sequence length，每条序列的 token 数。
- `H`：Hidden size，每个 token 的输入/输出特征数。
- `I`：Intermediate size，FFN 内部特征数。
- `ε`：epsilon，防止除零的小正数。
- `γ`：gamma，RMSNorm 的逐特征 weight。

## 提交证据

1. 粘贴 `0005_explore_rmsnorm_gated_ffn.py` 的完整输出。
2. 粘贴完成 TODO 后 `0005_rmsnorm_gated_ffn.py` 的完整输出。

## 1. RMSNorm 手算（关键题，25 分）

单个 token：

```text
x = [3, 4]
H = 2
ε = 0
γ = [2, 0.5]
```

依次计算：

1. `x^2`；
2. 平方平均值；
3. RMS；
4. `x/RMS`；
5. 最终 `(x/RMS)*γ`，保留四位小数。

## 2. RMSNorm 的 axis 与广播（15 分）

输入 `x.shape=[B=2,S=3,H=4]`：

1. `x.pow(2).mean(dim=-1, keepdim=True)` 的数字 shape 是什么？
2. 每个输出数由输入中的哪些元素计算？
3. RMSNorm weight 的 shape 是什么，怎样广播？
4. RMSNorm 最终输出 shape 是什么？

## 3. Gated FFN shape（关键题，25 分）

```text
B=2, S=5, H=8, I=20
```

写出：

1. gate_proj、up_proj、down_proj 三个 PyTorch Linear weight 的数字 shape；
2. gate_raw、SiLU(gate_raw)、up、mixed 的数字 shape；
3. 最终 output 的数字 shape；
4. 哪一步是逐元素乘法，为什么两边 shape 必须相同？

## 4. 门控数值（15 分）

已经给出 SiLU 结果，不需要自己计算 sigmoid：

```text
SiLU(gate_raw) = [0.0, 0.7311, -0.2689]
up             = [2.0, -3.0, 4.0]
```

计算 mixed 的三个值，保留四位小数。说明这里为什么不能使用矩阵乘法代替 `*`。

## 5. Module、梯度与 parity（10 分）

1. GatedFFN 的哪些对象会出现在 `named_parameters()`？写出名字。
2. 对 output 计算 loss 并 backward 后，为什么 gate_proj 和 up_proj 都应有 grad？
3. “测试通过”为什么还需要 reference parity，而不能只检查没有异常？

## 6. 迁移到 decoder（10 分）

1. RMSNorm 和 gated FFN 是否混合不同 token？为什么？
2. 它们分别操作哪个轴？
3. 为什么 gated FFN 最后要从 I 投影回 H？

## 通过标准

- 总分至少 80 分。
- 独立作业、第 1 题和第 3 题全部通过。
- 必须正确区分 Linear weight `[out,in]` 与 Tensor 数学 shape。
- 未通过时只补 RMS 计算、广播或 gated shape 中的实际缺口。
