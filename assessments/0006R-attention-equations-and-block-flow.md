# 0006R 闭卷复测：公式轴与 Decoder Block 路径

关闭课程和代码后作答。

## 1. 两个逐元素公式（关键题，45 分）

给定 `Q,K,V:[B,S,D]`，完整写出 scaled score 和 output 的逐元素公式。公式必须带 `b,q,k,d` 索引、`sqrt(D)`，并分别说明哪个轴被求和、输出留下哪些轴。

## 2. Decoder block 路径（关键题，40 分）

从 `hidden_states` 开始，按顺序写出第一段 RMSNorm、Q/K/V projection、causal attention、output projection、第一次 residual，以及第二段 RMSNorm、gated FFN、第二次 residual。最后说明还要经过什么组件才得到词表 logits。

## 3. 代码 contract（15 分）

粘贴 `exercises/0006R_attention_contract.py` 的运行输出，并解释：

1. 为什么检查条件是 `D <= 0` 而不是 `D < 0`？
2. 为什么 mask 应建在 `query.device`？

## 通过标准

- 独立代码、第 1、2 题全部通过。
- 总分至少 80。
