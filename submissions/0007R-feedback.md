# 0007R 反馈

## 结论

**98/100，通过。第 0007 课正式完成，进入第 0008 课 RoPE。**

0007R 公布的两个门禁均已通过：独立 shape contract 脚本通过；闭卷题能够稳定区分 raw、split、logical、head output、merge 与 `o_proj`，并正确解释显式 head axis 和 concat/average 的差异。本次没有复测已通过的 RoPE/系统路径、attention 数值或 Cache 计算。

## 代码：关键项通过

使用课程 Conda 环境重新运行：

```text
Lesson 0007R guided stage trace passed.
Lesson 0007R: all GQA stage-shape checks passed.
```

- 正确验证所有输入为正数、`Nq % Nkv == 0` 和 `H == Nq*D`。
- raw、split、logical、scores/weights、head output、merge 与 `o_output` 的数字 shape 全部正确。
- 三组非法输入均正确抛出 `ValueError`。

## 闭卷题：98/100，关键题通过

### 1. raw projection：正确

`hidden=[1,6,24]`、`q_raw=[1,6,24]`、`k_raw/v_raw=[1,6,12]` 全部正确，已经修正首次 0007 提交中把 K/V raw width 写成 24 的问题。

### 2. split 到 output 的完整 shape：正确

Q/K/V split、logical K/V、scores/weights、head output、merge 和 `o_proj` output 全部正确；同时写出了 reshape、transpose、repeat 和 matmul 的具体来源。

### 3. 显式 Nq head axis：正确

正确列出 Q split、logical K/V、scores/weights 和 head output；也明确说明 merge 与 `o_proj` 已是 rank-3 `[B,S,width]`，没有独立 head axis。

### 4. Nkv 存储与 raw width：18/20

正确指出 Cache 持久保存 `k_split/v_split:[B,Nkv,S,D]`，raw width 12 来自 `Nkv*D=3*4`，且 raw K/V 尚无独立 head axis。

唯一局部术语问题：`raw_k/raw_v` 的 shape 是 `(1,6,12)`，因此它们是 **rank 3 tensor**，不是“2D 矩阵”。你同时写出了完整三维 shape 和正确 axis 结论，所以这是术语笔误，不构成概念门禁。

### 5. concat/average：正确

concat 为 `[2,4,20,40]`，average 为 `16.5`；并正确解释平均会丢失各 head 特征，后续 Linear 无法恢复。

## 最终裁决

第 0007 课现已证明：MHA/GQA/MQA 边界、连续 Q→KV 分组、projection width、各阶段 rank/axis、per-head attention、concat/merge、K/V 持久存储、Cache 容量以及 decode 带宽意义。

RoPE 与完整 Cache 增量执行没有从本次补强中偷渡验收；RoPE 将在 0008 从零教学后单独门禁。
