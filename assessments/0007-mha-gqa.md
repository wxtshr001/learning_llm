# 0007 闭卷测试：MHA、GQA 与 KV Cache 成本

完成引导和独立代码后，关闭课程、代码与速查表作答。

## 提交证据

粘贴两个脚本的完整运行输出。

## 1. Head 映射与边界（关键题，20 分）

给定 `Nq=12,Nkv=3`：

1. 计算 `group_size`；
2. 写出 Q head 0～11 分别映射到哪个 KV head；
3. 说明它为什么既不是 MHA 也不是 MQA；
4. 分别写出相同 `Nq=12` 时 MHA 与 MQA 的 `Nkv`。

## 2. 完整 shape 数据流（关键题，25 分）

给定 `B=2,S=7,H=24,Nq=6,Nkv=2,D=4`，写出 hidden_states、q/k/v projection raw output、split 后 Q/K/V、K/V 映射后的逻辑 shape、scores/weights、per-head output、merge output 和 o_proj output。

说明哪些 tensor 的 head axis 是 Nq，哪些持久保存为 Nkv，以及为什么不能把 6 个 head output 求平均。

## 3. 逐 head 数值迁移（15 分）

`B=1,Nq=4,Nkv=2,S=2,D=1`，Q/K 全为 0。KV head 0 的 `V=[2,6]`，KV head 1 的 `V=[20,60]`。

1. 写出 q0、q1 的 causal weights；
2. 写出四个 Q head 在 q0、q1 的 output；
3. 写出 merge 后 `[B,S,Nq*D]` 的具体数值。

## 4. Projection 参数（10 分）

给定 `H=24,Nq=6,Nkv=2,D=4`，忽略 bias：

1. 写出 q/k/v/o projection 的 `Linear(in,out)`、weight shape `[out,in]` 和参数数；
2. 与 `Nkv=Nq=6` 的 MHA 相比，哪些 projection 参数减少，哪些不变？

## 5. KV Cache 字节（关键题，20 分）

给定 `B=2,L=24,S_cache=2048,D=64,FP16=2 bytes,Nq=16`：

1. 写出 K+V Cache 字节公式；
2. 分别计算 MHA `Nkv=16` 和 GQA `Nkv=4` 的 bytes 与 MiB；
3. 计算节省的 MiB 和 GQA/MHA 容量比例；
4. 说明 Cache 应保存 repeat 前还是 repeat 后的 K/V，为什么？

## 6. Transformer inference 迁移（关键题，10 分）

1. 解释 GQA 为什么在单步 decode 中尤其重要；
2. 为什么不能仅凭 `Nkv` 从 16 降到 4 就断言端到端生成速度提升 4 倍？
3. full-sequence/prefill 的 attention weights head axis 是 Nq 还是 Nkv？为什么？
4. 从 hidden_states 到 residual add，写出 GQA 子层的完整顺序，并指出 RoPE 与 KV Cache 将在这条路径的哪里接入。

## 通过标准

- 独立代码与第 1、2、5、6 题全部通过；
- 总分至少 80；
- 未通过时只补 head mapping、shape、Cache 计算或系统执行路径中的实际缺口。
