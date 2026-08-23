# 0007R 闭卷复测：GQA Tensor 阶段与系统路径

关闭课程、代码和速查表后作答。

## 提交证据

粘贴引导脚本与独立脚本的完整运行输出。

## 1. Tensor 阶段与显式 head axis（关键题，50 分）

给定 `B=1,S=6,H=24,Nq=6,Nkv=3,D=4`：

1. 写出 `hidden_states`、q/k/v projection raw output 的 shape；
2. 写出 transpose 后 Q/K/V split tensor、逻辑映射后 K/V、scores/weights、head output、merge 和 `o_proj` output 的 shape；
3. 从以上 tensor 中列出哪些具有显式的 `Nq` head axis；明确说明 merge 和 `o_proj` output 是否仍有 head axis；
4. 哪些 K/V tensor 以 `Nkv` 持久保存在 Cache？raw K/V 的最后一维为什么是 12？raw K/V 此时是否已经有独立 head axis？
5. 另取一个 `Nq=4,D=1` 的最小例子。固定一个 token，四个 head output 分别为 `[2]`、`[4]`、`[20]`、`[40]`。分别写出 concat/merge 和 average 的结果，并解释为什么模型必须使用 concat/merge 后接 `o_proj`，不能直接平均。

## 2. Decoder 中的完整系统路径（关键题，40 分）

1. 从 `hidden_states` 开始，依次写出 RMSNorm、q/k/v projections、split heads、RoPE、KV Cache、GQA causal attention、merge、`o_proj` 和 residual add 的完整顺序；
2. RoPE 作用于 Q/K/V 中的哪些 tensor？它位于 split 前还是 split 后、score 计算前还是后？本题不要求写旋转公式；
3. 在这条路径里各举一个 Parameter、Activation 和跨 decode step 的 State；
4. 说明 GQA 子层输出是否已经是 vocabulary logits；如果不是，它之后还要经过什么大致链条才产生 logits？

## 3. 两个百分比（10 分）

相同 `B,L,S_cache,D,dtype` 下，MHA 为 `Nq=Nkv=12`，GQA 为 `Nq=12,Nkv=3`。计算：

1. `GQA/MHA` Cache 容量比例；
2. Cache 节省比例。

## 通过标准

- 独立脚本、第 1 题和第 2 题全部通过；
- 总分至少 80；
- 不考 RoPE 的旋转数学或实现，只考它在系统路径中的位置。
