# 0009 闭卷测试：Tiny Decoder Layer

完成 notebook、引导脚本和独立代码后，关闭课程、代码与速查表作答。以下各题只讨论本课的**一个** Tiny Decoder Layer，不讨论 embedding、final norm、LM Head 或多个 layer。

## 全卷共同前提

- 本层接收 `hidden_states=x0 [B,S,H]`（当前一批 token 的 hidden states）和 `position_ids [B,S]`（这些 token 的位置编号）。`B` 是 batch 中的序列条数，`S` 是本次输入的 token 数，`H` 是每个 token 的 hidden 宽度。
- 本层由两套**独立**的 `TinyRMSNorm`（`input_norm`、`post_attention_norm`）、一个 `TinyGQAAttention`（`q_proj/k_proj/v_proj/o_proj`，内部对 Q/K 使用 RoPE 和 causal mask）和一个 `GatedFFN`（`gate_proj/up_proj/down_proj`）组成。`Nq` 是 Q head 数，`Nkv` 是 K/V head 数，`D` 是每个 head 的宽度，`I` 是 FFN 内部宽度。本课规定 `H=Nq*D`。
- 各 projection 均为**无 bias** 的 Linear；每个 RMSNorm 有一个可学习的 `weight`。0009 的每次调用都是完整序列 forward，不读写跨调用的 KV Cache。比较两次调用时，模型参数和 `position_ids` 保持相同。第 4 题最后一小题才假设下一课加入 Cache。
- `x1` 表示第一次 residual add 后的 hidden states；`x2` 表示第二次 residual add 后的本层输出。只要求本层的模块、数据流与对象分类；不要求重写 RoPE 内部公式或实现 Cache。

## 提交证据

粘贴引导脚本与独立脚本的完整运行输出。

## 1. 两条 pre-norm residual 路径（关键题，25 分）

1. 从 `x0=hidden_states` 开始，按执行顺序写出 `attention_input`、`attention_output`、`x1`、`ffn_input`、`ffn_output`、`x2` 分别由哪个模块或加法得到。把两次 residual add 的左右操作数写清楚。
2. `residual_0` 和 `residual_1` 在两次加法前分别指向哪个已有 tensor（`x0` 还是 `x1`）？
3. 如果第二次加法写成 `x0 + ffn_output`，会漏掉第一次 residual add 中哪一个直接加回主干的项？
4. `attention_output` 和 `ffn_output` 各要与哪个 `[B,S,H]` tensor 相加？说明 `o_proj` 和 `down_proj` 各自负责什么。这里 `Nq*D=H`，所以 `merged` 与主干的**数值宽度相同**；请区分“宽度相同”与“已经经过 `o_proj` 的学习组合”。`I` 则是 FFN 内部宽度。

## 2. 完整 shape 追踪（关键题，30 分）

本题把共同前提中的符号具体设为 `B=2,S=5,H=12,Nq=3,Nkv=1,D=4,I=20`，即 2 条序列、每条 5 个 token，每个 Q/K/V head 宽 4，且 `H=3*4=12`。按一次完整序列 forward 作答：

1. 按顺序写出 `x0`、`attention_input`、`q_raw/k_raw/v_raw`（各 projection 的三维输出）、split 后的 `Q/K/V`、causal attention 的 `scores/weights`、各 Q head 的 `head_output`、合并 head 后的 `merged`、经 `o_proj` 的 `attention_output`、`x1` 的 shape。`scores/weights` 的两个位置轴都是当前长度 `S=5`。
2. 接着写出 `ffn_input`、`gate_proj/up_proj` 输出及相乘后的 `mixed`、经 `down_proj` 的 `ffn_output`、`x2` 的 shape。
3. 用上面的具体数字说明 B、S、H、Nq、Nkv、D、I 各表示什么；在至少一个四维 shape 中标明每个轴的含义。
4. 在第 1、2 小题列出的 tensor 中，指出哪些有单独的 head axis，哪些只有扁平的 hidden 或 intermediate 特征轴。

## 3. token mixing 与因果性（关键题，25 分）

考虑 `B=1,S=5` 的一次完整序列 forward。只修改 `hidden_states` 中下标 4 的向量；其他位置的输入、`position_ids=[0,1,2,3,4]` 和模型参数保持不变。比较修改前后**同一层**的输出：

1. `input_norm(x0)` 的位置 0～3 会不会改变？说明它读取哪些轴。
2. `self_attn(input_norm(x0), position_ids)` 的位置 0～3 会不会改变？结合 causal mask 说明这些 query 能读取哪些 key/value 位置。
3. 本层最终输出 `x2` 的位置 0～3 会不会改变？结合两条 residual 路径说明。
4. 如果改为只修改下标 0 的输入，`x2` 的位置 1～4 是否可能改变？指出本层中负责跨 token 读取的组件。

## 4. Parameter、activation、state 与执行阶段（20 分）

1. 只针对“全卷共同前提”中列出的**这个 layer**，按 `input_norm`、`post_attention_norm`、`TinyGQAAttention`、`GatedFFN` 分组，列出各模块拥有的可学习 Parameter **名称**。不要求写 shape、具体数值，也不统计模块外的 Parameter；题目已说明 Linear 无 bias。
2. 在本次 forward 中，分别把 `position_ids`、split 后的 Q/K/V、由位置计算出的 cos/sin、attention weights、`residual_0`、`residual_1`、FFN 的 `mixed` 归入“运行输入 / 本次计算的 activation / 模型 Parameter / 跨调用请求 state”。如果 residual 名称只是指向已有 tensor，请指出它分别指向什么；无需假设发生了额外复制。
3. 就当前 0009 代码回答：本层在一次调用结束后，是否自行保存供下一次调用读取的 K/V？区分本次算出的 K/V 与跨调用请求 state。
4. **假设**下一课加入 KV Cache：历史 K/V 中什么内容会跨 decode step 保存？再以“prompt 有 5 个 token，之后每步生成 1 个 token”为例，分别给出不保存 Cache 的 full-sequence forward、建立 Cache 的 prefill、单步 decode 的**本次输入** `S`。这里不要求实现 Cache，也不把累计历史长度算作单步 decode 的 `S`。

## 通过标准

- 独立代码与第 1、2、3 题全部通过；
- 总分至少 80；
- 不要求实现 KV Cache 增量更新，不考训练 backward，也不考多个 layer 堆叠。
