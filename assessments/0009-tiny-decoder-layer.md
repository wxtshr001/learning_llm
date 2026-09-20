# 0009 闭卷测试：Tiny Decoder Layer

完成 notebook、引导脚本和独立代码后，关闭课程、代码与速查表作答。

## 提交证据

粘贴引导脚本与独立脚本的完整运行输出。

## 1. 两条 pre-norm residual 路径（关键题，25 分）

1. 从 `hidden_states=x0` 开始，依次写出 `attention_input`、`attention_output`、`x1`、`ffn_input`、`ffn_output`、`x2` 的计算式。
2. 哪两个对象分别被保存为 `residual_0` 和 `residual_1`？
3. 为什么不能把第二次 residual 写成 `x0 + ffn_output`？
4. 为什么 Attention 和 FFN 的最终输出都必须回到 H，而不能停留在 `Nq*D` 或 I？

## 2. 完整 shape 追踪（关键题，30 分）

给定 `B=2,S=5,H=12,Nq=3,Nkv=1,D=4,I=20`，并且 `H=Nq*D`：

1. 写出 `x0`、`attention_input`、q/k/v raw、Q/K/V split、scores/weights、head output、merged、`attention_output`、`x1` 的 shape。
2. 写出 `ffn_input`、gate/up/mixed、`ffn_output`、`x2` 的 shape。
3. 标出每个 shape 中 B、S、H、Nq、Nkv、D、I 分别表示什么。
4. 哪些 tensor 有显式 head axis？哪些只有扁平 hidden/intermediate axis？

## 3. token mixing 与因果性（关键题，25 分）

一条长度为 5 的序列中，只把最后一个 token（下标 4）的 hidden state 改掉：

1. `input_norm` 的位置 0～3 输出会不会改变？为什么？
2. causal Attention 的位置 0～3 输出会不会改变？为什么？
3. 完整 layer 的位置 0～3 输出会不会改变？为什么？
4. 若只改变 token 0，后面位置 1～4 是否可能改变？指出发生跨 token 传播的唯一组件。

## 4. Parameter、activation、state 与执行阶段（20 分）

1. 列出本层两类 RMSNorm Parameter、Attention Parameter 和 FFN Parameter。
2. `position_ids`、Q/K/V、cos/sin、weights、两个 residual 和 FFN mixed 各属于什么？
3. 0009 的 full-sequence layer 是否保存跨调用请求 state？
4. 到 0010 加入 KV Cache 后，什么会成为请求 state？full-sequence、prefill、单步 decode 的 S 分别可能是什么？

## 通过标准

- 独立代码与第 1、2、3 题全部通过；
- 总分至少 80；
- 不要求实现 KV Cache 增量更新，不考训练 backward，也不考多个 layer 堆叠。
