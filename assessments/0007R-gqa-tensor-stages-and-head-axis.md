# 0007R 闭卷复测：GQA Tensor 阶段与显式 Head Axis

关闭课程、代码和速查表后作答。

## 提交证据

粘贴引导脚本与独立脚本的完整运行输出。

## 1. Tensor 阶段与显式 head axis（关键题，100 分）

给定 `B=1,S=6,H=24,Nq=6,Nkv=3,D=4`：

1. 写出 `hidden_states`、q/k/v projection raw output 的 shape；
2. 写出 transpose 后 Q/K/V split tensor、逻辑映射后 K/V、scores/weights、head output、merge 和 `o_proj` output 的 shape；
3. 从以上 tensor 中列出哪些具有显式的 `Nq` head axis；明确说明 merge 和 `o_proj` output 是否仍有 head axis；
4. 哪些 K/V tensor 以 `Nkv` 持久保存在 Cache？raw K/V 的最后一维为什么是 12？raw K/V 此时是否已经有独立 head axis？
5. 另取一个 `Nq=4,D=1` 的最小例子。固定一个 token，四个 head output 分别为 `[2]`、`[4]`、`[20]`、`[40]`。分别写出 concat/merge 和 average 的结果，并解释为什么模型必须使用 concat/merge 后接 `o_proj`，不能直接平均。

## 通过标准

- 独立脚本和第 1 题通过；
- 总分至少 80；
- 不重复考已经通过的 head 映射、attention 数值、Cache 计算或 inference 迁移；也不考完整 decoder/GQA 系统路径和 RoPE。
