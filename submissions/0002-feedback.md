# 0002 反馈

## 结论

**66/100，未通过；进入 0002R 针对性重学。**

代码输出证明 `split_heads()` 与 `merge_heads()` 的主路径能够完成往返变换，且原有 11 项检查通过。但是第 3、7 题是关键题：第 3 题没有给出要求的直接公式和具体索引，第 7 题留空并明确表示不理解 Q/K/V projection 与 head layout 的连接，因此不能进入第 0003 课。

## 分项评分

| 证据 | 得分 | 说明 |
|---|---:|---|
| 概念解释 | 13/20 | 第 4 题方向正确；第 5 题对 `contiguous()`、`reshape()` 有基本理解，但 `transpose` 在本题中返回 view、通常不复制 storage，应明确回答“通常不复制”，不能只写“不一定”。 |
| shape/计算 | 23/30 | 第 1、2 题正确；第 3 题未完成直接映射和指定数字；第 6 题索引与 offset 已推到 18，却最终写成 17。 |
| 实现与测试 | 30/35 | 主路径和已有 11 项检查通过；缺少题目明确要求的 rank 检查与 `num_heads > 0` 检查。当前输出来自 CPU 环境，本机未安装 PyTorch，Agent 无法独立复跑。 |
| 迁移题 | 0/15 | 第 7 题为空，且学习者明确记录不理解 projection、reshape、transpose 的连接。 |

## 逐题反馈

### 题 1：通过

- `D=12/3=4`，正确。
- `[B,S,N,D]=[2,5,3,4]`，正确。
- `[B,N,S,D]=[2,3,5,4]`，正确。
- 原 tensor 有 `2*5*12=120` 个元素，错误目标有 `2*5*3*5=150` 个元素，不能 reshape，正确。

### 题 2：通过

dim 1 是长度 5 的 Sequence 轴，dim 2 是长度 3 的 Head 轴；交换后为 `[2,3,5,4]`，轴依次是 Batch、Head、Sequence、Head dimension。transpose 只移动轴的位置，不改变轴自己的长度。

### 题 3：关键题未通过

你写出的 stride 展开说明已经在尝试从 storage offset 推导，但题目需要的是更直接的逻辑索引关系：

```text
heads[b,n,s,d] = x[b,s,n*D+d]
```

当 `D=8` 时，`heads[1,2,4,3]` 对应 `x[1,4,19]`，因为 `2*8+3=19`。答卷只写“数值代回即可”，没有完成这一问。

### 题 4：基本通过

方向正确。完整表达应包含：固定 `b` 和 `n` 后，`heads[b,n,:,:]` 的 shape 是 `[S,D]`，表示当前序列、当前 head 下全部 `S` 个 token 的 `D` 维向量。

### 题 5：部分通过

- `transpose` 在这里返回共享 storage 的 view，通常不复制数据。
- transpose 后 shape 与 stride 的组合可能不支持目标 `view()`，所以会失败。
- `contiguous()` 按当前逻辑顺序生成连续布局，理解正确。
- `reshape()` 能共享时可返回 view，不能共享时可复制，因此调用者不能依赖其中一种实现，方向正确。

### 题 6：部分通过

shape、stride 和索引映射 `y[1,2,1] = x[1,1,2]` 都正确。该位置的 offset 是 `1*12+1*4+2=18`，`torch.arange(24)` 在 offset 18 的值就是 18；最后一行写成 17，与前面的正确推导矛盾。

### 题 7：关键题未通过

留空，且记录了明确知识缺口。0002R 将只补 projection 输出宽度如何决定 `N` 与 `D`，再迁移到 Q、K、V 的 `[B,N,S,D]`。

## 代码反馈

核心变换正确：

```text
[B,S,H] -> reshape(B,S,N,D) -> transpose(1,2)
[B,N,S,D] -> transpose(1,2) -> contiguous() -> view(B,S,N*D)
```

仍需补上两个输入边界：

1. `x.ndim != 3` 时主动抛出 `ValueError`；
2. `num_heads <= 0` 时主动抛出 `ValueError`。

## 下一步

完成 `lessons/0002R-index-and-qkv-layout.html`、运行 `exercises/0002R_trace_qkv_layout.py`，修正原代码的两个输入边界，然后闭卷提交 `submissions/0002R.md`。
