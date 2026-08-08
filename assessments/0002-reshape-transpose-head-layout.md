# 第 0002 课过关测试：reshape、transpose 与 Head Layout

## 本试卷使用的符号

| 符号 | 英文 | 含义 |
|---|---|---|
| B | Batch size | 同时处理的序列数量 |
| S | Sequence length | 每条序列的 token 数量 |
| H | Hidden size | 每个 token 的总特征数 |
| N | Number of heads | Attention head 数量 |
| D | Head dimension | 每个 head 的特征数，`D=H/N` |

完成两个代码脚本后，关闭课程、源码和速查表，使用纸笔回答。不要运行代码。

## 提交证据

请先粘贴：

1. `0002_explore_head_layout.py` 最后的 `restored equals x` 结果；
2. `0002_head_layout.py` 的完整输出。

## 1. 从 hidden 拆成 heads（关键题）

```text
x.shape = [B=2, S=5, H=12]
N=3
```

- `D` 等于多少？写出计算。
- reshape 后的 `[B,S,N,D]` 是什么数字 shape？
- 再执行 `transpose(1,2)` 后的 `[B,N,S,D]` 是什么数字 shape？
- 为什么不能 reshape 为 `[2,5,3,5]`？请比较变换前后的元素总数。

## 2. dim/axis 的含义（关键题）

对 <code>[B,S,N,D] = [2,5,3,4]</code> 执行 `transpose(1,2)`：

- dim 1 原来表示什么？长度是多少？
- dim 2 原来表示什么？长度是多少？
- 输出 shape 是什么？输出的四个轴依次表示什么？
- transpose 是否修改任何轴本身的长度？

## 3. 元素索引映射（关键题）

设：

```text
x.shape = [B,S,H]
H=N*D
heads = x.reshape(B,S,N,D).transpose(1,2)
```

- 请补全：`heads[b,n,s,d] = x[ ?, ?, ? ]`。
- 当 `D=8` 时，`heads[1,2,4,3]` 对应 x 的哪个完整索引？

## 4. 为什么使用 `[B,N,S,D]`

用自己的话解释：为什么 Attention 更方便把 heads 放在 sequence 之前？提示：固定 `b` 和 `n` 后，`heads[b,n,:,:]` 的 shape 是什么？它包含什么？

## 5. view、reshape 与 contiguous

- transpose 通常是否复制底层 storage？
- 为什么 transpose 后直接 `view()` 可能失败？
- `contiguous()` 做了什么？
- 为什么官方文档说不要依赖 `reshape()` 一定返回 view 或一定复制？

## 6. 具体数值追踪

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
```

- `x.shape` 和 `y.shape` 分别是什么？
- 根据 transpose 的轴交换，`y[1,2,1]` 对应 `x` 的哪个索引？
- `x` 按 0～23 顺序填充，因此这个元素的数值是多少？

## 7. 迁移到 Q/K/V layout（关键题）

```text
B=2
S=5
query_heads=4
kv_heads=2
D=32
```

Q、K、V 完成 projection、reshape 和 transpose 后分别是什么数字 shape？本题只检查 layout，暂时不解释 GQA 如何共享 K/V。

## 通过标准

- 两个脚本通过。
- 题 1、2、3、7 必须正确；总分至少 80%。
- 如果只是 view/reshape API 边界不稳，安排短补强。
- 如果 B/S/H/N/D 的含义、轴顺序或元素映射错误，则用更小 tensor 重新演示。
