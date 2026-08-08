# 第 0002 课过关测试：reshape、transpose 与 Head Layout

完成代码作业后关闭课程、源码和速查表，闭卷回答。可以纸笔推导，不要运行代码。

## 1. reshape（关键题）

```text
x: [batch=2, seq=5, hidden=12]
num_heads=3
```

- `head_dim` 是多少？
- 把 hidden 拆为 heads 和 head_dim 后，shape 是什么？
- 为什么 `[2,5,3,5]` 是非法 reshape？

## 2. transpose（关键题）

对 `[B=2,S=5,N=3,D=4]` 执行 `transpose(1,2)`：

- 输出 shape 是什么？
- 输出四个轴依次表示什么？
- 哪些轴只是交换位置，哪些轴的长度发生了改变？

## 3. 元素映射（关键题）

设：

```text
projected: [B,S,N*D]
heads = projected.reshape(B,S,N,D).transpose(1,2)
```

请用索引表达 `heads[b,n,s,d]` 对应 `projected` 中哪个元素。

## 4. 为什么把 heads 放到 seq 前面

Attention 常把 Q/K 整理为 `[B,N,S,D]`。结合每个 head 独立计算 attention score，解释这种排列为什么方便。

## 5. view、reshape 与 contiguous

- transpose 通常是否复制 tensor 数据？
- 为什么 transpose 后直接 `view()` 可能失败？
- `contiguous().view(...)` 与 `reshape(...)` 各如何处理这个问题？

## 6. 代码阅读

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
```

- `y.shape` 是什么？
- `y[1,2,1]` 对应 `x` 的哪个元素？
- 它们的数值是多少？

## 7. 迁移到 GQA（关键题）

```text
B=2, S=5, query_heads=4, kv_heads=2, head_dim=32
```

Q、K、V 完成 projection、reshape、transpose 后分别是什么 shape？这里只回答 layout，不解释 GQA 共享机制。

## 提交证据与通过标准

- 粘贴练习脚本的完整输出。
- 题 1、2、3、7 和代码必须通过；总分至少 80%。
- 如果只是 `view/reshape` API 边界不稳，安排短补强；如果轴语义或元素映射错误，则降低维度重新练习。
