# Linear shape、索引与手算门禁已通过

学习者已经证明能够推导 Linear 输出 shape、实现矩阵乘法与 bias 广播、展开 `Y[i,j]`，并把相同索引结构迁移到 Q projection。在纠正 `b[i]`/`b[j]` 后，0001R2 的三组逐元素计算全部正确，因此可以进入多维 tensor layout，不再重复二维 Linear 基础。

## Evidence

- 纯 Python Linear 实现的 8 项测试通过。
- 0001R2 的向量输出、2×2 多输出和含负数的无 bias 对照均逐元素计算正确。
