# 直接 head 索引与 Q/K/V layout 迁移已证明

0002R 书面复测 95/100，三个关键概念题全部通过。

## 已证明

- 能直接使用 `heads[b,n,s,d] = x[b,s,n*D+d]`，并在新数字上完成具体索引计算。
- 能将 transpose 后的逻辑索引映射回原 tensor，并正确区分 storage offset 与元素数值。
- 能从 `query_heads*D` 与 `kv_heads*D` 算出 Q/K/V projection 输出宽度。
- 能分别推出 Q 的 `[B,Nq,S,D]` 与 K/V 的 `[B,Nkv,S,D]`。
- 基本理解 transpose view、stride、contiguous 与 reshape 的边界。

## 尚未完成的门禁

- `split_heads()` 的 rank 检查写在 shape 解包之后，错误 rank 时主动检查不可达。只需修正 Python 检查顺序并提供运行证据，不需要新的概念课程。

## Evidence

- `submissions/0002R.md`
- `submissions/0002R-feedback.md`
