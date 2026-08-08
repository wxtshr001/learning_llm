# Head layout 部分能力已证明，直接映射与 Q/K/V 迁移待补强

## 已证明

- 能由 `H` 和 `N` 算出 `D`，并保持 reshape 前后元素总数一致。
- 能解释 `transpose(1,2)` 交换 Sequence 与 Head 轴，得到 `[B,N,S,D]`。
- 能实现 split/merge 主路径，提交输出显示原有 11 项检查通过。
- 能根据 transpose 后的 stride 计算具体 storage offset。

## 尚未证明

- 尚未把 stride 推导收敛为 `heads[b,n,s,d] = x[b,s,n*D+d]`，且未完成指定数字的索引。
- 尚不能把 Q/K/V projection 的输出宽度连接到各自的 head 数量与 `[B,N,S,D]`。
- 对 `transpose` 是否复制 storage 的回答不够确定。
- 代码尚未主动处理错误 rank 与非正 `num_heads`。

## Evidence

- `submissions/0002.md`
- `submissions/0002-feedback.md`
