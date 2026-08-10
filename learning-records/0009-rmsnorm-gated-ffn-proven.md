# 第 0005 课正式通过：RMSNorm 与 Gated FFN

0005R 最终 90/100，两个关键题均通过，第 0005 课正式完成。

## 已证明

- 能手算并实现 RMSNorm，区分 mean square、weight 与输出 tensor，解释 last-axis reduction 和广播索引。
- 能区分 `Linear(in,out)` 构造、`weight=[out,in]` storage 和运行 tensor 数据流。
- 能实现 gated FFN 的 gate/up/down 三投影、SiLU 和逐元素门控。
- 能执行 RMSNorm/reference parity、Gated FFN 显式公式 parity 和梯度检查。
- 能解释 RMSNorm→FFN→residual 的 decoder 数据流与 token-local 边界。

## 非阻塞提醒

- 0005R 第 1 题 output shape 单次沿用了旧例 H=4，但其他 H=6 证据一致。
- 第 2 题漏写逐元素公式；所有构造、storage 和运行 tensor 数字 shape 正确。

## Evidence

- `exercises/0005_rmsnorm_gated_ffn.py`
- `submissions/0005.md`
- `submissions/0005-feedback.md`
- `submissions/0005R.md`
- `submissions/0005R-feedback.md`
