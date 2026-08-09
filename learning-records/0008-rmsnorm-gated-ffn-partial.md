# RMSNorm 与 gated FFN 主数据流已证明，shape 分类待补强

## 已证明

- 能完整手算 RMSNorm 的平方、mean square、RMS、归一化与 gamma 缩放。
- 能实现 RMSNorm last-axis reduction、float32 中间计算、逐特征 weight 和 dtype 恢复，并与 `nn.RMSNorm` 对齐。
- 能实现 bias-free gate/up/down 三投影、SiLU 门控与逐元素相乘，数值 parity error 为 0。
- 能解释梯度分叉、token-local 计算、Attention 上下文边界和 decoder 中的 residual 数据流。

## 尚未证明

- 尚未稳定区分 `mean_square [B,S,1]`、RMSNorm `weight [H]` 与输出 `[B,S,H]`。
- 闭卷时把三个 PyTorch Linear weight shape 全部按 `[in,out]` 写反，尚未证明掌握 `weight=[out,in]`。

## Evidence

- `exercises/0005_rmsnorm_gated_ffn.py`
- `submissions/0005.md`
- `submissions/0005-feedback.md`
