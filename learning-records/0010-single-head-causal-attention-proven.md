# 第 0006 课正式通过：单头 Causal Attention

第 0006 课首次提交 87/100，0006R 最终 95/100，全部关键项通过。

## 已证明

- 能解释 Q/K 决定读取权重、V 提供被读取内容，并区分三者角色。
- 能手算 scaled dot-product attention，写出 score 对 feature d 求和、output 对 key k 求和的逐元素关系。
- 能推导 Q/K/V、K transpose、scores、mask、weights 与 output 的 shape，并说明 softmax 沿 key axis。
- 能实现 causal mask、softmax 和 value aggregation，验证 weights 行和、未来权重与修改未来 V 不影响过去输出。
- 能解释 `sqrt(D)` scale 的数值目的，并区分 D 与 sequence length S。
- 能把 Attention 放回 pre-norm decoder block：RMSNorm → Q/K/V projections → Attention → output projection → residual → RMSNorm → FFN → residual。
- 能区分 Attention output 与词表 logits，知道 decoder layers 后还需 final RMSNorm 和 LM Head。
- 能正确处理 `D > 0` 和 mask 跟随 `query.device` 的实现 contract。

## 非阻塞提醒

- 0006R 公式中的求和上界和括号书写不够规范，但同时给出的轴说明与运行证据一致正确；以后使用 `sum_{d=0}^{D-1}` 和 `sum_{k=0}^{S-1}` 避免符号歧义。

## Evidence

- `exercises/0006_single_head_causal_attention.py`
- `exercises/0006R_attention_contract.py`
- `submissions/0006.md`
- `submissions/0006-feedback.md`
- `submissions/0006R.md`
- `submissions/0006R-feedback.md`
