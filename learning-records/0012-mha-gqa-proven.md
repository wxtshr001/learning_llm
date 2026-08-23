# 第 0007 课正式通过：MHA、GQA 与 KV Cache 成本

0007R 最终 98/100，代码与关键 shape/axis 题通过，第 0007 课正式完成。

## 已证明

- 能区分 MHA、GQA、MQA，并计算 `group_size=Nq/Nkv` 和连续 Q head → KV head 映射。
- 能区分 q/k/v raw projection、split、logical K/V、scores/weights、head output、merge 与 `o_proj` 的 rank 和数字 shape。
- 能判断哪些 tensor 有显式 `Nq` head axis，以及 merge 后 `Nq*D` 只是扁平特征宽度。
- 能实现 GQA causal attention、K/V 逻辑映射、head merge 和完整 shape contract。
- 能解释 concat/merge 保留各 head 特征，而 average 会不可逆丢失信息并破坏 `o_proj` 输入宽度。
- 能说明 Cache 持久保存未展开的 `[B,Nkv,S,D]` K/V，计算 MHA/GQA Cache 字节与 decode 带宽收益。
- 能说明 full-sequence/prefill 仍有 `Nq` 份 attention weights，且不能由 `Nkv` 比例直接推出端到端加速比。

## 非阻塞提醒

- 0007R 把 `(1,6,12)` raw K/V 称作“2D 矩阵”，但完整 rank-3 shape 与 head-axis 结论正确，仅记术语笔误。
- `repeat_kv_for_query_heads()` 的非法 rank/零 K/V head 检查顺序以后可改进，不影响本课概念门禁。

## 评分边界记录

- 原 0007 Q6.4 因完整 GQA+RoPE+Cache 路径未被充分讲授而撤销；本记录不据此声称已经掌握 RoPE。

## Evidence

- `exercises/0007_mha_gqa.py`
- `submissions/0007.md`
- `submissions/0007-feedback.md`
- `exercises/0007R_gqa_shape_contract.py`
- `submissions/0007R.md`
- `submissions/0007R-feedback.md`
