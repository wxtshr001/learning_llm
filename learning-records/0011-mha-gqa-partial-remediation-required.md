# MHA/GQA 主计算与 Cache 成本已证明，仅 Tensor 阶段待补强

第 0007 课异议复核后 89/100，总分达标；关键题 2 未通过，因此尚未完成第 0007 课。原关键题 6.4 因教学覆盖不足撤销。

## 已证明

- 能计算 `Nq/Nkv` group size，并写出连续 Q head → KV head 映射以及 MHA/GQA/MQA 边界。
- 能手算多个 GQA heads 的 causal weights、V 聚合和 concat/merge 结果。
- 能实现 K/V 分组展开、causal attention、head merge 和 KV Cache 字节计算；课程脚本在 CUDA 上复跑通过。
- 能正确计算 MHA/GQA Cache bytes 和 MiB，并解释 Cache 保存 repeat 前的 `Nkv` heads。
- 能解释 GQA 在 decode 中降低历史 K/V 读取量，以及为什么 head 比例不能直接推出端到端加速比。
- 能说明 full-sequence/prefill attention weights 的 head axis 由 `Nq` 决定。
- 能回答题面中已经充分教授的 Transformer inference 迁移问题；完整 GQA+RoPE+Cache 路径尚未教学，不作为能力缺口。

## 尚未证明

- 尚未稳定区分 q/k/v raw projection、split、逻辑 K/V 映射、merge 与 `o_proj` 各阶段的 rank/axis。
- 尚未证明理解“显式 head axis”与“扁平宽度包含 `Nq*D`”的区别。
- 漏答 concat/merge 不能替换为 head average 的原因。

## 非阻塞提醒

- 首次提交把 `GQA/MHA` 容量比例 25% 写成节省比例 75%；主公式和绝对容量正确。
- `repeat_kv_for_query_heads()` 的非法 rank/零 K/V head 检查顺序可改进，但不作为概念门禁。

## 评分更正

- 0007 只用一句话预告 RoPE/Cache 位置，不足以支撑 Q6.4 关键门禁；该题撤销，0007R 不复测完整系统路径。

## Evidence

- `exercises/0007_mha_gqa.py`
- `submissions/0007.md`
- `submissions/0007-feedback.md`
