# 0006 课程设计反馈与修订裁决

## 结论

本次不是学习者答题错误，也不评分。0006 初版没有满足仓库已经声明的“从零解释”与 Transformer 系统上下文要求，判定为课程设计缺陷；第 0006 课保持 `in_progress`，门禁成绩仍为空。

## 学习者指出的问题

1. 正文没有真正从零解释 Q/K/V 角色、score、scale、causal mask、softmax axis 和 value 加权。
2. 课程属于 Transformer inference 模块，却没有说明 Transformer 整体、各组件作用、结构嵌入位置与端到端数据流。
3. 仅按课程计划罗列概念无法覆盖学习者尚不知道要问什么的隐含知识，不能服务最终的 Qwen/LiteRT/Android 推理目标。

## 修订内容

- 从 token ids、embedding、decoder layers、final norm、LM Head/logits 建立整体 Causal LM 数据流。
- 解释 pre-norm decoder layer、两条 residual、Attention 与 FFN 的分工，以及 Attention 输出如何回到 `[B,S,H]` 主路径。
- 从 hidden states 的三组 Linear projection 解释 Q/K/V 的来源、角色与类比边界。
- 使用 D=2 和 D=1 的逐元素数字，完整推导 score、`sqrt(D)` scale、causal mask、softmax key axis 与 V 加权。
- 增加 Parameter、activation、KV Cache 的生命周期区别，以及 full-sequence、prefill、decode 三种执行视角。
- 区分 causal mask 与 padding mask，说明 Attention 是动态信息路由而不是笼统的“模型记忆”。
- 引导脚本现在显式输出 raw scores、future mask、masked scores、weights、行和与 output。
- 速查表和闭卷题同步加入 Transformer 结构位置与完整数据流检查。
- `AGENTS.md`、`COURSE-PROTOCOL.md` 与 `CURRICULUM.md` 加入从最终目标反向审查隐含前置知识的规则。

## 门禁影响

- 不记录任何学习者知识失败。
- 不生成 0007。
- 学习者从修订版 0006 重新开始；提交前已明确把 Q/K/V 角色与 Transformer 数据流纳入关键题，总分门禁仍为 80。
