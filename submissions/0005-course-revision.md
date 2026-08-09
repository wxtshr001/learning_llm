# 0005 课程设计修订记录

## 学习者反馈

初版“Gated FFN 的四步”只列出了公式和 shape，没有先解释普通 FFN，也没有充分说明 SiLU、gate_proj、up_proj、mixed、down_proj 的具体职责、结构差异、数据流、收益与成本。

## 裁决

这是课程设计缺口，不是学习者知识缺口，不影响后续 0005 门禁评分。

## 已完成修订

- 从普通 FFN 的“升维 → 激活 → 降维”开始建立基线；
- 分别解释 gate_proj、SiLU、up_proj、逐元素 mixed 和 down_proj；
- 增加普通 FFN 结构图和 gated FFN 分支/汇合数据流图；
- 增加一个 token 的完整数值追踪；
- 对比普通 FFN 与 gated FFN 的公式、表达能力、参数量和主要计算成本；
- 明确论文结果不是对所有模型和预算的无条件保证；
- 同步更新速查表和闭卷迁移题。

## 第二轮全课审查

学习者追问“其他章节”后，确认不能只修第 4 章。其余章节也按同一标准重写：

- 第 1 章增加已证明/未证明前置能力、rank 3 与三个 axis 的数字含义；
- 第 2 章增加 RMSNorm 的作用、结构图、epsilon 与 gamma 的具体数值；
- 第 3 章增加两个 token 的 reduction、广播索引和 C++ 循环类比，并将误导性的 `variance` 改为 `mean_square`；
- 第 5 章增加 PyTorch Linear `[out,in]` 的逐元素手算；
- 第 6 章增加 Module 注册表、bias 边界和 backward 分叉图；
- 第 7 章增加 token 独立性的索引证明和双 token 思想实验；
- 第 8 章增加 parity 的完整控制变量、误差和梯度验证流程；
- 第 9 章增加最小实现与 Qwen3 源码对象的逐项映射；
- 第 10、11 章增加分阶段完成证据和指定资料的精确阅读范围；
- 探索脚本现在输出 RMSNorm 与 gated FFN 的关键中间值；独立作业新增 float16 输出 dtype 验证。
