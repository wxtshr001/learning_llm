# 0009 题面设计反馈（尚未批改）

2026-09-25，学习者指出闭卷测试的问题描述缺少前提；第 4.1 题“列 Parameter”没有指明是哪一个模型层、它包含哪些模块，也未说明输入和无 bias 配置。该问题属于评测设计缺陷，不是学习者知识缺口。

已修订 `assessments/0009-tiny-decoder-layer.md`：

- 卷首限定为一个 Tiny Decoder Layer，明确输入、模块组成、无 bias、`H=Nq*D`、full-sequence 和无 Cache 的范围。
- 第 1 题区分相同数值宽度与 `o_proj` 的学习组合，并明确两次 residual 的操作数。
- 第 2 题补齐 raw、split、merge 和 FFN 各阶段的对象定义。
- 第 3 题固定 `position_ids`、参数及未改动的 token，限定比较范围。
- 第 4 题明确只列该 layer 的 Parameter 名称，解释 residual 引用，并用具体 prompt 长度限定 Cache 假设。

题目总分和关键项（独立代码、第 1、2、3 题）不变；`progress.json` 仍为等待 0009 提交，不对学习者评分。
