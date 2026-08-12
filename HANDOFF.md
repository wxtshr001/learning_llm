# Agent Handoff

## Objective

继续项目驱动、门禁式的 On-device LLM Systems 课程，最终完成 Qwen3.5-4B -> LiteRT-Torch/LiteRT-LM -> Android ARM 的正确性、性能与可复现工程成果。

## Current position

- 初始知识诊断已完成；不要重新诊断或重新制定总路线。
- 第 0001 课已在两次针对性补强后正式通过，证据见 `learning-records/0002-*`、`learning-records/0003-*` 和 `submissions/0001*-feedback.md`。
- 第 0002 课首次提交为 66/100；shape 与 split/merge 主路径已有证据，但关键题 3、7 未通过。
- 0002R 最终成绩 95/100，直接索引、具体数值和 Q/K/V layout 三个关键题全部通过，第 0002 课正式完成。
- rank 检查位置经学习者裁决为非概念性代码疏忽，不再阻塞门禁；保留为非阻塞工程提醒。
- 第 0003 课最终裁决 92/100 并正式通过。初次评分错误增加了题面未声明的 dtype 条件，已按学习者异议更正并撤销 0003R。
- 第 0004 课最终 97/100 并正式通过。
- 第 0005 课首次提交 82/100，0005R 最终 90/100；统计量/Parameter/运行 tensor 与 Linear `[out,in]` 门禁通过，第 0005 课正式完成。
- 第 0006 课首次提交 87/100：手算、shape 主链、causal 行为、scale 与 PyTorch 主计算通过，但代码 device/positive-D contract、完整两轴公式和 decoder block 路径关键项未通过。当前进入短 0006R，只复测这些缺口；不要生成第 0007 课。

精确文件和下一动作以 `CURRENT-STATE.md` 与 `progress.json` 为准。

## Important boundaries

- 学习者已有成熟 C++/ARM Linux/嵌入式能力，但 PyTorch、Tensor layout、decoder inference 和 ML graph/runtime 分层仍在建立。
- 已纠正 Linear bias 使用输出索引 `b[j]` 的误区；不要重复从头教授，只做间隔检索。
- 另一平台必须自行检测硬件，不能假设 GPU 型号。
- Qwen3.5 Gated DeltaNet 必须等 Tiny Decoder 与 KV Cache 门禁通过后再进入。

## Suggested skills

- `teach`：继续生成短小、可验证、带反馈闭环的课程。
- `handoff`：平台切换时更新交接摘要，引用现有材料而非复制整份路线。
- `github-publish-windows` 或平台对应的 GitHub 流程：安全同步课程和进度。
- `analyze`：需要对 Qwen/LiteRT 源码做证据化分析时使用。

## Resume procedure

遵守 `AGENTS.md`，先拉取 `main`。收到 0006R 后只检查代码 contract、两个求和轴公式和完整 decoder block 路径；不要重复已经通过的 0006 第 1、2、3、5 题。
