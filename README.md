# Learning LLM Systems

这是一个可由多个 Codex Agent 跨平台接力维护的自适应学习仓库。目标是利用已有 C++、ARM Linux、嵌入式和性能工程经验，完成从 Transformer inference 到 Qwen3.5 / LiteRT-LM / Android Runtime 的能力迁移。

## Agent 接手入口

新 Agent 不要从零规划，也不要重新做初始诊断。按顺序阅读：

1. [`AGENTS.md`](./AGENTS.md) — 必须遵守的教学与同步协议。
2. [`HANDOFF.md`](./HANDOFF.md) — 交接摘要与关键路径。
3. [`CURRENT-STATE.md`](./CURRENT-STATE.md) — 当前课程和下一动作。
4. [`progress.json`](./progress.json) — 机器可读的进度状态。
5. [`MISSION.md`](./MISSION.md) 与 [`COURSE-PROTOCOL.md`](./COURSE-PROTOCOL.md) — 目标和过关规则。

## 学习者从哪里开始

打开 [`CURRENT-STATE.md`](./CURRENT-STATE.md)，只执行“学习者下一步”。完整路线在 [`LEARNING-ROUTE.md`](./LEARNING-ROUTE.md)，但它不是当前作业清单。

## 仓库结构

- `lessons/`：自包含 HTML 课程。
- `reference/`：HTML 速查表。
- `exercises/`：练习与自动测试。
- `assessments/`：闭卷过关测试。
- `submissions/`：答卷模板和 Agent 评分反馈。
- `learning-records/`：已由证据证明的能力与纠正过的误区。
- `RESOURCES.md`：筛选过的官方资料与原始论文。
- `SYNC.md`：跨平台同步协议。

课程按能力门禁推进，不按日历自动推进。代码运行成功不自动等于掌握；只有概念、shape、实现和迁移证据均满足门禁，才生成下一课。
