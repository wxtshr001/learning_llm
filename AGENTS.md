# Agent Operating Contract

本仓库用于让任意 Codex Agent 无缝接手同一套自适应 LLM Systems 课程。GitHub `main` 是课程、进度和交接信息的唯一共享事实来源。

## 每次会话开始

1. 执行 `git pull --ff-only origin main`，不得在旧状态上评分或生成课程。
2. 依次阅读 `HANDOFF.md`、`CURRENT-STATE.md`、`progress.json`、`MISSION.md`、`LEARNER-PROFILE.md`、`COURSE-PROTOCOL.md`。
3. 阅读 `progress.json.current` 指向的 lesson、exercise、assessment、submission，以及最近相关的 feedback 和 learning record。
4. 不重复已在 `learning-records/` 中由证据证明的内容；用间隔检索题复习即可。
5. 不直接相信旧材料对能力的乐观判断；以答卷、代码、解释和测试输出为证据。

## 教学门禁

- 每课必须包含指定资料、讲解、练习、独立作业、闭卷测试和通过标准。
- 学习者提交后，按 `COURSE-PROTOCOL.md` 评分为：通过、针对性补强或重做。
- 未通过关键题时，不生成正常编号的下一课；生成 `NNNNR` 补强课或短测。
- 通过后才更新 `learning-records/`，并生成下一课。
- 避免长篇即时纠错；先判断断层是概念、数学、shape、Python、PyTorch 还是 Runtime。
- 评分只能使用题面明确要求和课程预先公布的门禁，不得在看到答案后增加 dtype、device、格式或实现方式等新条件。
- 单次算术或抄写错误只有在破坏关键结论、重复出现或暴露错误推理时才构成概念门禁；必须结合代码和跨题证据判断。
- 学习者对评分提出异议时，先逐字复核原题已知条件与要求，再决定维持或更正，不以评分者原意替代实际题面。

## 新课程生成门禁

创建或重写 lesson 前，必须逐项满足；任何一项不满足都不得交付：

1. 从 `LEARNER-PROFILE.md` 和最近答卷列出本课可以依赖、不能依赖的知识，不凭课程编号推测能力。
2. 首次出现的字母、缩写、符号和 API 必须立即用中文解释；tensor 必须同时解释 rank、每个 axis 的含义及 shape 中各数字的来源。
3. 抽象公式前先给一个可逐元素追踪的小数字例子；涉及 layout 时必须展示至少一个元素在变换前后的索引映射。
4. 新概念必须连接到学习者已掌握的知识。本学习者优先使用 C++ 连续内存、数组索引、指针和系统执行路径作类比，但必须说明类比的边界。
5. 区分“已完成示范”和“独立作业”：先提供可运行的探索脚本，再提供不含完整答案的 TODO 作业。
6. 闭卷题必须使用不同数字或 shape，且覆盖“含义、计算、实现、迁移”；不能只让学习者复述正文。
7. 实际运行课程脚本和测试。所有外部链接必须发起 HTTP 请求验证，不能只检查 URL 拼写；使用 `python tools/validate_course.py --external`。
8. 最后用初学者视角检查：读者不打开外部资料，也能理解作业中的全部符号和要求。官方资料是证据与延伸阅读，不是弥补正文缺失的前置教材。

如果学习者反馈“看不懂”或指出未解释符号，默认判定为课程设计未通过，而不是学习者未通过；先降低抽象层级并重写课程，再继续门禁流程。

## 每次评分或课程更新后

必须在同一个提交中完成：

1. 把反馈写入 `submissions/NNNN-feedback.md`。
2. 更新 `CURRENT-STATE.md` 与 `progress.json`。
3. 只在能力被证明确认后新增或 supersede `learning-records/`。
4. 若通过门禁，创建下一课的 lesson、exercise、assessment、submission template 和必要 reference。
5. 检查相对链接、运行适用测试，并扫描绝对路径与秘密。
6. `git pull --rebase origin main`，解决冲突后提交并推送 `main`。

## 跨平台约束

- 只使用仓库相对路径，不提交用户目录、盘符或设备 IP。
- 不提交模型权重、数据集、虚拟环境、密钥、token、`.env` 或原始大型 benchmark。
- 环境能力以本机验证为准，但环境差异不得改变共享学习结论。
- Windows PyTorch 与 LiteRT-Torch Linux 环境可以不同；在 `ENVIRONMENT.md` 记录可复现配置。

## Git 提交格式

```text
<intent: why this learning state changed>

Constraint: <what shaped the decision>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Tested: <fresh evidence>
Not-tested: <known gaps>
```

禁止 force push、重写共享历史或删除他方进度。冲突时保留双方证据，以较新的已验证提交为当前状态。
