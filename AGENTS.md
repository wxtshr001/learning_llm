# Agent Operating Contract

本仓库用于让任意 Codex Agent 无缝接手同一套自适应 LLM Systems 课程。GitHub `main` 是课程、进度和交接信息的唯一共享事实来源。

## 每次会话开始

1. 执行 `git pull --ff-only origin main`，不得在旧状态上评分或生成课程。
2. 依次阅读 `HANDOFF.md`、`CURRENT-STATE.md`、`progress.json`、`MISSION.md`、`COURSE-PROTOCOL.md`。
3. 阅读 `progress.json.current` 指向的 lesson、exercise、assessment、submission，以及最近相关的 feedback 和 learning record。
4. 不重复已在 `learning-records/` 中由证据证明的内容；用间隔检索题复习即可。
5. 不直接相信旧材料对能力的乐观判断；以答卷、代码、解释和测试输出为证据。

## 教学门禁

- 每课必须包含指定资料、讲解、练习、独立作业、闭卷测试和通过标准。
- 学习者提交后，按 `COURSE-PROTOCOL.md` 评分为：通过、针对性补强或重做。
- 未通过关键题时，不生成正常编号的下一课；生成 `NNNNR` 补强课或短测。
- 通过后才更新 `learning-records/`，并生成下一课。
- 避免长篇即时纠错；先判断断层是概念、数学、shape、Python、PyTorch 还是 Runtime。

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
