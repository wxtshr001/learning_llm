# Cross-platform Sync Protocol

GitHub `main` 是课程、答卷、反馈、学习记录和当前状态的唯一共享事实来源。

## 开始学习或评分前

```bash
git switch main
git pull --ff-only origin main
```

阅读 `AGENTS.md`、`HANDOFF.md` 和 `CURRENT-STATE.md`。若本地有未提交修改，先提交到独立分支或暂存；不要用 reset/checkout 覆盖另一平台内容。

## 一次学习会话如何写回

按实际结果更新最小集合：

- 学习者答卷：`submissions/NNNN.md`
- Agent 反馈：`submissions/NNNN-feedback.md`
- 当前状态：`CURRENT-STATE.md`、`progress.json`
- 已证明的新能力：`learning-records/NNNN-*.md`
- 仅在门禁通过后：下一课 lesson/exercise/assessment/submission/reference

提交前：

```bash
git pull --rebase origin main
git status
git diff --check
```

然后只添加本次会话明确涉及的文件，按 `AGENTS.md` 格式提交，并执行 `git push origin main`。

## 冲突处理

- 两个平台同时更新状态时，比较提交与实际答卷证据，不按文件时间盲目覆盖。
- 若产生不同补强课，保留两套文件并用新的唯一编号；在 `CURRENT-STATE.md` 说明采用哪一条及原因。
- 禁止 force push 和删除共享历史。

## 不同步的内容

- Conda/venv 本体、模型权重、下载缓存、设备 IP、SSH 配置、token、`.env`。
- 大型原始 benchmark 放外部存储；仓库只提交脚本、摘要、图表和可追溯元数据。
