# 从这里开始

当前课程状态：**第 0001～0004 课已通过；正在进行第 0005 课。**

请完成 [第 0005 课：RMSNorm 与 Gated FFN](./lessons/0005-rmsnorm-gated-ffn.html)。

## 今天要做的事

1. 阅读第 0005 课，先用具体数字理解 RMS、last axis 和三条 FFN 投影。
2. 运行 [0005 探索脚本](./exercises/0005_explore_rmsnorm_gated_ffn.py)。
3. 完成 [0005 独立作业](./exercises/0005_rmsnorm_gated_ffn.py) 中的 TODO。
4. 查看 [RMSNorm 与 Gated FFN 速查表](./reference/0005-rmsnorm-gated-ffn-cheatsheet.html)，然后关闭资料。
5. 闭卷完成 [0005 过关测试](./assessments/0005-rmsnorm-gated-ffn.md)。
6. 把答案直接发给 Codex，或者填写 [0005 提交模板](./submissions/0005.md) 后告诉 Codex 检查。

## 我会如何判定

- **通过**：独立作业与关键题 1、3 正确且总分至少 80，进入第 0006 课。
- **针对性补强**：只补 RMS 计算、last-axis 广播或 gated FFN shape 中的实际缺口。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

第 0005 课预计 100～130 分钟，目标是实现 RMSNorm 与 gated FFN，并与 PyTorch/reference 公式完成数值对齐。

## 其他文件是什么

- [课程地图](./CURRICULUM.md)：课程顺序、门禁和可能的分支。
- [完整路线](./LEARNING-ROUTE.md)：24 周项目全景，不是每天从头照读的教材。
- [资料清单](./RESOURCES.md)：经过筛选的官方资料库。
- [学习使命](./MISSION.md)：所有课程为什么要学、哪些内容不学。
- `learning-records/`：记录已经真正证明掌握的能力。
- `reference/`：以后反复查阅的速查表。

## 环境说明

已建立独立 Conda 环境 `llm`：Python 3.11.15、PyTorch 2.12.1、CUDA 12.6 和 NumPy 2.4.6。

```powershell
conda activate llm
python exercises/0000_verify_pytorch.py
```
