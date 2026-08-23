# 从这里开始

当前课程状态：**第 0001～0006 课已通过；第 0007 课异议复核后 89/100，正在进行只补 shape/axis 的 0007R。**

请完成 [0007R：GQA Tensor 阶段与显式 Head Axis](./lessons/0007R-gqa-tensor-stages-and-head-axis.html)。

## 今天要做的事

1. 阅读 0007R，逐阶段区分 raw、split、逻辑 K/V、head output、merge 与 `o_proj`。
2. 运行 [0007R 探索脚本](./exercises/0007R_explore_gqa_stages.py)。
3. 完成 [0007R 独立作业](./exercises/0007R_gqa_shape_contract.py) 中的 TODO。
4. 关闭资料后完成 [0007R 闭卷复测](./assessments/0007R-gqa-tensor-stages-and-head-axis.md)。
5. 填写 [0007R 提交模板](./submissions/0007R.md)，然后告诉 Codex 检查。本次不考完整系统路径或 RoPE 位置。

## 我会如何判定

- **通过**：独立作业与关键题 1 正确且总分至少 80，正式完成第 0007 课并进入第 0008 课。
- **针对性补强**：只继续补实际仍未证明的 tensor 阶段，不重复已通过的 GQA 主计算，也不加入未充分讲授的系统路径。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

0007R 预计 25～35 分钟；完整 decoder/GQA 系统路径和 RoPE 均不在本次复测范围。

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
