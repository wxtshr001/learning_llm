# 从这里开始

当前课程状态：**第 0001、0002、0003 课已通过；正在进行第 0004 课。**

请完成 [第 0004 课：Module、Parameter、Autograd 与 Optimizer](./lessons/0004-module-autograd-optimizer.html)。

## 今天要做的事

1. 阅读第 0004 课，用具体数字区分 forward、backward 和 optimizer。
2. 运行 [0004 探索脚本](./exercises/0004_explore_training_step.py)。
3. 完成 [0004 独立作业](./exercises/0004_module_autograd_optimizer.py) 中的 TODO。
4. 查看 [训练步骤速查表](./reference/0004-training-step-cheatsheet.html)，然后关闭资料。
5. 闭卷完成 [0004 过关测试](./assessments/0004-module-autograd-optimizer.md)。
6. 把答案直接发给 Codex，或者填写 [0004 提交模板](./submissions/0004.md) 后告诉 Codex 检查。

## 我会如何判定

- **通过**：代码与关键题 2、3 正确且总分至少 80，进入第 0005 课。
- **针对性补强**：只补 Parameter 注册、梯度手算或步骤职责中的实际缺口。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

第 0004 课预计 90～120 分钟，目标是实现一次可解释、可用 finite difference 验证的最小训练步骤。

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
