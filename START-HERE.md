# 从这里开始

当前课程状态：**第 0001～0007 课已通过；正在进行第 0008 课 RoPE。**

请完成 [第 0008 课：RoPE——从二维旋转到 GQA](./lessons/0008-rope-from-rotation-to-gqa.html)。

## 今天要做的事

1. 阅读第 0008 课，从二维旋转和 Qwen half-split 开始理解 RoPE。
2. 运行 [0008 探索脚本](./exercises/0008_explore_rope.py)。
3. 完成 [0008 独立作业](./exercises/0008_rope.py) 中的 TODO。
4. 查看 [0008 RoPE 速查表](./reference/0008-rope-cheatsheet.html)，然后关闭资料。
5. 完成 [0008 闭卷测试](./assessments/0008-rope.md)，填写 [0008 提交模板](./submissions/0008.md)。

## 我会如何判定

- **通过**：独立代码与关键题 1、2、3 正确且总分至少 80，进入 Tiny Decoder Layer。
- **针对性补强**：只补 half-split、位置旋转、相对 score 或 GQA 广播中的实际缺口。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

第 0008 课预计 100～130 分钟；不考长上下文 RoPE scaling，也不要求实现完整 KV Cache。

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
