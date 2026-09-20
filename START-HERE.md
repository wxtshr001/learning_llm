# 从这里开始

当前课程状态：**第 0001～0008 课已通过；下一课是第 0009 课 Tiny Decoder Layer。**

第 0008 课验收见 [0008 反馈](./submissions/0008-feedback.md)。第 0009 课将把已经分别掌握的 RMSNorm、GQA Attention、RoPE、gated FFN 和 residual 组装成一个完整 layer。

## 今天要做的事

第 0008 课已经完成，无需重做。下一步从第 0009 课开始组装 Tiny Decoder Layer。

## 我会如何判定

- **0008 已通过**：95/100，独立代码与关键题 1、2、3 全部通过。
- **非阻塞提醒**：`position_ids` 是运行输入；历史 K_rope 与 V 才是请求 state。
- **评分约束**：不增加题面未声明条件，单次算术笔误结合完整证据判断。

第 0009 课内容尚未生成；生成时会依据 0008 的实际证据，不重复已掌握的 RoPE 计算。

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

启动当前 notebook：

```powershell
conda activate llm
jupyter lab notebooks/0008-rope-from-rotation-to-gqa.ipynb
```

kernel 选择 `Python 3 (llm)`。
