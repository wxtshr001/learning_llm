# 从这里开始

当前课程状态：**第 0001 课已通过；正在进行第 0002 课。**

请完成 [第 0002 课：reshape、transpose 与 Head Layout](./lessons/0002-reshape-transpose-head-layout.html)。

## 今天要做的事

1. 打开第 0002 课，按页面要求阅读两个官方资料片段。
2. 编辑 [第 0002 课练习代码](./exercises/0002_head_layout.py) 中的两个 `TODO`。
3. 运行练习，直到全部测试通过。
4. 闭卷完成 [第 0002 课过关测试](./assessments/0002-reshape-transpose-head-layout.md)。
5. 把答案直接发给 Codex，或者填写 [第 0002 课提交模板](./submissions/0002.md) 后告诉 Codex 检查。

## 我会如何判定

- **通过**：关键题全部正确、代码测试全通过，进入第 0003 课。
- **针对性补强**：总体理解正确，但 view/reshape API 边界不稳；只补对应部分，再做短测。
- **重做本课**：轴语义或元素映射存在根本误解；换更低维的 tensor 重新练习。

第 0002 课预计 60～90 分钟，目标是能独立把 `[B,S,H]` 转换为 Attention 使用的 `[B,N,S,D]`，并解释元素映射和 contiguous 边界。

## 其他文件是什么

- [课程地图](./CURRICULUM.md)：课程顺序、门禁和可能的分支。
- [完整路线](./LEARNING-ROUTE.md)：24 周项目全景，不是每天从头照读的教材。
- [资料清单](./RESOURCES.md)：经过筛选的官方资料库。
- [学习使命](./MISSION.md)：所有课程为什么要学、哪些内容不学。
- `learning-records/`：记录已经真正证明掌握的能力。
- `reference/`：以后反复查阅的速查表。

## 环境说明

已建立独立 Conda 环境 `llm`：Python 3.11.15、PyTorch 2.12.1、CUDA 12.6 和 NumPy 2.4.6。第 1 课仍从纯 Python 实现开始，后续课程再使用 PyTorch 对齐。

验证环境：

```powershell
conda activate llm
python exercises/0000_verify_pytorch.py
```
