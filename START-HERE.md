# 从这里开始

当前课程状态：**第 0001、0002 课已通过；正在进行第 0003 课。**

请完成 [第 0003 课：PyTorch Tensor、dtype 与 device](./lessons/0003-tensor-dtype-device.html)。

## 今天要做的事

1. 打开第 0003 课，先理解 Tensor 的 shape、dtype、device 和字节数。
2. 运行 [0003 探索脚本](./exercises/0003_explore_dtype_device.py)，逐段解释输出。
3. 编辑 [第 0003 课独立作业](./exercises/0003_tensor_dtype_device.py) 中的三个 `TODO`，直到全部测试通过。
4. 查看 [dtype/device 速查表](./reference/0003-dtype-device-cheatsheet.html)，然后关闭资料。
5. 闭卷完成 [第 0003 课过关测试](./assessments/0003-tensor-dtype-device.md)。
6. 把答案直接发给 Codex，或者填写 [第 0003 课提交模板](./submissions/0003.md) 后告诉 Codex 检查。

## 我会如何判定

- **通过**：关键题全部正确、代码测试全通过，进入第 0004 课。
- **针对性补强**：总体理解正确，但 dtype、device、字节数或容差中只有一个局部缺口；只补对应部分。
- **重做本课**：无法区分 shape、dtype、device，或不能建立一致的设备执行路径；降低抽象程度重新练习。

第 0003 课预计 60～90 分钟，目标是能在 CPU/CUDA 和 float32/float16 之间显式转换 Tensor，计算内存字节数，并使用容差判断结果是否对齐。

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
