# 从这里开始

当前课程状态：**第 0001、0002 课已通过；第 0003 课首次提交 75/100，正在进行 0003R 局部补强。**

请完成 [第 0003R 课：完整 device 路径与误差](./lessons/0003R-device-path-and-error.html)。

## 今天要做的事

1. 阅读 0003R，只看完整三输入迁移、比较格式和误差小数位。
2. 运行 [0003R 引导脚本](./exercises/0003R_trace_device_path.py)。
3. 关闭资料，完成 [0003R 闭卷复测](./assessments/0003R-device-path-and-error.md)。
4. 把答案直接发给 Codex，或者填写 [0003R 提交模板](./submissions/0003R.md) 后告诉 Codex 检查。

## 我会如何判定

- **通过**：0003R 关键题 1、2 正确且总分至少 80，第 0003 课正式完成。
- **继续局部补测**：仍只有一个小项错误，只补该项。
- **不重做代码**：0003 主作业已经在 CUDA 上全部通过。

0003R 预计 20～30 分钟，目标是能完整写出 X/W/b/Y/candidate 的 device/dtype 路径，并稳定计算小数误差。

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
