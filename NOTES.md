# Teaching Notes

- 学习语言：中文；代码、API 和张量名保留英文。
- 教学方式：先通过具体问题诊断，再给针对性内容；避免“你会不会”的自评式提问。
- 学习方式：项目驱动，知识 -> 最小实验 -> reference parity -> 验收。
- 不把阅读、观看或完成课程本身当成验收标准。
- 每周至少 10 小时；RTX 3060 12GB 和 Android 设备均可使用。
- 已有优势：多年 C++、ARM Linux、嵌入式、并发、资源管理、板端部署、性能分析和故障定位。
- 当前断层：Tensor layout、PyTorch、标准 decoder 实现、KV Cache layout、export/lowering/runtime 分层。
- 课程采用自适应门禁：每节课必须完成资料阅读、可运行作业和闭卷测试；根据提交结果决定通过、补强或降阶，不按周数自动推进。
- 已建立专用 Conda 环境 `llm`：Python 3.11.15、PyTorch 2.12.1+cu126、NumPy 2.4.6。
- PyTorch 实测识别 GPU 为 NVIDIA GeForce RTX 2060 SUPER；此前材料写 RTX 3060，以实际运行结果为当前环境依据。
- 第 0001 课首次门禁：代码实现与迁移题较好，但闭卷时把 Linear bias 写成 `b[i]`，且 2×2 矩阵手算错误。当前安排 0001R，只补输出索引和逐元素手算，不重复 Python 实现。
- 0001R 复测：`b[j]`、完整索引公式和 Q projection 迁移已纠正；仅 2×2 matmul+bias 算术仍错。进入 0001R2 三题手算，不重复概念课。
- 0001R2 三题逐元素手算全部正确。第 0001 课正式通过，进入第 0002 课：reshape、transpose、head layout 与 contiguous/view 边界。
