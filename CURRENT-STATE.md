# Current Learning State

更新时间：2026-08-08

## 当前门禁

**第 0002 课进行中，等待学习者提交。**

主题：reshape、transpose、head layout 与 tensor view/contiguous 边界。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。

## 学习者下一步

1. 学习 `lessons/0002-reshape-transpose-head-layout.html`。
2. 完成 `exercises/0002_head_layout.py` 中的 TODO，并运行全部测试。
3. 闭卷完成 `assessments/0002-reshape-transpose-head-layout.md`。
4. 把代码输出和 7 道答案写入 `submissions/0002.md` 或直接提交给当前 Agent。

## Agent 下一步

- 收到答卷前，不生成第 0003 课。
- 收到后按 `COURSE-PROTOCOL.md` 评分，写入 `submissions/0002-feedback.md`。
- 关键题 1、2、3、7 与代码均通过且总分至少 80% 才能进入 PyTorch Tensor/dtype/device。
- 若只在 view/reshape API 边界出错，插入短补强；若轴语义或元素映射错误，降低维度重新练习。

## 最近证据

- `learning-records/0003-linear-shape-and-calculation-passed.md`
- `submissions/0001R2-feedback.md`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
