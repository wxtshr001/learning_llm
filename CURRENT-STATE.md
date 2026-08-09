# Current Learning State

更新时间：2026-08-09

## 当前门禁

**第 0003 课进行中，等待学习者提交。**

学习者已明确要求继续。第 0003 课聚焦 PyTorch Tensor 的 dtype、device、数据字节数和 CPU/CUDA 数值对齐，不提前进入 autograd。

## 已完成

- 初始能力诊断。
- 第 0001 课：Linear shape、元素索引、bias broadcasting、Q projection 迁移。
- 0001R：纠正 `b[i]`/`b[j]` 输出索引误区。
- 0001R2：二维 Linear 逐元素手算复测通过。
- 0002 已证明部分：能计算 `D=H/N`、保持 reshape 元素总数、解释 Sequence/Head 轴交换，并实现 split/merge 主路径。
- 0002R 书面题 95/100：直接索引、具体数值、Q/K/V projection 输出宽度与 head layout 三个关键题全部通过。
- 第 0002 课与 0002R 门禁正式完成。

## 非阻塞工程提醒

- `split_heads()` 当前先解包 shape 再检查 rank；以后修改生产代码时应先验证再解包，但不将此记录为知识概念缺口。

## 学习者下一步

1. 学习 `lessons/0003-tensor-dtype-device.html`。
2. 运行 `exercises/0003_explore_dtype_device.py`。
3. 完成 `exercises/0003_tensor_dtype_device.py` 中的三个 TODO，并运行测试。
4. 闭卷完成 `assessments/0003-tensor-dtype-device.md`。
5. 将输出和答案写入 `submissions/0003.md` 或直接提交给当前 Agent。

## Agent 下一步

- 收到答卷前不生成第 0004 课。
- 收到后按 `COURSE-PROTOCOL.md` 检查代码、属性、dtype 精度、device 路径、容差和 LLM 迁移。
- 关键题 1、2、3 与代码均通过且总分至少 80 才进入第 0004 课。

## 最近证据

- `submissions/0002R-feedback.md`
- `learning-records/0005-head-index-and-qkv-layout-proven.md`
- `lessons/0003-tensor-dtype-device.html`

环境配置见 `ENVIRONMENT.md`。硬件信息只代表记录时主机，另一平台必须运行 `exercises/0000_verify_pytorch.py` 自行验证。
