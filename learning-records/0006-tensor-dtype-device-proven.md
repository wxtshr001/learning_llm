# 第 0003 课正式通过：Tensor dtype、device 与数值对齐

第 0003 课最终成绩 92/100。独立作业在 `cuda:0` 上全部通过；学习者对初次评分提出有效异议，确认评分不得增加题面未声明的 dtype 条件，单次算术笔误不得在概念证据充分时自动触发补强。

## 已证明

- 能区分 Tensor 的 shape、dtype、device、numel 和 element size。
- 能用 `numel()*element_size()` 计算 float32/float16 元素区域字节数。
- 能解释 float32 转 float16 的舍入损失，以及转回 float32不能恢复信息。
- 能把参与 Linear 的 Tensor 显式迁移到目标 device/dtype，并在 CUDA 上运行。
- 能把 candidate 移回 CPU 与 reference 比较，计算最大绝对误差并依据门限判断。
- 理解频繁 CPU/GPU 往返会产生数据搬运与同步成本。

## 非阻塞提醒

- 一次逐元素误差把 0.0010 写成 0.0001，但最大误差与门限结论正确；记录为算术注意力问题，不作为 dtype/device 概念缺口。
- token id 的整数索引语义可在 embedding 课程中自然强化，无需为此插入补强课。

## Evidence

- `exercises/0003_tensor_dtype_device.py`
- `submissions/0003.md`
- `submissions/0003-feedback.md`
