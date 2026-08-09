# dtype/device 实现已证明，执行路径表达与误差计算待补强

## 已证明

- 能用 `numel()*element_size()` 计算 Tensor 元素区域字节数。
- 能把 X、W、b 转换到相同 device/dtype，并在 `cuda:0` 上完成 Linear。
- 能计算最大绝对误差，并主动检查 reference/candidate 的 shape、dtype、device。
- 理解 float32 转 float16 会丢失信息，转回 float32不能恢复。
- 能计算 activation 在 float32/float16 下的字节数。

## 尚未证明

- 书面描述 device 路径时只迁移了 X，没有完整写出 X/W/b 和执行 dtype 的统一。
- candidate 回 CPU 时没有写明还需统一为 float32 后再与 reference 比较。
- 一个绝对误差出现十进制位错误：0.0010 写成 0.0001。
- 对 token id 使用整数 dtype 的解释尚未明确到“离散、精确的索引”。

## Evidence

- `exercises/0003_tensor_dtype_device.py`
- `submissions/0003.md`
- `submissions/0003-feedback.md`
