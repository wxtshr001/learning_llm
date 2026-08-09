# 0003R 闭卷复测：完整执行路径与误差

运行引导脚本后，关闭课程和源码作答。

## 符号表

- `X`：Linear 输入。
- `W`：Linear weight，权重。
- `b`：Linear bias，偏置。
- `Y`：Linear 输出。
- `reference`：CPU/float32 参考结果。
- `candidate`：转换成与 reference 相同比较格式后的待检查结果。

## 提交证据

粘贴 `0003R_trace_device_path.py` 的完整输出。

## 1. 完整 device/dtype 路径（关键题，50 分）

原始 X、W、b 都在 CPU/float32。要求在 `cuda:0`/float16 上计算 `Y = X @ W + b`，再与 CPU/float32 reference 比较。

逐行写出：

1. X、W、b 分别怎样得到 compute Tensor；每个 compute Tensor 的 device/dtype 是什么？
2. `y_compute` 的 device/dtype 是什么？
3. candidate 怎样从 `y_compute` 得到；它最终的 device/dtype 是什么？
4. 为什么 candidate 转回 float32不能恢复 float16 计算丢失的精度？

## 2. 误差与门限（关键题，30 分）

```text
reference = [0.5000, -1.2500, 4.0000]
candidate = [0.4992, -1.2515, 4.0009]
```

1. 写出三个逐元素绝对误差，保留四位小数。
2. 最大绝对误差是多少？
3. 门限是“不超过 0.0010”，是否通过？

## 3. 属性与索引语义（20 分）

```python
ids = torch.tensor([[7, 12, 3], [9, 4, 1]])
```

1. 写出 rank、shape、两个 axis 的长度含义、dtype、numel、element size 和元素区域字节数。
2. 为什么 token id 应使用整数 dtype？答案必须包含“离散”“精确”“索引”三个含义。

## 通过标准

- 总分至少 80 分。
- 第 1、2 题均通过。
- 通过后第 0003 课正式完成；否则只继续补仍错的小项。
