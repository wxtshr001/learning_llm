# 0003 反馈

## 结论

**75/100，未通过；进入 0003R 针对性补强。**

独立作业已由 Agent 在本机 `cuda:0` 复跑，全部检查通过。代码证明你能够使用 `.to(device=..., dtype=...)` 迁移 X、W、b，计算字节数并求最大绝对误差。书面题的主要缺口是：device 执行路径只写了迁移 X，没有写 W、b 和统一 dtype；candidate 回 CPU 时也没有写回 float32。第 4 题第三个误差小数位算错。由于关键题 3 未通过，暂不进入第 0004 课。

## 分项评分

| 题目 | 得分 | 说明 |
|---|---:|---|
| 1. 属性与自动推断 | 15/20 | rank、shape、dtype、numel、element size 正确；没有明确解释两个 axis 的含义，也漏写了 a=48 bytes、b=24 bytes。 |
| 2. dtype 转换与精度 | 20/20 | shape 不变、向下转换会舍入、转回 float32 不能恢复信息，全部正确。 |
| 3. device 执行路径 | 14/25 | 知道混合 device 会报错，也知道搬运有成本；但只迁移了 X，缺少 W、b 和 dtype，candidate 也只写了回 CPU。关键题未通过。 |
| 4. 容差与误差 | 14/20 | 前两个误差、最大误差和不通过结论正确；第三个误差应为 0.0010，不是 0.0001；逐 bit 不等的解释还需落到舍入、运算顺序与设备实现。 |
| 5. 迁移到 LLM | 12/15 | 480/240 bytes 和 shape 不变正确；token id 的核心原因是它表示离散、必须精确的数组/词表索引，而不是 float16 范围是否足够。 |

## 代码证据

以下实现均正确：

- `tensor.numel() * tensor.element_size()`；
- X、W、b 分别转换到目标 device/dtype；
- 检查 reference/candidate 的 shape、dtype、device；
- `(reference - candidate).abs().max().item()`。

Agent 复跑输出：

```text
Lesson 0003: all checks passed.
target device: cuda:0
float32 data bytes: 24
float16 data bytes: 12
float32 max absolute error: 0.0
float16 max absolute error: 0.0003905296325683594
```

## 逐题纠正

### 题 1

两者 rank 都是 2，axis 0 长度 2 表示有两行/两个样本，axis 1 长度 3 表示每行三个元素/特征。

```text
a: int64, 6 * 8 = 48 bytes
b: float16, 6 * 2 = 24 bytes
```

### 题 3

若选择 CUDA/float16 计算，完整路径必须是：

```python
x_compute = X.to(device="cuda:0", dtype=torch.float16)
w_compute = W.to(device="cuda:0", dtype=torch.float16)
b_compute = b.to(device="cuda:0", dtype=torch.float16)
y_compute = x_compute @ w_compute + b_compute
candidate = y_compute.to(device="cpu", dtype=torch.float32)
```

三个参与运算的 Tensor 必须一起进入目标执行空间。最后回 CPU/float32 是为了和 CPU/float32 reference 使用一致的比较格式，不会恢复 float16 计算已经丢失的精度。

### 题 4

```text
abs(-3.0000 - -2.9990) = abs(-0.0010) = 0.0010
```

三个误差是 `[0.0005, 0.0020, 0.0010]`，最大值是 0.0020，所以阈值 0.001 不通过。

### 题 5

token id 是词表中的离散下标，必须精确表示整数并用于索引；float16 是近似浮点格式，即使数值范围覆盖某个 id，也不等于适合做索引。

## 下一步

完成 `lessons/0003R-device-path-and-error.html`，运行 `exercises/0003R_trace_device_path.py`，再闭卷提交 `assessments/0003R-device-path-and-error.md`。代码主作业不需要重做。
