# 0002R 反馈

## 结论

**书面题 95/100；关键代码项未通过，暂不结束第 0002 课。**

直接索引、具体数值和 Q/K/V layout 三道关键题均已通过，说明 0002 首次提交暴露的主要概念缺口已经补上。当前只剩一个 Python 输入检查顺序问题，不生成新课程、不重做书面题。

## 逐题评分

### 题 1：25/25，关键题通过

```text
heads[b,n,s,d] = x[b,s,n*5+d]
heads[1,2,3,4] = x[1,3,2*5+4] = x[1,3,14]
```

公式和具体乘加都正确。

### 题 2：20/20，关键题通过

`y[1,3,2]` 对应 `x[1,2,3]`。连续 x 的 offset 为 `1*12+2*4+3=23`，`torch.arange(24)` 在 offset 23 的数值也是 23。对 offset 从 0 开始的理解正确。

### 题 3：35/35，关键题通过

- Q 宽度：`6*16=96`，projection 后 `[3,7,96]`，head layout `[3,6,7,16]`。
- K、V 宽度：`2*16=32`，projection 后均为 `[3,7,32]`，head layout 均为 `[3,2,7,16]`。

已证明能够把 projection 输出宽度迁移到不同的 query heads 与 KV heads。

### 题 4：15/20

- “可能在原内存地址上构建”表达了 view 共享 storage 的核心认识；更精确的答案是：本题中的 `torch.transpose` 返回 view，不复制底层 storage。
- 已理解 transpose 后逻辑读取顺序与连续内存顺序不同，现有 stride 可能无法支持目标 `view()`。
- 已理解 `contiguous()` 使当前逻辑顺序连续，`reshape()` 能共享时共享、不能时复制。

措辞仍可更精确，但不构成新的概念门禁。

## 代码关键项：未通过

当前顺序是：

```python
B, S, H = x.shape
if len(x.shape) != 3 or num_heads <= 0:
    raise ValueError(...)
```

当 rank 不是 3 时，第一行已经因解包数量不匹配抛出 Python 自带的 `ValueError`，后面的主动 rank 检查不会运行。这与要求的“先验证输入，再读取 B/S/H”不一致。

修正顺序应为：

```python
if x.ndim != 3:
    raise ValueError(...)
if num_heads <= 0:
    raise ValueError(...)
B, S, H = x.shape
```

之后再执行 `H % num_heads`，避免 `num_heads=0` 进入取模表达式。

## 唯一下一步

只修正 `split_heads()` 的检查顺序，并重新提供：

1. 原有 11 项检查通过的输出；
2. rank 2 输入由主动检查抛出 `ValueError` 的输出；
3. `num_heads=0` 由主动检查抛出 `ValueError` 的输出。

无需重答第 1～4 题。三项代码证据通过后，第 0002 课即可正式完成；按学习者要求，届时只保存进度，不生成第 0003 课。
