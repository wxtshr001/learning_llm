# 0002R 闭卷复测：直接索引与 Q/K/V Layout

完成引导脚本和代码边界修正后，关闭课程、源码和速查表作答。

## 提交证据

1. 粘贴 `0002R_trace_qkv_layout.py` 的完整输出。
2. 粘贴修正后的 `0002_head_layout.py` 完整输出。
3. 说明错误 rank 和 `num_heads=0` 各会抛出什么异常。

## 1. 直接索引映射（关键题，25 分）

```text
x.shape = [B=2,S=4,H=15]
N=3
D=5
heads = x.reshape(B,S,N,D).transpose(1,2)
```

- 补全：`heads[b,n,s,d] = x[ ?, ?, ? ]`。
- `heads[1,2,3,4]` 对应 x 的哪个完整索引？写出乘加过程。

## 2. transpose 后的具体数值（关键题，20 分）

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
```

- `y[1,3,2]` 对应 x 的哪个索引？
- 该元素的 storage offset 和数值分别是多少？
- 为什么这里不需要把 offset 减 1？

## 3. Q/K/V 迁移（关键题，35 分）

```text
B=3
S=7
query_heads=6
kv_heads=2
D=16
```

- Q projection 的输出宽度是多少？projection 后 `[B,S,Hq]` 的数字 shape 是什么？
- K、V projection 的输出宽度是多少？projection 后各自的数字 shape 是什么？
- reshape 与 transpose 后，Q、K、V 各自的 `[B,N,S,D]` 数字 shape 是什么？

## 4. storage 与 API（20 分）

用自己的话回答：

- 本课的 `transpose` 是否复制 storage？
- 为什么 transpose 后直接 `view()` 可能失败？
- `contiguous()` 与 `reshape()` 分别提供什么保证？

## 通过标准

- 总分至少 80 分。
- 代码边界检查与第 1、2、3 题全部通过。
- 未通过时只继续补对应缺口，不重复第 0001 课或 0002 已证明的 shape 基础。
