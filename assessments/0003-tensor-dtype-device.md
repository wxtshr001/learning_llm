# 0003 闭卷测试：Tensor dtype 与 device

完成探索脚本和独立作业后，关闭课程、源码和速查表作答。不要只粘贴程序结果；需要解释为什么。

## 符号表

- `B`：Batch size，一次处理的样本数。
- `Hin`：Input hidden size，每个输入样本的特征数。
- `Hout`：Output hidden size，每个输出样本的特征数。
- `X`：Linear 输入，shape 为 `[B,Hin]`。
- `W`：Linear 权重，shape 为 `[Hin,Hout]`。
- `b`：Linear bias，shape 为 `[Hout]`。
- `reference`：参考结果，本测试默认是 CPU/float32 结果。
- `candidate`：待比较结果。

## 提交证据

1. 粘贴 `0003_explore_dtype_device.py` 的完整输出。
2. 粘贴完成 TODO 后 `0003_tensor_dtype_device.py` 的完整输出。
3. 写出实际检测到的 target device；如果是 CPU，说明 CUDA 分支为何被跳过。

## 1. 属性与自动推断（关键题，20 分）

```python
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float16)
```

分别写出 `a` 和 `b` 的：

- rank；
- 数字 shape，并解释 axis 0、axis 1 的长度；
- dtype；
- `numel()`；
- `element_size()`；
- Tensor 元素区域的字节数。

## 2. dtype 转换与精度（关键题，20 分）

回答：

1. 把 float32 转成 float16 后再转回 float32，shape 是否变化？
2. 为什么数值不一定恢复为最初的 float32 数值？
3. 最后转回 float32 是否能恢复 float16 阶段已经丢失的信息？为什么？

## 3. device 执行路径（关键题，25 分）

已知：

```text
X.device = cpu
W.device = cuda:0
b.device = cuda:0
```

需要计算 `Y = X @ W + b`：

1. 直接执行通常会发生什么？
2. 如果选择在 CUDA 上计算，应该怎样处理 X、W、b？
3. 为了与 CPU reference 比较，candidate 最后应怎样处理？
4. 为什么不应在模型的每一个算子前后都往返 CPU 与 GPU？

## 4. 容差与误差（20 分）

```text
reference = [1.0000, 2.0000, -3.0000]
candidate = [1.0005, 1.9980, -2.9990]
```

1. 三个元素的绝对误差分别是多少？
2. 最大绝对误差是多少？
3. 如果验收条件是最大绝对误差不超过 `0.001`，是否通过？
4. 为什么 CPU/GPU 或 float32/float16 结果比较不应该默认使用逐 bit 相等？

## 5. 迁移到 LLM（15 分）

一个 activation Tensor 的 shape 是 `[B=2,S=5,H=12]`：

1. float32 元素区域占多少字节？float16 占多少字节？写出计算过程。
2. 仅把 dtype 从 float32 改为 float16，shape 会不会变？
3. token id 为什么通常使用整数 dtype，而不是把它当作 float16 activation？

## 通过标准

- 总分至少 80 分。
- 独立作业测试与第 1、2、3 题全部通过。
- 第 4 题必须正确区分“非零误差”和“超过容差”。
- 未通过时只补 dtype、device、内存或误差中的实际缺口，不重做已证明的 head layout。
