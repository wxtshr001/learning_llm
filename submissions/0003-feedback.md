# 0003 反馈

## 最终结论

**92/100，通过。第 0003 课正式完成。**

初次评分曾判为 75/100 并要求 0003R，学习者提出反对后重新核对题面，确认原判给关键题 3 增加了未声明的 dtype 条件，并忽略了 W、b 已在 `cuda:0` 的已知条件。该判定无效，0003R 已撤销。

独立作业已由 Agent 在本机 `cuda:0` 复跑，全部检查通过。结合代码、书面答案和跨题证据，学习者已经证明掌握第 0003 课要求的 Tensor dtype、device、字节数、显式迁移和容差判断。

## 最终评分

| 题目 | 得分 | 说明 |
|---|---:|---|
| 1. 属性与自动推断 | 16/20 | rank、shape、dtype、numel、element size 正确；本题漏写总字节数和 axis 语义，但第 5 题已独立证明字节计算能力。 |
| 2. dtype 转换与精度 | 20/20 | shape 不变、向下转换会舍入、转回 float32 不能恢复信息，全部正确。 |
| 3. device 执行路径 | 25/25 | 题设已给 W、b 在 `cuda:0`，因此只需把 X 迁移到 CUDA；candidate 回 CPU 后即可与 CPU reference 建立共同 device。题面未要求指定 dtype，不得据此扣分。 |
| 4. 容差与误差 | 17/20 | 第三个绝对误差把 0.0010 写成 0.0001，但最大误差 0.002 和门限不通过结论均正确，判定为单次算术笔误而非概念缺口。 |
| 5. 迁移到 LLM | 14/15 | 480/240 bytes 和 shape 不变正确；已表达 token id 不应使用近似小数格式，离散索引措辞可在后续自然强化。 |

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

## 学习者裁决带来的评分修正

原题明确给出：

```text
X.device = cpu
W.device = cuda:0
b.device = cuda:0
```

问题只问选择 CUDA 时怎样处理 X、W、b。答案将 X 移至 `cuda:0` 已满足条件，W、b 无需重复迁移。题目没有给目标 dtype，也没有要求 candidate 转成 float32。因此原反馈要求三者统一为 float16、candidate 统一为 float32，属于评分者添加题外验收条件。

单次小数位错误也不能覆盖已有证据：学习者正确算出最大误差 0.002，并正确判断超过 0.001 门限。这证明“计算逐元素误差—取最大值—与门限比较”的概念链已经建立。

## 下一步

第 0003 课通过，进入第 0004 课：用一个只有 scale 和 bias 的最小 `nn.Module`，区分 forward、backward 与 optimizer 的职责，并用 finite difference 验证梯度。
