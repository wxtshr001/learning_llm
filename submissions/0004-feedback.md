# 0004 反馈

## 结论

**97/100，通过。第 0004 课正式完成。**

独立作业由 Agent 在本机复跑，全部检查通过。Parameter 注册、forward/backward/optimizer 职责、手算梯度、SGD 更新、梯度累积、finite difference 和 inference mode 均达到本课门禁。

## 分项评分

| 题目 | 得分 | 说明 |
|---|---:|---|
| 1. optimizer 更新对象 | 15/15 | `named_parameters()` 只有 scale；offset 可得到梯度但未注册，默认 optimizer 只更新 scale。 |
| 2. 完整手算 | 29/30 | prediction、loss、两个梯度、更新值和新 loss 全部正确。loss 一行写成 `(1-4)^2`，应为 `(4-5)^2`；后续数值按正确关系计算，属于笔误。 |
| 3. 四步职责 | 25/25 | 正确区分 zero_grad、forward/loss、backward 写 `.grad`、step 修改参数。 |
| 4. 梯度累积 | 10/10 | 第二次为 -12，明确知道 backward 累加，普通步骤在 forward 前清梯度。 |
| 5. finite difference | 8/10 | central difference 公式和验证用途正确；恢复参数的核心是避免污染后续模型状态，不只是第二个采样点的写法。 |
| 6. 推理模式 | 10/10 | 正确区分 eval 的层行为切换与 inference_mode 的 autograd 关闭。 |

## 代码证据

```text
Lesson 0004: all checks passed.
autograd scale gradient: -16.0
finite-difference scale gradient: -15.99884033203125
updated scale/bias: 4.599999904632568 0.800000011920929
new loss: 0.0
gradient after two backward calls without zeroing: -32.0
inference prediction requires_grad: False
```

实现正确覆盖：

- `nn.Parameter` 注册 scale/bias；
- `zero_grad -> forward/loss -> backward -> step`；
- 在 step 前读取 grad、step 后读取新参数；
- `torch.no_grad()` 下做 central finite difference 并恢复 scale。

## 非阻塞代码提醒

`training_step()` 声明返回 `dict[str, float]`，但 `prediction` 和 `loss` 当前存入的是标量 Tensor。测试能够运行，但若要满足类型契约，应使用：

```python
ret["prediction"] = prediction.item()
ret["loss"] = loss.item()
```

不要求为此重做或补测。

## 下一步

进入第 0005 课：实现 RMSNorm 与 gated FFN，并与 PyTorch/reference 公式做数值对齐。
