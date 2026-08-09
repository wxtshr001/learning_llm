# 0004 闭卷测试：Module、Parameter、Autograd 与 Optimizer

完成探索脚本和独立作业后，关闭课程、源码和速查表作答。

## 符号表

- `x`：输入，不是模型参数。
- `target`：目标值。
- `scale`、`bias`：模型中的两个可训练参数。
- `prediction`：模型前向计算得到的预测值。
- `loss`：预测误差，本测试使用 `(prediction-target)^2`。
- `lr`：learning rate，学习率。
- `grad`：gradient，梯度。

## 提交证据

1. 粘贴 `0004_explore_training_step.py` 的完整输出。
2. 粘贴完成 TODO 后 `0004_module_autograd_optimizer.py` 的完整输出。

## 1. 哪些对象会被 optimizer 更新（15 分）

```python
class Example(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))
        self.offset = torch.tensor(1.0, requires_grad=True)

    def forward(self, x):
        return self.scale * x + self.offset
```

1. `model.named_parameters()` 默认包含哪些名字？
2. `offset` 能否通过 backward 得到梯度？
3. `torch.optim.SGD(model.parameters(), lr=0.1)` 默认会更新谁？为什么？

## 2. 一次完整手算（关键题，30 分）

```text
x = 3
target = 5
scale = 1
bias = 1
loss = (prediction-target)^2
lr = 0.05
```

依次计算并写出过程：

1. prediction 和 loss；
2. `d(loss)/d(prediction)`；
3. `scale.grad` 和 `bias.grad`；
4. SGD 更新后的 scale 和 bias；
5. 使用新参数重新计算 prediction 和 loss。

## 3. 四个步骤的职责（关键题，25 分）

对下面每项分别写“负责什么”和“不负责什么”：

1. `optimizer.zero_grad()`；
2. `prediction = model(x)` 与 loss 计算；
3. `loss.backward()`；
4. `optimizer.step()`。

必须明确指出：哪一步修改参数，哪一步把梯度写入 `.grad`。

## 4. 梯度累积（10 分）

同一组参数连续对两个等价新 forward 得到的 loss 调用 backward，中间没有 zero_grad。第一次 `scale.grad=-6`：

1. 第二次 backward 后 `scale.grad` 是多少？
2. 为什么不是仍然等于 -6？
3. 如果这是普通独立训练步骤，应该在哪里清梯度？

## 5. finite difference（10 分）

1. 写出 central finite difference 近似某个参数梯度的公式。
2. 为什么扰动参数后必须恢复原值？
3. 它在本课中是训练方法还是验证方法？

## 6. 推理模式（10 分）

1. `model.eval()` 是否会关闭 autograd？
2. `torch.inference_mode()` 负责什么？
3. 为什么实际推理代码经常同时使用二者？

## 通过标准

- 总分至少 80 分。
- 独立作业、第 2 题和第 3 题全部通过。
- 必须区分 backward 计算梯度与 optimizer 修改参数。
- 未通过时只补实际缺口，不重新考已经通过的 dtype/device。
