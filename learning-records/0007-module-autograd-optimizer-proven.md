# 第 0004 课正式通过：Module、Autograd 与 Optimizer 职责

第 0004 课最终成绩 97/100，代码全部检查通过，关键手算与职责题通过。

## 已证明

- 能用 `nn.Parameter` 注册可训练标量，并解释普通 requires-grad Tensor 不会自动进入 `model.parameters()`。
- 能手算 squared error 对 prediction、scale 和 bias 的梯度。
- 能正确执行 `zero_grad -> forward/loss -> backward -> optimizer.step`。
- 能区分 forward 计算、backward 累积 `.grad` 和 optimizer 修改参数。
- 能解释并验证梯度累积。
- 能用 central finite difference 对齐 autograd 梯度，并在扰动后恢复参数。
- 能区分 `model.eval()` 与 `torch.inference_mode()`。

## 非阻塞提醒

- 两处公式变量名/抄写笔误没有改变后续数值与推理链，不作为概念缺口。
- `training_step()` 的 prediction/loss 返回标量 Tensor，而类型注解写 `dict[str,float]`；以后生产代码应让运行值满足接口声明。

## Evidence

- `exercises/0004_module_autograd_optimizer.py`
- `submissions/0004.md`
- `submissions/0004-feedback.md`
