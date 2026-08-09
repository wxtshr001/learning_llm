"""Guided exploration of Module, Parameter, autograd, and optimizer roles.

Run from the repository root:
    python exercises/0004_explore_training_step.py
"""

from __future__ import annotations

import torch
from torch import nn


class ScalarAffine(nn.Module):
    """prediction = scale * x + bias, with two registered parameters."""

    def __init__(self, scale: float = 3.0, bias: float = 0.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.bias


def squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction - target).pow(2)


def finite_difference_scale(
    model: ScalarAffine,
    x: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> float:
    """Approximate d(loss)/d(scale) without building an autograd graph."""
    original = model.scale.detach().clone()
    with torch.no_grad():
        model.scale.copy_(original + epsilon)
        loss_plus = squared_error(model(x), target).item()
        model.scale.copy_(original - epsilon)
        loss_minus = squared_error(model(x), target).item()
        model.scale.copy_(original)
    return (loss_plus - loss_minus) / (2 * epsilon)


def main() -> None:
    model = ScalarAffine(scale=3.0, bias=0.0)
    x = torch.tensor(2.0)
    target = torch.tensor(10.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    print("=== 1. Registered parameters ===")
    for name, parameter in model.named_parameters():
        print(name, "value=", parameter.item(), "shape=", tuple(parameter.shape), "requires_grad=", parameter.requires_grad)

    scale_before = model.scale.detach().clone()
    bias_before = model.bias.detach().clone()
    optimizer.zero_grad()
    prediction = model(x)
    loss = squared_error(prediction, target)
    print("\n=== 2. Forward ===")
    print("prediction =", prediction.item())
    print("loss =", loss.item())
    print("parameters unchanged after forward =", bool(torch.equal(scale_before, model.scale) and torch.equal(bias_before, model.bias)))

    loss.backward()
    print("\n=== 3. Backward ===")
    print("scale.grad =", model.scale.grad.item())
    print("bias.grad =", model.bias.grad.item())
    print("parameters unchanged after backward =", bool(torch.equal(scale_before, model.scale) and torch.equal(bias_before, model.bias)))

    numeric_grad = finite_difference_scale(model, x, target)
    print("\n=== 4. Finite difference ===")
    print("autograd scale gradient =", model.scale.grad.item())
    print("finite-difference scale gradient =", numeric_grad)
    print("gradient absolute error =", abs(model.scale.grad.item() - numeric_grad))

    optimizer.step()
    print("\n=== 5. Optimizer step ===")
    print("updated scale =", model.scale.item())
    print("updated bias =", model.bias.item())
    new_prediction = model(x)
    new_loss = squared_error(new_prediction, target)
    print("new prediction =", new_prediction.item())
    print("new loss =", new_loss.item())

    model.eval()
    with torch.inference_mode():
        inference_prediction = model(x)
    print("\n=== 6. Inference ===")
    print("model.training =", model.training)
    print("prediction.requires_grad =", inference_prediction.requires_grad)

    torch.testing.assert_close(torch.tensor(numeric_grad), torch.tensor(-16.0), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(model.scale.detach(), torch.tensor(4.6), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.bias.detach(), torch.tensor(0.8), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(new_loss.detach(), torch.tensor(0.0), rtol=0.0, atol=1e-10)
    assert not inference_prediction.requires_grad
    print("\nLesson 0004 guided exploration passed.")


if __name__ == "__main__":
    main()
