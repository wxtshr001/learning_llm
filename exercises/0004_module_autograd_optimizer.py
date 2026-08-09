"""Independent assignment for lesson 0004.

Complete the TODOs, then run:
    python exercises/0004_module_autograd_optimizer.py

Do not edit run_tests().
"""

from __future__ import annotations

import torch
from torch import nn


class ScalarAffine(nn.Module):
    """A scalar model: prediction = scale * x + bias."""

    def __init__(self, scale: float, bias: float) -> None:
        super().__init__()
        # TODO 1: register scale and bias as scalar float32 nn.Parameter objects.
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2: return scale * x + bias.
        return self.scale * x + self.bias


def squared_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the scalar squared error (prediction - target)^2."""
    return (prediction - target).pow(2)


def training_step(
    model: ScalarAffine,
    x: torch.Tensor,
    target: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run zero_grad -> forward/loss -> backward -> step and report evidence."""
    # TODO 3: implement the canonical training order. Capture prediction, loss,
    # scale.grad, and bias.grad before optimizer.step(), then return this dict:
    # {
    #   "prediction": ...,
    #   "loss": ...,
    #   "scale_grad": ...,
    #   "bias_grad": ...,
    #   "updated_scale": ...,
    #   "updated_bias": ...,
    # }
    ret = dict()
    optimizer.zero_grad()
    prediction = model(x)
    ret["prediction"] = prediction
    loss = squared_error(prediction, target)
    ret["loss"] = loss
    loss.backward()
    ret["scale_grad"] = model.scale.grad.item()
    ret["bias_grad"] = model.bias.grad.item()
    optimizer.step()
    ret["updated_scale"] = model.scale.item()
    ret["updated_bias"] = model.bias.item()
    return ret


def finite_difference_scale(
    model: ScalarAffine,
    x: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> float:
    """Approximate d(loss)/d(scale), restore scale, and return a Python float."""
    # TODO 4: evaluate loss at scale+epsilon and scale-epsilon under
    # torch.no_grad(), restore the original scale, then use central difference.
    original = model.scale.detach().clone()
    with torch.no_grad():
        model.scale.copy_(original + epsilon)
        loss_plus = squared_error(model(x), target).item()
        model.scale.copy_(original - epsilon)
        loss_minus = squared_error(model(x), target).item()
        model.scale.copy_(original)
    return (loss_plus - loss_minus) / (2 * epsilon)


def run_tests() -> None:
    model = ScalarAffine(scale=3.0, bias=0.0)
    parameter_names = dict(model.named_parameters())
    assert set(parameter_names) == {"scale", "bias"}
    assert parameter_names["scale"].shape == torch.Size([])
    assert parameter_names["bias"].shape == torch.Size([])
    assert parameter_names["scale"].requires_grad
    assert parameter_names["bias"].requires_grad

    x = torch.tensor(2.0)
    target = torch.tensor(10.0)
    torch.testing.assert_close(model(x), torch.tensor(6.0))

    numeric_grad = finite_difference_scale(model, x, target)
    torch.testing.assert_close(torch.tensor(numeric_grad), torch.tensor(-16.0), rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(model.scale.detach(), torch.tensor(3.0))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    evidence = training_step(model, x, target, optimizer)
    expected = {
        "prediction": 6.0,
        "loss": 16.0,
        "scale_grad": -16.0,
        "bias_grad": -8.0,
        "updated_scale": 4.6,
        "updated_bias": 0.8,
    }
    for key, expected_value in expected.items():
        assert abs(evidence[key] - expected_value) <= 1e-5, (key, evidence[key])

    new_loss = squared_error(model(x), target)
    torch.testing.assert_close(new_loss.detach(), torch.tensor(0.0), rtol=0.0, atol=1e-10)

    accumulation_model = ScalarAffine(scale=3.0, bias=0.0)
    first_loss = squared_error(accumulation_model(x), target)
    first_loss.backward()
    first_scale_grad = accumulation_model.scale.grad.item()
    second_loss = squared_error(accumulation_model(x), target)
    second_loss.backward()
    second_scale_grad = accumulation_model.scale.grad.item()
    assert first_scale_grad == -16.0
    assert second_scale_grad == -32.0

    accumulation_model.eval()
    with torch.inference_mode():
        inference_prediction = accumulation_model(x)
    assert not accumulation_model.training
    assert not inference_prediction.requires_grad

    print("Lesson 0004: all checks passed.")
    print("autograd scale gradient:", evidence["scale_grad"])
    print("finite-difference scale gradient:", numeric_grad)
    print("updated scale/bias:", evidence["updated_scale"], evidence["updated_bias"])
    print("new loss:", new_loss.item())
    print("gradient after two backward calls without zeroing:", second_scale_grad)
    print("inference prediction requires_grad:", inference_prediction.requires_grad)


if __name__ == "__main__":
    run_tests()
