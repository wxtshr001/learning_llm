"""Independent assignment for lesson 0003: dtype and device.

Complete the three TODO functions, then run:
    python exercises/0003_tensor_dtype_device.py

Do not edit run_tests().
"""

from __future__ import annotations

import torch


def tensor_data_bytes(tensor: torch.Tensor) -> int:
    """Return bytes occupied by the tensor's elements.

    Count only element storage described by this tensor:
        number of elements * bytes per element
    Do not use a hard-coded dtype table.
    """
    # TODO 1: use Tensor methods to calculate the result.
    return tensor.numel() * tensor.element_size()


def prepare_linear_inputs(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return X, W, and b converted to the same device and dtype.

    Do not modify the original tensors in place.
    """
    # TODO 2: convert all three tensors explicitly and return them.
    return (x.to(device = device, dtype=dtype), weight.to(device = device, dtype=dtype), bias.to(device = device, dtype=dtype))


def max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return max(abs(reference - candidate)) as a Python float.

    Both inputs must already have the same shape, dtype, and device.
    """
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype or reference.device != candidate.device:
        raise ValueError("invalid input!")
    # TODO 3: calculate the elementwise absolute difference and its maximum.
    return (reference - candidate).abs().max().item()


def run_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run Y = X @ W + b and return the result as CPU float32."""
    x_compute, weight_compute, bias_compute = prepare_linear_inputs(
        x, weight, bias, device, dtype
    )
    output = x_compute @ weight_compute + bias_compute
    return output.to(device="cpu", dtype=torch.float32)


def run_tests() -> None:
    x = torch.tensor([[0.25, -0.5, 1.0], [1.5, 0.2, -0.75]], dtype=torch.float32)
    weight = torch.tensor([[0.5, 1.0], [-1.0, 0.25], [2.0, -0.5]], dtype=torch.float32)
    bias = torch.tensor([0.1, -0.2], dtype=torch.float32)

    # [2,3] has 6 elements: float32 uses 24 bytes; float16 uses 12 bytes.
    assert tensor_data_bytes(x) == 24
    assert tensor_data_bytes(x.to(dtype=torch.float16)) == 12

    target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    moved_x, moved_weight, moved_bias = prepare_linear_inputs(
        x, weight, bias, target, torch.float16
    )
    for tensor in (moved_x, moved_weight, moved_bias):
        assert tensor.device == target
        assert tensor.dtype == torch.float16
    assert x.device.type == "cpu" and x.dtype == torch.float32
    assert weight.device.type == "cpu" and weight.dtype == torch.float32
    assert bias.device.type == "cpu" and bias.dtype == torch.float32

    reference = run_linear(x, weight, bias, torch.device("cpu"), torch.float32)
    expected = torch.tensor([[2.725, -0.575], [-0.85, 1.725]], dtype=torch.float32)
    torch.testing.assert_close(reference, expected, rtol=1e-6, atol=1e-6)

    candidate32 = run_linear(x, weight, bias, target, torch.float32)
    candidate16 = run_linear(x, weight, bias, target, torch.float16)
    error32 = max_abs_error(reference, candidate32)
    error16 = max_abs_error(reference, candidate16)
    assert error32 <= 1e-5
    assert error16 <= 2e-3

    print("Lesson 0003: all checks passed.")
    print("target device:", target)
    print("float32 data bytes:", tensor_data_bytes(x))
    print("float16 data bytes:", tensor_data_bytes(x.to(dtype=torch.float16)))
    print("float32 max absolute error:", error32)
    print("float16 max absolute error:", error16)


if __name__ == "__main__":
    run_tests()
