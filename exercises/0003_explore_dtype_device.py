"""Guided exploration for Tensor dtype, device, memory, and numerical error.

Run from the repository root:
    python exercises/0003_explore_dtype_device.py

This file is a completed demonstration, not the independent assignment.
"""

from __future__ import annotations

import torch


def describe(name: str, tensor: torch.Tensor) -> None:
    """Print the metadata introduced in lesson 0003."""
    print(
        f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"device={tensor.device}, numel={tensor.numel()}, "
        f"element_size={tensor.element_size()}, "
        f"data_bytes={tensor.numel() * tensor.element_size()}"
    )


def max_abs_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Compare two CPU float32 tensors and return their largest absolute error."""
    return (reference - candidate).abs().max().item()


def linear_on(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run Y = X @ W + b on the requested device/dtype and return CPU float32."""
    x_compute = x.to(device=device, dtype=dtype)
    weight_compute = weight.to(device=device, dtype=dtype)
    bias_compute = bias.to(device=device, dtype=dtype)
    output = x_compute @ weight_compute + bias_compute
    return output.to(device="cpu", dtype=torch.float32)


def main() -> None:
    print("=== 1. dtype inference ===")
    integers = torch.tensor([1, 2, 3])
    decimals = torch.tensor([1.0, 2.0, 3.0])
    describe("integers", integers)
    describe("decimals", decimals)
    assert integers.dtype == torch.int64
    assert decimals.dtype == torch.float32

    print("\n=== 2. Same shape, different dtype and bytes ===")
    x32 = torch.tensor([[0.1, 0.2, 0.3], [1000.3, -2.7, 3.5]], dtype=torch.float32)
    x16 = x32.to(dtype=torch.float16)
    describe("x32", x32)
    describe("x16", x16)
    assert x32.shape == x16.shape == (2, 3)
    assert x32.numel() * x32.element_size() == 24
    assert x16.numel() * x16.element_size() == 12

    round_trip = x16.to(dtype=torch.float32)
    print("x32 values       =", x32.tolist())
    print("float16 roundtrip=", round_trip.tolist())
    print("roundtrip max absolute error =", max_abs_error(x32, round_trip))

    print("\n=== 3. Same Linear on a target device/dtype ===")
    # X: [B=2,Hin=3], W: [Hin=3,Hout=2], b: [Hout=2].
    x = torch.tensor([[0.1, 0.2, 0.3], [1.1, -0.7, 0.25]], dtype=torch.float32)
    weight = torch.tensor([[0.5, -0.25], [1.5, 0.75], [-0.8, 2.0]], dtype=torch.float32)
    bias = torch.tensor([0.05, -0.1], dtype=torch.float32)

    reference = linear_on(x, weight, bias, torch.device("cpu"), torch.float32)
    target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_fp32 = linear_on(x, weight, bias, target, torch.float32)
    target_fp16 = linear_on(x, weight, bias, target, torch.float16)

    print("target device =", target)
    print("CPU float32 reference =\n", reference)
    print("target float32 output =\n", target_fp32)
    print("target float16 output returned as CPU float32 =\n", target_fp16)
    print("float32 max absolute error =", max_abs_error(reference, target_fp32))
    print("float16 max absolute error =", max_abs_error(reference, target_fp16))
    torch.testing.assert_close(reference, target_fp32, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference, target_fp16, rtol=2e-3, atol=2e-3)

    print("\n=== 4. Mixed-device boundary ===")
    if torch.cuda.is_available():
        cpu_vector = torch.ones(2, device="cpu")
        cuda_vector = torch.ones(2, device="cuda:0")
        try:
            _ = cpu_vector + cuda_vector
        except RuntimeError as error:
            print("CPU tensor + CUDA tensor failed as expected:")
            print(str(error).splitlines()[0])
        else:
            raise AssertionError("Expected a device mismatch RuntimeError")
    else:
        print("CUDA is unavailable; mixed-device failure demo skipped.")

    print("\nLesson 0003 guided exploration passed.")


if __name__ == "__main__":
    main()
