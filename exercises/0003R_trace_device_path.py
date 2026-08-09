"""Guided trace for the complete device/dtype execution path.

Run from the repository root:
    python exercises/0003R_trace_device_path.py
"""

from __future__ import annotations

import torch


def metadata(name: str, tensor: torch.Tensor) -> None:
    print(f"{name}: device={tensor.device}, dtype={tensor.dtype}, shape={tuple(tensor.shape)}")


def main() -> None:
    target = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.tensor([[0.25, -0.5]], dtype=torch.float32)
    weight = torch.tensor([[2.0], [1.0]], dtype=torch.float32)
    bias = torch.tensor([0.1], dtype=torch.float32)

    reference = x @ weight + bias
    x_compute = x.to(device=target, dtype=torch.float16)
    weight_compute = weight.to(device=target, dtype=torch.float16)
    bias_compute = bias.to(device=target, dtype=torch.float16)
    y_compute = x_compute @ weight_compute + bias_compute
    candidate = y_compute.to(device="cpu", dtype=torch.float32)

    print("=== compute inputs ===")
    metadata("X", x_compute)
    metadata("W", weight_compute)
    metadata("b", bias_compute)
    metadata("Y", y_compute)

    print("\n=== comparison inputs ===")
    metadata("reference", reference)
    metadata("candidate", candidate)
    error = (reference - candidate).abs()
    print("absolute error:", error.tolist())
    print("max absolute error:", error.max().item())

    assert x_compute.device == weight_compute.device == bias_compute.device == y_compute.device
    assert x_compute.dtype == weight_compute.dtype == bias_compute.dtype == y_compute.dtype
    assert reference.device == candidate.device == torch.device("cpu")
    assert reference.dtype == candidate.dtype == torch.float32
    print("\nLesson 0003R guided trace passed.")


if __name__ == "__main__":
    main()
