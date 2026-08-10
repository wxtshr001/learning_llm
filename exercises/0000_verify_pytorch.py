"""Verify the dedicated LLM learning environment."""

from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail when CUDA is unavailable instead of validating the CPU backend",
    )
    args = parser.parse_args()

    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was required but is unavailable in this environment")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    torch.manual_seed(7)
    x = torch.randn(128, 64, device=device, requires_grad=True)
    w = torch.randn(64, 32, device=device, requires_grad=True)
    loss = (x @ w).square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert torch.isfinite(loss).item()
    print(f"Loss: {loss.detach().item():.6f}")
    print(f"PyTorch {device.type.upper()} matmul and autograd: PASS")


if __name__ == "__main__":
    main()
