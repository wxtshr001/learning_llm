"""Verify the dedicated LLM learning environment."""

from __future__ import annotations

import numpy as np
import torch


def main() -> None:
    print(f"PyTorch: {torch.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the llm environment")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(device)}")

    torch.manual_seed(7)
    x = torch.randn(128, 64, device=device, requires_grad=True)
    w = torch.randn(64, 32, device=device, requires_grad=True)
    loss = (x @ w).square().mean()
    loss.backward()
    torch.cuda.synchronize()

    assert x.grad is not None and x.grad.shape == x.shape
    assert w.grad is not None and w.grad.shape == w.shape
    assert torch.isfinite(loss).item()
    print(f"GPU loss: {loss.detach().item():.6f}")
    print("PyTorch CUDA matmul and autograd: PASS")


if __name__ == "__main__":
    main()
