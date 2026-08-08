"""Lesson 0002: split and merge attention heads with PyTorch.

Run:
    python exercises/0002_head_layout.py

Only edit the two functions marked TODO.
"""

from __future__ import annotations

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Convert [batch, seq, hidden] to [batch, heads, seq, head_dim]."""
    # TODO 1:
    # - validate rank, num_heads, and divisibility of hidden
    # - reshape to [batch, seq, heads, head_dim]
    # - transpose seq and heads
    raise NotImplementedError


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Convert [batch, heads, seq, head_dim] back to [batch, seq, hidden]."""
    # TODO 2:
    # - validate rank
    # - transpose heads and seq
    # - make the layout safe for view(), then merge heads * head_dim
    raise NotImplementedError


def run_tests() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This lesson expects the configured CUDA environment")

    device = torch.device("cuda")
    x = torch.arange(2 * 3 * 8, dtype=torch.float32, device=device).reshape(2, 3, 8)

    heads = split_heads(x, num_heads=2)
    assert heads.shape == (2, 2, 3, 4)
    assert heads.device.type == "cuda"
    assert heads.dtype == torch.float32

    # hidden index = head_index * head_dim + dim_index
    assert heads[1, 1, 2, 3].item() == x[1, 2, 7].item()
    assert heads[0, 0, 1, 2].item() == x[0, 1, 2].item()

    restored = merge_heads(heads)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)
    assert restored.is_contiguous()

    try:
        split_heads(torch.zeros(2, 3, 10, device=device), num_heads=4)
    except ValueError:
        pass
    else:
        raise AssertionError("hidden not divisible by heads must raise ValueError")

    transposed = torch.arange(24, device=device).reshape(2, 3, 4).transpose(1, 2)
    assert not transposed.is_contiguous()
    assert transposed.reshape(2, 12).shape == (2, 12)

    print("Lesson 0002: all 11 checks passed.")
    print(f"split shape: {tuple(heads.shape)}")
    print(f"split stride: {heads.stride()}")
    print(f"restored contiguous: {restored.is_contiguous()}")


if __name__ == "__main__":
    run_tests()
