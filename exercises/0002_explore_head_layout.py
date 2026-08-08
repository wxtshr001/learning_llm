"""Explore reshape, transpose, strides, and attention-head layout.

This file is a guided demonstration, not homework. Read the output slowly.

Run from the repository root:
    conda activate llm
    python exercises/0002_explore_head_layout.py
"""

from __future__ import annotations

import torch


def describe(name: str, tensor: torch.Tensor) -> None:
    """Print the metadata used throughout this lesson."""
    print(f"{name}.shape       = {tuple(tensor.shape)}")
    print(f"{name}.stride      = {tensor.stride()}")
    print(f"{name}.ndim        = {tensor.ndim}")
    print(f"{name}.contiguous  = {tensor.is_contiguous()}")
    print(f"{name}.device      = {tensor.device}")
    print()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Symbol definitions:
    # B = batch size: number of sequences processed together
    # S = sequence length: number of tokens in each sequence
    # H = hidden size: total features representing one token
    # N = number of attention heads
    # D = head dimension: features assigned to one head
    batch_size = 1       # B
    sequence_length = 3  # S
    hidden_size = 8      # H
    num_heads = 2        # N
    head_dim = hidden_size // num_heads  # D = 4

    x = torch.arange(
        batch_size * sequence_length * hidden_size,
        dtype=torch.float32,
        device=device,
    ).reshape(batch_size, sequence_length, hidden_size)
    print("=== 1. Original tensor: [B, S, H] ===")
    describe("x", x)
    print("x[0, 0] (token index 0) =", x[0, 0].tolist())
    print("x[0, 1] (token index 1) =", x[0, 1].tolist())
    print()

    reshaped = x.reshape(batch_size, sequence_length, num_heads, head_dim)
    print("=== 2. Split H into N x D: [B, S, N, D] ===")
    describe("reshaped", reshaped)
    print("token 0, head 0 =", reshaped[0, 0, 0].tolist())
    print("token 0, head 1 =", reshaped[0, 0, 1].tolist())
    print("mapping check: reshaped[0,0,1,2] == x[0,0,6]")
    print(reshaped[0, 0, 1, 2].item(), "==", x[0, 0, 6].item())
    print()

    heads = reshaped.transpose(1, 2)
    print("=== 3. Swap S and N: [B, N, S, D] ===")
    describe("heads", heads)
    print("head 0 across all tokens:")
    print(heads[0, 0].cpu())
    print("head 1 across all tokens:")
    print(heads[0, 1].cpu())
    print("mapping check: heads[0,1,0,2] == x[0,0,6]")
    print(heads[0, 1, 0, 2].item(), "==", x[0, 0, 6].item())
    print()

    print("=== 4. Storage and contiguous layout ===")
    print("reshaped and heads share the same data pointer:", reshaped.data_ptr() == heads.data_ptr())
    print("transpose changed logical axis order without moving the storage.")
    print()

    try:
        heads.view(batch_size, sequence_length, hidden_size)
    except RuntimeError as error:
        print("Direct view after transpose failed, as expected:")
        print(str(error).splitlines()[0])
    print()

    restored = heads.transpose(1, 2).contiguous().view(
        batch_size, sequence_length, hidden_size
    )
    print("=== 5. Merge heads back to [B, S, H] ===")
    describe("restored", restored)
    print("restored equals x:", torch.equal(restored, x))


if __name__ == "__main__":
    main()
