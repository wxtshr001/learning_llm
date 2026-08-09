"""Lesson 0002 homework: split and merge attention heads with PyTorch.

Prerequisite:
    First run exercises/0002_explore_head_layout.py and understand its output.

Run from the repository root:
    conda activate llm
    python exercises/0002_head_layout.py

Symbol legend used below:
    B / batch_size     = number of sequences processed together
    S / sequence_len   = number of tokens in each sequence
    H / hidden_size    = total features representing one token
    N / num_heads      = number of attention heads
    D / head_dim       = features per head; D = H // N

Only edit the two functions marked TODO.
"""

from __future__ import annotations

import torch


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Change [B, S, H] into [B, N, S, D].

    Example:
        input shape:  [B=2, S=3, H=8]
        num_heads:    N=2
        head_dim:     D=H/N=4
        output shape: [B=2, N=2, S=3, D=4]

    The values are regrouped, not created or deleted.
    """
    # TODO 1, in this exact reasoning order:
    # 1. x must have rank 3, meaning exactly [B, S, H].
    # 2. num_heads must be greater than zero.
    # 3. Read B, S, H from x.shape.
    # 4. H must be divisible by num_heads; otherwise D is not an integer.
    # 5. Compute D = H // num_heads.
    # 6. Reshape [B, S, H] -> [B, S, N, D].
    # 7. Transpose dim 1 and dim 2 -> [B, N, S, D].
    B, S, H = x.shape
    if len(x.shape) != 3 or num_heads <= 0:
        raise ValueError("Not valid x/num_heads")
    if H % num_heads != 0:
        raise ValueError("Not a valid")
    D = H // num_heads
    ret = x.reshape(B, S, num_heads, D).transpose(1, 2)
    return ret


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """Change [B, N, S, D] back into [B, S, H], where H=N*D.

    This is the inverse layout operation of split_heads().
    """
    # TODO 2, in this exact reasoning order:
    # 1. x must have rank 4, meaning exactly [B, N, S, D].
    # 2. Read B, N, S, D from x.shape.
    # 3. Transpose dim 1 and dim 2 -> [B, S, N, D].
    # 4. Make that tensor contiguous because transpose usually changes strides.
    # 5. Merge N and D -> H=N*D, producing [B, S, H].
    B, N, S, D = x.shape
    ret = x.transpose(1, 2).contiguous().view(B, S, N*D)
    return ret



def run_tests() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Concrete test dimensions:
    # B=2 sequences, S=3 tokens, H=8 total features per token.
    # N=2 heads, so D=4 features per head.
    x = torch.arange(2 * 3 * 8, dtype=torch.float32, device=device).reshape(2, 3, 8)

    heads = split_heads(x, num_heads=2)
    assert heads.shape == (2, 2, 3, 4)  # [B=2, N=2, S=3, D=4]
    assert heads.device == device
    assert heads.dtype == torch.float32

    # General mapping: heads[b,n,s,d] == x[b,s,n*D+d].
    # Here D=4, so head n=1 and position d=3 map to hidden index 1*4+3=7.
    assert heads[1, 1, 2, 3].item() == x[1, 2, 7].item()
    # Head n=0 and position d=2 map to hidden index 0*4+2=2.
    assert heads[0, 0, 1, 2].item() == x[0, 1, 2].item()

    restored = merge_heads(heads)
    assert restored.shape == x.shape
    assert torch.equal(restored, x)
    assert restored.is_contiguous()

    # H=10 cannot be split evenly across N=4 heads.
    try:
        split_heads(torch.zeros(2, 3, 10, device=device), num_heads=4)
    except ValueError:
        pass
    else:
        raise AssertionError("hidden not divisible by heads must raise ValueError")

    # transpose usually creates a non-contiguous tensor view.
    transposed = torch.arange(24, device=device).reshape(2, 3, 4).transpose(1, 2)
    assert not transposed.is_contiguous()
    # reshape is allowed to create a copy when a view is impossible.
    assert transposed.reshape(2, 12).shape == (2, 12)

    print("Lesson 0002: all 11 checks passed.")
    print(f"split shape [B,N,S,D]: {tuple(heads.shape)}")
    print(f"split stride: {heads.stride()}")
    print(f"restored shape [B,S,H]: {tuple(restored.shape)}")
    print(f"restored contiguous: {restored.is_contiguous()}")


if __name__ == "__main__":
    run_tests()
