"""Independent assignment for lesson 0008: Qwen-style rotary position embedding."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Return [-second_half, first_half] along the last axis."""
    # TODO 1: validate positive even D, then implement Qwen-style half rotation.
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")

    head_dim = x.shape[-1]
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(
            f"last dimension must be positive and even, got {head_dim}"
        )

    half = head_dim // 2
    first_half = x[..., :half]
    second_half = x[..., half:]
    return torch.cat((-second_half, first_half), dim=-1)

def build_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin [B,S,D] for position_ids [B,S]."""
    # TODO 2: validate rank/integer positions, positive even D, and positive base.
    # TODO 3: build inv_freq[D/2], angles[B,S,D/2], then duplicate to D.
    if position_ids.ndim != 2:
        raise ValueError(
            f"position_ids must be rank 2 [B,S], got shape {tuple(position_ids.shape)}"
        )
    if position_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError(f"position_ids must be integer, got {position_ids.dtype}")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    device = position_ids.device
    dtype = torch.float32

    # inv_freq: [D/2]
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, head_dim, 2, device=device, dtype=dtype)
            / head_dim
        )
    )

    # position_ids: [B,S] -> [B,S,1]
    # inv_freq:     [D/2] -> [1,1,D/2]
    angles = torch.einsum('bs,i->bsi', position_ids, inv_freq)

    # Qwen-style: duplicate frequencies to full head_dim.
    emb = torch.cat((angles, angles), dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q [B,Nq,S,D] and K [B,Nkv,S,D] without changing shape."""
    # TODO 4: validate rank, B/S/D, device, dtype, cos/sin [B,S,D], and even D.
    # TODO 5: unsqueeze cos/sin on head axis and apply x*cos + rotate_half(x)*sin.
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError(
            f"query and key must be rank 4 [B,N,S,D], got {query.ndim} and {key.ndim}"
        )
    if cos.ndim != 3 or sin.ndim != 3:
        raise ValueError(
            f"cos and sin must be rank 3 [B,S,D], got {cos.ndim} and {sin.ndim}"
        )

    B, Nq, S, D = query.shape
    Bk, Nkv, Sk, Dk = key.shape
    if B != Bk or S != Sk or D != Dk:
        raise ValueError(
            "query and key must share B/S/D, "
            f"got query {tuple(query.shape)} and key {tuple(key.shape)}"
        )
    if D <= 0 or D % 2 != 0:
        raise ValueError(f"last dimension must be positive and even, got {D}")
    if cos.shape != (B, S, D) or sin.shape != (B, S, D):
        raise ValueError(
            f"cos/sin must have shape {(B, S, D)}, "
            f"got {tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    if query.device != key.device or query.device != cos.device or query.device != sin.device:
        raise ValueError("query, key, cos, sin must be on the same device")
    if query.dtype != key.dtype or query.dtype != cos.dtype or query.dtype != sin.dtype:
        raise ValueError("query, key, cos, sin must have the same dtype")
    if not query.is_floating_point():
        raise ValueError("query and key must be floating point")

    cos = cos.unsqueeze(1)  # [B, 1, S, D]
    sin = sin.unsqueeze(1)  # [B, 1, S, D]

    query_rotated = query * cos + rotate_half(query) * sin
    key_rotated = key * cos + rotate_half(key) * sin
    return query_rotated, key_rotated


def run_tests() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    query = torch.arange(1, 1 + 1 * 4 * 3 * 4, device=device, dtype=torch.float32)
    query = query.reshape(1, 4, 3, 4)
    key = torch.arange(1, 1 + 1 * 2 * 3 * 4, device=device, dtype=torch.float32)
    key = key.reshape(1, 2, 3, 4)
    value = key + 1000
    position_ids = torch.tensor([[0, 1, 2]], device=device)

    cos, sin = build_rope_cos_sin(position_ids, head_dim=4)
    query_rotated, key_rotated = apply_rope(query, key, cos, sin)

    assert cos.shape == sin.shape == (1, 3, 4)
    assert query_rotated.shape == query.shape
    assert key_rotated.shape == key.shape
    torch.testing.assert_close(query_rotated[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(key_rotated[:, :, 0], key[:, :, 0])
    torch.testing.assert_close(
        torch.linalg.vector_norm(query_rotated, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(value, key + 1000)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    torch.testing.assert_close(
        rotate_half(x),
        torch.tensor([-3.0, -4.0, 1.0, 2.0], device=device),
    )
    manual_cos = torch.tensor([[[0.0, 1.0, 0.0, 1.0]]], device=device)
    manual_sin = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], device=device)
    manual_q = x.reshape(1, 1, 1, 4)
    manual_k = (x * 10).reshape(1, 1, 1, 4)
    manual_q_rotated, manual_k_rotated = apply_rope(
        manual_q, manual_k, manual_cos, manual_sin
    )
    torch.testing.assert_close(
        manual_q_rotated.flatten(),
        torch.tensor([-3.0, 2.0, 1.0, 4.0], device=device),
    )
    torch.testing.assert_close(
        manual_k_rotated.flatten(),
        torch.tensor([-30.0, 20.0, 10.0, 40.0], device=device),
    )

    invalid_calls = [
        lambda: rotate_half(torch.zeros(3, device=device)),
        lambda: build_rope_cos_sin(position_ids, head_dim=3),
        lambda: apply_rope(query, key[..., :2], cos, sin),
        lambda: apply_rope(query, key, cos.to(torch.float64), sin.to(torch.float64)),
    ]
    for call in invalid_calls:
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid RoPE contract must raise ValueError")

    print("Lesson 0008: all RoPE checks passed.")
    print("device:", device)
    print("position_ids:", position_ids)
    print("cos/sin shape:", tuple(cos.shape), tuple(sin.shape))
    print("Q/K rotated shape:", tuple(query_rotated.shape), tuple(key_rotated.shape))
    print("manual rotated Q:", manual_q_rotated.flatten())
    print("V unchanged: True")


if __name__ == "__main__":
    run_tests()
