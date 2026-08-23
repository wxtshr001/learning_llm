"""Independent assignment for lesson 0008: Qwen-style rotary position embedding."""

from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Return [-second_half, first_half] along the last axis."""
    # TODO 1: validate positive even D, then implement Qwen-style half rotation.
    raise NotImplementedError


def build_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    base: float = 10_000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin [B,S,D] for position_ids [B,S]."""
    # TODO 2: validate rank/integer positions, positive even D, and positive base.
    # TODO 3: build inv_freq[D/2], angles[B,S,D/2], then duplicate to D.
    raise NotImplementedError


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate Q [B,Nq,S,D] and K [B,Nkv,S,D] without changing shape."""
    # TODO 4: validate rank, B/S/D, device, dtype, cos/sin [B,S,D], and even D.
    # TODO 5: unsqueeze cos/sin on head axis and apply x*cos + rotate_half(x)*sin.
    raise NotImplementedError


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
