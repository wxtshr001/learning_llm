"""Guided PyTorch exploration for lesson 0008: Qwen-style RoPE."""

from __future__ import annotations

import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Qwen/Hugging Face layout: [first_half, second_half] -> [-second, first]."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply [B,S,D] cos/sin to Q/K shaped [B,N,S,D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def main() -> None:
    print("=== 1. Qwen half-split pairing ===")
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated_half = rotate_half(x)
    print("x:", x.tolist())
    print("paired axes: (0,2), (1,3)")
    print("rotate_half(x):", rotated_half.tolist())
    torch.testing.assert_close(rotated_half, torch.tensor([-3.0, -4.0, 1.0, 2.0]))

    print("\n=== 2. One exact position rotation ===")
    # Pair (0,2) rotates 90 degrees; pair (1,3) rotates 0 degrees.
    cos = torch.tensor([0.0, 1.0, 0.0, 1.0])
    sin = torch.tensor([1.0, 0.0, 1.0, 0.0])
    x_rope = x * cos + rotate_half(x) * sin
    print("cos:", cos.tolist())
    print("sin:", sin.tolist())
    print("x_rope:", x_rope.tolist())
    torch.testing.assert_close(x_rope, torch.tensor([-3.0, 2.0, 1.0, 4.0]))
    torch.testing.assert_close(torch.linalg.vector_norm(x_rope), torch.linalg.vector_norm(x))

    print("\n=== 3. Same relative distance gives the same score ===")
    q0 = torch.tensor([1.0, 0.0])
    q1 = torch.tensor([0.0, 1.0])
    k1 = torch.tensor([0.0, 1.0])
    k2 = torch.tensor([-1.0, 0.0])
    score_0_to_1 = torch.dot(q0, k1)
    score_1_to_2 = torch.dot(q1, k2)
    score_0_to_2 = torch.dot(q0, k2)
    print("score(pos 0, pos 1):", score_0_to_1.item())
    print("score(pos 1, pos 2):", score_1_to_2.item())
    print("score(pos 0, pos 2):", score_0_to_2.item())
    assert score_0_to_1.item() == score_1_to_2.item() == 0.0
    assert score_0_to_2.item() == -1.0

    print("\n=== 4. GQA shapes and broadcast ===")
    batch, sequence, head_dim = 1, 3, 4
    query = torch.arange(1, 1 + batch * 4 * sequence * head_dim, dtype=torch.float32)
    query = query.reshape(batch, 4, sequence, head_dim)
    key = torch.arange(1, 1 + batch * 2 * sequence * head_dim, dtype=torch.float32)
    key = key.reshape(batch, 2, sequence, head_dim)
    position_ids = torch.arange(sequence).unsqueeze(0)
    angles = position_ids[..., None].float() * torch.tensor([1.0, 0.5])
    embedding = torch.cat((angles, angles), dim=-1)
    q_rope, k_rope = apply_rope(query, key, embedding.cos(), embedding.sin())
    print("position_ids [B,S]:", tuple(position_ids.shape), position_ids.tolist())
    print("cos/sin [B,S,D]:", tuple(embedding.shape))
    print("after unsqueeze [B,1,S,D]:", (batch, 1, sequence, head_dim))
    print("Q/Q_rope [B,Nq,S,D]:", tuple(query.shape), tuple(q_rope.shape))
    print("K/K_rope [B,Nkv,S,D]:", tuple(key.shape), tuple(k_rope.shape))
    print("position 0 is identity:", torch.equal(q_rope[:, :, 0], query[:, :, 0]))
    assert q_rope.shape == query.shape
    assert k_rope.shape == key.shape
    torch.testing.assert_close(q_rope[:, :, 0], query[:, :, 0])
    torch.testing.assert_close(k_rope[:, :, 0], key[:, :, 0])
    torch.testing.assert_close(
        torch.linalg.vector_norm(q_rope, dim=-1),
        torch.linalg.vector_norm(query, dim=-1),
        rtol=1e-5,
        atol=1e-5,
    )
    assert math.isclose(float(embedding.cos()[0, 0, 0]), 1.0)
    print("Lesson 0008 guided RoPE exploration passed.")


if __name__ == "__main__":
    main()
