"""PyTorch guided exploration of single-head causal attention."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return raw scores, future mask, masked scores, weights, and output."""
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("Q, K, and V must all have rank 3 [B,S,D]")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Q, K, and V must have equal [B,S,D] shapes in this lesson")

    _, sequence_length, head_dim = query.shape
    if head_dim <= 0:
        raise ValueError("head dimension D must be positive")

    # [B,S,D] @ [B,D,S] -> [B,S_query,S_key]. The D axis is summed.
    scores = query @ key.transpose(-2, -1) / math.sqrt(head_dim)

    # [S_query,S_key]. True above the diagonal means that future key is forbidden.
    future_mask = torch.triu(
        torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=1,
    )
    masked_scores = scores.masked_fill(future_mask, float("-inf"))

    # dim=-1 is the key axis: each fixed [b,q,:] row becomes a distribution.
    weights = F.softmax(masked_scores, dim=-1)

    # [B,S_query,S_key] @ [B,S_key,D] -> [B,S_query,D]. The key axis is summed.
    output = weights @ value
    return scores, future_mask, masked_scores, weights, output


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # B=1, S=3, D=1. These are real torch.Tensor objects, not Python matrices.
    query = torch.ones(1, 3, 1, device=device)
    key = torch.ones(1, 3, 1, device=device)
    value = torch.arange(1.0, 6.0, 2.0, device=device).reshape(1, 3, 1)

    scores, future_mask, masked_scores, weights, output = causal_attention(
        query, key, value
    )

    print("device:", device)
    print("Q is torch.Tensor:", isinstance(query, torch.Tensor))
    print("Q/K/V shape [B,S,D]:", tuple(query.shape))
    print("Q:\n", query)
    print("K:\n", key)
    print("V:\n", value)
    print("raw scores [B,S_query,S_key]:\n", scores)
    print("future mask [S_query,S_key] (True means forbidden):\n", future_mask)
    print("masked scores:\n", masked_scores)
    print("softmax weights along key axis dim=-1:\n", weights)
    print("weight row sums:\n", weights.sum(dim=-1))
    print("attention output [B,S,D]:\n", output)

    expected_weights = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]]],
        device=device,
    )
    expected_output = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 3, device=device))

    changed_value = value.clone()
    changed_value[:, 2, :] = 500.0
    _, _, _, changed_weights, changed_output = causal_attention(
        query, key, changed_value
    )
    torch.testing.assert_close(changed_weights, weights)
    torch.testing.assert_close(changed_output[:, :2, :], output[:, :2, :])
    assert not torch.equal(changed_output[:, 2, :], output[:, 2, :])
    print("changing V does not change attention weights: True")
    print("changing future V[2] leaves query positions 0 and 1 unchanged: True")

    # A D=2 scalar check connects PyTorch matmul back to the hand calculation.
    query_d2 = torch.tensor([[[1.0, 2.0]]], device=device)
    key_d2 = torch.tensor([[[1.0, 0.0]]], device=device)
    value_d2 = torch.tensor([[[7.0, 9.0]]], device=device)
    scores_d2, _, _, _, _ = causal_attention(query_d2, key_d2, value_d2)
    manual_score = (1.0 * 1.0 + 2.0 * 0.0) / math.sqrt(2)
    torch.testing.assert_close(scores_d2[0, 0, 0], torch.tensor(manual_score, device=device))
    print("D=2 score [1,2]·[1,0]/sqrt(2):", scores_d2[0, 0, 0].item())
    print("Lesson 0006 PyTorch guided exploration passed.")


if __name__ == "__main__":
    main()
