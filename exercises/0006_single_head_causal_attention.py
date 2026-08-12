"""Independent assignment for lesson 0006: single-head causal attention."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def make_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    """Return bool [S,S]; True means the key position is forbidden."""
    # TODO 1: positions above the main diagonal are future positions.
    ret=torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device = device), diagonal=1)
    return ret

def single_head_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute output [B,S,D] and weights [B,S,S]."""
    # TODO 2: require rank 3, equal Q/K/V shapes, and a positive D.
    # TODO 3: scores = Q @ K^T / sqrt(D), shape [B,S,S].
    # TODO 4: mask future keys before softmax over the key axis.
    # TODO 5: output = weights @ V, shape [B,S,D].
    if query.ndim!=3 or key.ndim!=3 or value.ndim!=3:
        raise ValueError("invalid shape")
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("invalid shape")
    _, sequence_length, dim_head = query.shape
    if dim_head < 0:
        raise ValueError("invalid dim")

    scores=query@key.transpose(-2,-1) / math.sqrt(dim_head)
    masked_scores = scores.masked_fill(make_causal_mask(sequence_length, torch.device("cpu")), float("-inf"))
    weights = F.softmax(masked_scores, dim=-1)
    output = weights@value
    return output, weights


def run_tests() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    query = torch.ones(1, 3, 1, device=device)
    key = torch.ones(1, 3, 1, device=device)
    value = torch.tensor([[[1.0], [3.0], [5.0]]], device=device)

    output, weights = single_head_causal_attention(query, key, value)
    expected_weights = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1 / 3, 1 / 3, 1 / 3]]],
        device=device,
    )
    expected_output = torch.tensor([[[1.0], [2.0], [3.0]]], device=device)
    assert output.shape == (1, 3, 1)
    assert weights.shape == (1, 3, 3)
    torch.testing.assert_close(weights, expected_weights)
    torch.testing.assert_close(output, expected_output)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 3, device=device))
    assert torch.equal(weights[0].triu(diagonal=1), torch.zeros(3, 3, device=device))

    changed_value = value.clone()
    changed_value[:, 2, :] = 500.0
    changed_output, _ = single_head_causal_attention(query, key, changed_value)
    torch.testing.assert_close(changed_output[:, :2, :], output[:, :2, :])
    assert not torch.equal(changed_output[:, 2, :], output[:, 2, :])

    query2 = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device)
    key2 = query2.clone()
    value2 = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]], device=device)
    output2, weights2 = single_head_causal_attention(query2, key2, value2)
    scores2 = query2 @ key2.transpose(-2, -1) / math.sqrt(2)
    mask2 = torch.tensor([[False, True], [False, False]], device=device)
    reference_weights2 = F.softmax(scores2.masked_fill(mask2, float("-inf")), dim=-1)
    reference_output2 = reference_weights2 @ value2
    torch.testing.assert_close(weights2, reference_weights2)
    torch.testing.assert_close(output2, reference_output2)

    print("Lesson 0006: all 12 checks passed.")
    print("device:", device)
    print("weights [B,S_query,S_key]:", weights)
    print("output [B,S,D]:", output)
    print("each weight row sums to one: True")
    print("future-token weights are zero: True")
    print("changing future V leaves earlier outputs unchanged: True")


if __name__ == "__main__":
    run_tests()
