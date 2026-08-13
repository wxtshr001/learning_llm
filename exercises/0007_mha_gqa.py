"""Independent assignment for lesson 0007: MHA/GQA and KV Cache cost."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def repeat_kv_for_query_heads(
    key_or_value: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    """Return logical [B,Nq,S,D] using contiguous Q-to-KV groups."""
    # TODO 1: validate rank 4, positive head counts, and Nq % Nkv == 0.
    # TODO 2: repeat each KV head group_size times along the head axis.
    raise NotImplementedError


def gqa_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return square full-sequence output [B,Nq,S,D] and weights [B,Nq,S,S]."""
    # TODO 3: validate rank, B/S/D/device/dtype compatibility and D > 0.
    # TODO 4: map K/V to Nq heads, then apply scaled causal attention.
    # TODO 5: softmax along the key axis and aggregate V.
    raise NotImplementedError


def merge_query_heads(head_output: torch.Tensor) -> torch.Tensor:
    """Return [B,S,Nq*D] from [B,Nq,S,D]."""
    # TODO 6: transpose head/sequence, make contiguous, and reshape.
    raise NotImplementedError


def kv_cache_bytes(
    batch: int,
    layers: int,
    num_kv_heads: int,
    cached_tokens: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Return K+V cache payload bytes, excluding metadata/alignment."""
    # TODO 7: validate every factor is positive and return the byte formula.
    raise NotImplementedError


def run_tests() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    query = torch.zeros(1, 4, 2, 1, device=device)
    key = torch.zeros(1, 2, 2, 1, device=device)
    value = torch.tensor([[[[1.0], [3.0]], [[10.0], [30.0]]]], device=device)

    output, weights = gqa_causal_attention(query, key, value)
    expected_output = torch.tensor(
        [[[[1.0], [2.0]], [[1.0], [2.0]], [[10.0], [20.0]], [[10.0], [20.0]]]],
        device=device,
    )
    torch.testing.assert_close(output, expected_output)
    assert weights.shape == (1, 4, 2, 2)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 4, 2, device=device))
    assert torch.count_nonzero(weights[..., 0, 1]).item() == 0

    repeated_value = repeat_kv_for_query_heads(value, 4)
    torch.testing.assert_close(repeated_value[:, 0], value[:, 0])
    torch.testing.assert_close(repeated_value[:, 1], value[:, 0])
    torch.testing.assert_close(repeated_value[:, 2], value[:, 1])
    torch.testing.assert_close(repeated_value[:, 3], value[:, 1])

    merged = merge_query_heads(output)
    expected_merged = torch.tensor(
        [[[1.0, 1.0, 10.0, 10.0], [2.0, 2.0, 20.0, 20.0]]],
        device=device,
    )
    torch.testing.assert_close(merged, expected_merged)

    # MHA boundary: Nq=Nkv. MQA boundary: Nkv=1.
    mha_kv = torch.arange(8.0, device=device).reshape(1, 4, 2, 1)
    torch.testing.assert_close(repeat_kv_for_query_heads(mha_kv, 4), mha_kv)
    mqa_kv = torch.tensor([[[[5.0], [9.0]]]], device=device)
    repeated_mqa = repeat_kv_for_query_heads(mqa_kv, 4)
    assert repeated_mqa.shape == (1, 4, 2, 1)
    for query_head in range(4):
        torch.testing.assert_close(repeated_mqa[:, query_head], mqa_kv[:, 0])

    try:
        repeat_kv_for_query_heads(torch.zeros(1, 3, 2, 1, device=device), 4)
    except ValueError:
        pass
    else:
        raise AssertionError("Nq not divisible by Nkv must be rejected")

    changed_value = value.clone()
    changed_value[:, :, 1, :] = 999.0
    changed_output, _ = gqa_causal_attention(query, key, changed_value)
    torch.testing.assert_close(changed_output[:, :, 0, :], output[:, :, 0, :])

    mha_bytes = kv_cache_bytes(1, 32, 32, 1024, 128, 2)
    gqa_bytes = kv_cache_bytes(1, 32, 8, 1024, 128, 2)
    assert mha_bytes == 512 * 1024**2
    assert gqa_bytes == 128 * 1024**2
    assert gqa_bytes * 4 == mha_bytes

    print("Lesson 0007: all GQA checks passed.")
    print("device:", device)
    print("Q head -> KV head mapping: [0, 0, 1, 1]")
    print("head output:", output)
    print("merged output:", merged)
    print("MHA/GQA KV Cache MiB:", mha_bytes // 1024**2, gqa_bytes // 1024**2)


if __name__ == "__main__":
    run_tests()
