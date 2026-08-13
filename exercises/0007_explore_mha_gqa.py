"""Guided PyTorch exploration of MHA/GQA head mapping and cache cost."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def repeat_kv(key_or_value: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    """Map [B,Nkv,S,D] to logical [B,Nq,S,D] by contiguous groups."""
    num_kv_heads = key_or_value.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Nq must be divisible by Nkv")
    group_size = num_query_heads // num_kv_heads
    return key_or_value.repeat_interleave(group_size, dim=1)


def explicit_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return repeated K/V, weights, and output for square full-sequence GQA."""
    _, num_query_heads, query_length, head_dim = query.shape
    key_length = key.shape[2]
    key_for_compute = repeat_kv(key, num_query_heads)
    value_for_compute = repeat_kv(value, num_query_heads)
    scores = query @ key_for_compute.transpose(-2, -1) / math.sqrt(head_dim)
    future_mask = torch.triu(
        torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=1,
    )
    weights = F.softmax(scores.masked_fill(future_mask, float("-inf")), dim=-1)
    output = weights @ value_for_compute
    return key_for_compute, value_for_compute, weights, output


def merge_heads(head_output: torch.Tensor) -> torch.Tensor:
    """[B,Nq,S,D] -> [B,S,Nq*D]."""
    batch, num_heads, sequence, head_dim = head_output.shape
    return head_output.transpose(1, 2).contiguous().reshape(
        batch, sequence, num_heads * head_dim
    )


def kv_cache_bytes(
    batch: int,
    layers: int,
    num_kv_heads: int,
    cached_tokens: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    return (
        2
        * batch
        * layers
        * num_kv_heads
        * cached_tokens
        * head_dim
        * bytes_per_element
    )


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # B=1, Nq=4, Nkv=2, S=2, D=1. Zero Q/K gives exact uniform weights.
    query = torch.zeros(1, 4, 2, 1, device=device)
    key = torch.zeros(1, 2, 2, 1, device=device)
    value = torch.tensor([[[[1.0], [3.0]], [[10.0], [30.0]]]], device=device)

    key_compute, value_compute, weights, head_output = explicit_gqa(
        query, key, value
    )
    merged = merge_heads(head_output)

    print("device:", device)
    print("Q shape [B,Nq,S,D]:", tuple(query.shape))
    print("K/V stored shape [B,Nkv,S,D]:", tuple(key.shape))
    print("group_size Nq/Nkv:", query.shape[1] // key.shape[1])
    print("Q head -> KV head mapping: [0, 0, 1, 1]")
    print("logical V used by Q heads:\n", value_compute)
    print("weights [B,Nq,Sq,Sk]:\n", weights)
    print("head output [B,Nq,S,D]:\n", head_output)
    print("merged [B,S,Nq*D]:\n", merged)

    expected_head_output = torch.tensor(
        [[[[1.0], [2.0]], [[1.0], [2.0]], [[10.0], [20.0]], [[10.0], [20.0]]]],
        device=device,
    )
    expected_merged = torch.tensor(
        [[[1.0, 1.0, 10.0, 10.0], [2.0, 2.0, 20.0, 20.0]]],
        device=device,
    )
    torch.testing.assert_close(head_output, expected_head_output)
    torch.testing.assert_close(merged, expected_merged)
    assert key_compute.shape == query.shape

    mha_bytes = kv_cache_bytes(1, 32, 32, 1024, 128, 2)
    gqa_bytes = kv_cache_bytes(1, 32, 8, 1024, 128, 2)
    print("MHA KV Cache MiB:", mha_bytes // (1024 * 1024))
    print("GQA KV Cache MiB:", gqa_bytes // (1024 * 1024))
    print("saved MiB:", (mha_bytes - gqa_bytes) // (1024 * 1024))
    assert (mha_bytes, gqa_bytes) == (512 * 1024**2, 128 * 1024**2)
    print("Lesson 0007 guided GQA exploration passed.")


if __name__ == "__main__":
    main()
