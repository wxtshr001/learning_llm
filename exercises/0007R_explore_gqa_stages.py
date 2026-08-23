"""Guided trace for lesson 0007R: distinguish GQA tensor stages."""

from __future__ import annotations

import torch


def main() -> None:
    batch, sequence, hidden = 2, 3, 8
    num_query_heads, num_kv_heads, head_dim = 4, 2, 2

    hidden_states = torch.arange(
        batch * sequence * hidden, dtype=torch.float32
    ).reshape(batch, sequence, hidden)
    q_raw = hidden_states.clone()
    k_raw = hidden_states[..., : num_kv_heads * head_dim].clone()
    v_raw = k_raw + 1000

    query = q_raw.reshape(
        batch, sequence, num_query_heads, head_dim
    ).transpose(1, 2)
    key = k_raw.reshape(
        batch, sequence, num_kv_heads, head_dim
    ).transpose(1, 2)
    value = v_raw.reshape(
        batch, sequence, num_kv_heads, head_dim
    ).transpose(1, 2)
    group_size = num_query_heads // num_kv_heads
    logical_key = key.repeat_interleave(group_size, dim=1)
    logical_value = value.repeat_interleave(group_size, dim=1)
    scores_shape = (batch, num_query_heads, sequence, sequence)
    head_output = logical_value
    merged = head_output.transpose(1, 2).contiguous().reshape(
        batch, sequence, num_query_heads * head_dim
    )

    print("hidden rank/shape:", hidden_states.dim(), tuple(hidden_states.shape))
    print("q_raw rank/shape:", q_raw.dim(), tuple(q_raw.shape))
    print("k/v_raw rank/shape:", k_raw.dim(), tuple(k_raw.shape))
    print("Q split [B,Nq,S,D]:", tuple(query.shape))
    print("K/V stored [B,Nkv,S,D]:", tuple(key.shape), tuple(value.shape))
    print(
        "K/V logical [B,Nq,S,D]:",
        tuple(logical_key.shape),
        tuple(logical_value.shape),
    )
    print("scores/weights [B,Nq,S,S]:", scores_shape)
    print("head_output [B,Nq,S,D]:", tuple(head_output.shape))
    print("merge [B,S,Nq*D]:", tuple(merged.shape), "rank:", merged.dim())
    print("explicit Nq head axis: Q, logical K/V, scores/weights, head_output")
    print("no explicit head axis after merge or o_proj")
    print(
        "index check k_raw[1,2,3] == K[1,1,2,1]:",
        k_raw[1, 2, 3].item(),
        key[1, 1, 2, 1].item(),
    )

    assert tuple(k_raw.shape) == (2, 3, 4)
    assert tuple(key.shape) == (2, 2, 3, 2)
    assert tuple(logical_key.shape) == (2, 4, 3, 2)
    assert tuple(merged.shape) == (2, 3, 8)
    torch.testing.assert_close(k_raw[1, 2, 3], key[1, 1, 2, 1])
    print("Lesson 0007R guided stage trace passed.")


if __name__ == "__main__":
    main()
