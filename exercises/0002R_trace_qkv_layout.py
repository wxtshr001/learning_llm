"""Guided remediation for direct head-index and Q/K/V layout mapping.

Run from the repository root:
    python exercises/0002R_trace_qkv_layout.py

This is a complete guided example, not the independent retest.
"""

from __future__ import annotations


def hidden_index(head_index: int, head_dim: int, index_in_head: int) -> int:
    """Map (head index, index inside head) back to the hidden-axis index."""
    return head_index * head_dim + index_in_head


def projected_width(num_heads: int, head_dim: int) -> int:
    """Return the final projection width that will be split into heads."""
    return num_heads * head_dim


def head_shape(batch: int, sequence: int, num_heads: int, head_dim: int) -> tuple[int, ...]:
    """Return the layout after reshape and transpose: [B, N, S, D]."""
    return batch, num_heads, sequence, head_dim


def main() -> None:
    # Small direct-index example: B=1, S=2, H=6, N=2, D=3.
    index = hidden_index(head_index=1, head_dim=3, index_in_head=2)
    assert index == 5
    print("heads[0,1,1,2] maps to x[0,1,5]")

    # Q has two heads; K and V each have one head. Every head has D=4 features.
    batch = 1
    sequence = 3
    head_dim = 4
    query_heads = 2
    kv_heads = 1

    q_width = projected_width(query_heads, head_dim)
    kv_width = projected_width(kv_heads, head_dim)
    assert q_width == 8
    assert kv_width == 4

    print(f"Q projection: [{batch},{sequence},{q_width}]")
    print(f"K/V projection: [{batch},{sequence},{kv_width}]")
    print("Q heads [B,Nq,S,D]:", head_shape(batch, sequence, query_heads, head_dim))
    print("K/V heads [B,Nkv,S,D]:", head_shape(batch, sequence, kv_heads, head_dim))
    print("Lesson 0002R guided checks passed.")


if __name__ == "__main__":
    main()
