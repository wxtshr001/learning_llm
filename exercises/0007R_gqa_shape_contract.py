"""Independent assignment for lesson 0007R: return every GQA stage shape."""

from __future__ import annotations


def gqa_stage_shapes(
    batch: int,
    sequence: int,
    hidden: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> dict[str, tuple[int, ...]]:
    """Return the semantic shape at every named GQA stage."""
    # TODO 1: reject non-positive inputs, Nq % Nkv != 0, and H != Nq*D.
    # TODO 2: return the shapes named in run_tests().
    if batch <= 0 or sequence <= 0 or hidden <= 0 or num_query_heads <= 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("invalid inputs")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("invalid match")
    if hidden != num_query_heads * head_dim:
        raise ValueError("invalid size")

    ret = {
        "hidden": (batch, sequence, hidden),
        "q_raw": (batch, sequence, num_query_heads * head_dim),
        "k_raw": (batch, sequence, num_kv_heads * head_dim),
        "v_raw": (batch, sequence, num_kv_heads * head_dim),
        "q_split": (batch, num_query_heads, sequence, head_dim),
        "k_split": (batch, num_kv_heads, sequence, head_dim),
        "v_split": (batch, num_kv_heads, sequence, head_dim),
        "logical_k": (batch, num_query_heads, sequence, head_dim),
        "logical_v": (batch, num_query_heads, sequence, head_dim),
        "scores": (batch, num_query_heads, sequence, sequence),
        "weights": (batch, num_query_heads, sequence, sequence),
        "head_output": (batch, num_query_heads, sequence, head_dim),
        "merge": (batch, sequence, num_query_heads * head_dim),
        "o_output": (batch, sequence, num_query_heads * head_dim),
    }
    return ret


def run_tests() -> None:
    shapes = gqa_stage_shapes(3, 5, 32, 8, 2, 4)
    expected = {
        "hidden": (3, 5, 32),
        "q_raw": (3, 5, 32),
        "k_raw": (3, 5, 8),
        "v_raw": (3, 5, 8),
        "q_split": (3, 8, 5, 4),
        "k_split": (3, 2, 5, 4),
        "v_split": (3, 2, 5, 4),
        "logical_k": (3, 8, 5, 4),
        "logical_v": (3, 8, 5, 4),
        "scores": (3, 8, 5, 5),
        "weights": (3, 8, 5, 5),
        "head_output": (3, 8, 5, 4),
        "merge": (3, 5, 32),
        "o_output": (3, 5, 32),
    }
    assert shapes == expected
    invalid = [
        (0, 5, 32, 8, 2, 4),
        (3, 5, 24, 8, 2, 4),
        (3, 5, 32, 7, 2, 4),
    ]
    for args in invalid:
        try:
            gqa_stage_shapes(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must reject invalid shape contract: {args}")

    print("Lesson 0007R: all GQA stage-shape checks passed.")
    for name, shape in shapes.items():
        print(f"{name:>12}: {shape}")


if __name__ == "__main__":
    run_tests()
