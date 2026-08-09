"""Guided shape classification for lesson 0005R; no PyTorch dependency."""

from __future__ import annotations


def linear_weight_shape(in_features: int, out_features: int) -> tuple[int, int]:
    """PyTorch Linear stores weight as [out_features, in_features]."""
    return out_features, in_features


def main() -> None:
    batch, sequence, hidden, intermediate = 2, 3, 4, 7

    mean_square_shape = (batch, sequence, 1)
    norm_weight_shape = (hidden,)
    norm_output_shape = (batch, sequence, hidden)
    print("mean_square [B,S,1]:", mean_square_shape)
    print("RMSNorm weight [H]:", norm_weight_shape)
    print("RMSNorm output [B,S,H]:", norm_output_shape)

    gate_weight = linear_weight_shape(hidden, intermediate)
    up_weight = linear_weight_shape(hidden, intermediate)
    down_weight = linear_weight_shape(intermediate, hidden)
    print("gate/up Linear(H,I) weight [I,H]:", gate_weight, up_weight)
    print("down Linear(I,H) weight [H,I]:", down_weight)

    assert gate_weight == (7, 4)
    assert up_weight == (7, 4)
    assert down_weight == (4, 7)
    print("Lesson 0005R guided shape checks passed.")


if __name__ == "__main__":
    main()
