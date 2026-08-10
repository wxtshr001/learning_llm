"""Pure-Python guided exploration of single-head causal attention."""

from __future__ import annotations

import math


def softmax(values: list[float]) -> list[float]:
    """Stable softmax; masked -inf entries become exact zero weights."""
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        raise ValueError("a softmax row must contain at least one allowed key")
    maximum = max(finite_values)
    exps = [0.0 if value == -math.inf else math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def causal_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> tuple[
    list[list[float]],
    list[list[bool]],
    list[list[float]],
    list[list[float]],
    list[list[float]],
]:
    sequence = len(queries)
    head_dim = len(queries[0])
    scale = math.sqrt(head_dim)
    scores: list[list[float]] = []
    future_mask: list[list[bool]] = []
    masked_scores: list[list[float]] = []
    weights: list[list[float]] = []
    outputs: list[list[float]] = []

    for query_index in range(sequence):
        score_row = [
            sum(queries[query_index][d] * keys[key_index][d] for d in range(head_dim)) / scale
            for key_index in range(sequence)
        ]
        mask_row = [key_index > query_index for key_index in range(sequence)]
        masked_score_row = [
            -math.inf if forbidden else score
            for score, forbidden in zip(score_row, mask_row)
        ]
        weight_row = softmax(masked_score_row)
        output_row = [
            sum(weight_row[key_index] * values[key_index][d] for key_index in range(sequence))
            for d in range(head_dim)
        ]
        scores.append(score_row)
        future_mask.append(mask_row)
        masked_scores.append(masked_score_row)
        weights.append(weight_row)
        outputs.append(output_row)

    return scores, future_mask, masked_scores, weights, outputs


def main() -> None:
    q = [[1.0], [1.0], [1.0]]
    k = [[1.0], [1.0], [1.0]]
    v = [[1.0], [3.0], [5.0]]
    scores, future_mask, masked_scores, weights, outputs = causal_attention(q, k, v)

    print("Q/K/V runtime shape [S,D]: (3, 1)")
    print("raw scores [S_query,S_key]:", scores)
    print("future mask [q,k] (True means forbidden):", future_mask)
    print("masked scores:", masked_scores)
    print("softmax weights along the key axis:", weights)
    print("attention output [S,D]:", outputs)
    print("weight row sums:", [sum(row) for row in weights])

    expected = [[1.0], [2.0], [3.0]]
    for actual_row, expected_row in zip(outputs, expected):
        assert math.isclose(actual_row[0], expected_row[0])

    changed_v = [[1.0], [3.0], [500.0]]
    _, _, _, _, changed_outputs = causal_attention(q, k, changed_v)
    assert changed_outputs[:2] == outputs[:2]
    assert changed_outputs[2] != outputs[2]
    print("changing future V[2] leaves query positions 0 and 1 unchanged: True")

    selective_q = [[1.0, 2.0]]
    selective_k = [[1.0, 0.0]]
    selective_v = [[7.0, 9.0]]
    selective_scores, _, _, _, _ = causal_attention(selective_q, selective_k, selective_v)
    assert math.isclose(selective_scores[0][0], 1 / math.sqrt(2))
    print("D=2 dot score [1,2]·[1,0]/sqrt(2):", selective_scores[0][0])
    print("Lesson 0006 guided exploration passed.")


if __name__ == "__main__":
    main()
