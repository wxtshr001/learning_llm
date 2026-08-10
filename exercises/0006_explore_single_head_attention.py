"""Pure-Python guided exploration of single-head causal attention."""

from __future__ import annotations

import math


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return [value / total for value in exps]


def causal_attention(
    queries: list[list[float]],
    keys: list[list[float]],
    values: list[list[float]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    sequence = len(queries)
    head_dim = len(queries[0])
    scale = math.sqrt(head_dim)
    scores: list[list[float]] = []
    weights: list[list[float]] = []
    outputs: list[list[float]] = []

    for query_index in range(sequence):
        score_row = [
            sum(queries[query_index][d] * keys[key_index][d] for d in range(head_dim)) / scale
            for key_index in range(sequence)
        ]
        allowed_weights = softmax(score_row[: query_index + 1])
        weight_row = allowed_weights + [0.0] * (sequence - query_index - 1)
        output_row = [
            sum(weight_row[key_index] * values[key_index][d] for key_index in range(sequence))
            for d in range(head_dim)
        ]
        scores.append(score_row)
        weights.append(weight_row)
        outputs.append(output_row)

    return scores, weights, outputs


def main() -> None:
    q = [[1.0], [1.0], [1.0]]
    k = [[1.0], [1.0], [1.0]]
    v = [[1.0], [3.0], [5.0]]
    scores, weights, outputs = causal_attention(q, k, v)

    print("Q/K/V runtime shape [S,D]: (3, 1)")
    print("raw scores [S_query,S_key]:", scores)
    print("causal weights:", weights)
    print("attention output [S,D]:", outputs)
    print("weight row sums:", [sum(row) for row in weights])

    expected = [[1.0], [2.0], [3.0]]
    for actual_row, expected_row in zip(outputs, expected):
        assert math.isclose(actual_row[0], expected_row[0])

    changed_v = [[1.0], [3.0], [500.0]]
    _, _, changed_outputs = causal_attention(q, k, changed_v)
    assert changed_outputs[:2] == outputs[:2]
    assert changed_outputs[2] != outputs[2]
    print("changing future V[2] leaves query positions 0 and 1 unchanged: True")
    print("Lesson 0006 guided exploration passed.")


if __name__ == "__main__":
    main()
