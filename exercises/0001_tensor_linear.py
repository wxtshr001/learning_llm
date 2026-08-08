"""Lesson 0001 exercise: implement a small Linear operation without libraries.

Run:
    python exercises/0001_tensor_linear.py

Only edit the three functions marked TODO.
"""

from __future__ import annotations


Matrix = list[list[float]]
Vector = list[float]


def infer_output_shape(x_shape: tuple[int, int], w_shape: tuple[int, int]) -> tuple[int, int]:
    """Return the shape of X @ W, or raise ValueError for incompatible shapes."""
    # TODO 1: validate the inner dimensions and return the output shape.
    if x_shape[1] != w_shape[0]:
        raise ValueError("x shape and w shape not match!")
    return (x_shape[0], w_shape[1])

def get_colume(input: Matrix, col_index: int) -> Vector:
    ret: Vector = []
    assert(col_index < len(input[0]))
    for row in input:
        ret.append(row[col_index])
    return ret

def multi_vector(left: Vector, right: Vector) -> float:
    assert(len(left) == len(right))
    return sum(l * r for l, r in zip(left, right))

def linear_element(x: Matrix, w: Matrix, b: Vector, i: int, j: int) -> float:
    """Compute one element Y[i][j] of Y = X @ W + b."""
    # TODO 2: sum over the shared dimension, then add the correct bias element.
    # x[i,:]*w[:,j]+b[i]
    return multi_vector(x[i], get_colume(w, j)) + b[j]


def linear_forward(x: Matrix, w: Matrix, b: Vector) -> Matrix:
    """Compute every element of Y = X @ W + b."""
    # TODO 3: build the complete output matrix using linear_element().
    rows, cols = len(x), len(b)
    ret: Matrix = [[linear_element(x, w, b, i, j) for j in range(cols)] for i in range(rows)]
    return ret


def assert_close(actual: Matrix, expected: Matrix, tolerance: float = 1e-9) -> None:
    assert len(actual) == len(expected), (actual, expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row), (actual, expected)
        for actual_value, expected_value in zip(actual_row, expected_row):
            assert abs(actual_value - expected_value) <= tolerance, (actual, expected)


def run_tests() -> None:
    assert infer_output_shape((4, 128), (128, 64)) == (4, 64)
    assert infer_output_shape((2, 3), (3, 1)) == (2, 1)

    try:
        infer_output_shape((2, 3), (4, 1))
    except ValueError:
        pass
    else:
        raise AssertionError("Incompatible shapes must raise ValueError")

    x = [[1.0, 2.0], [3.0, 4.0]]
    w = [[1.0, 0.0], [0.0, 2.0]]
    b = [10.0, 20.0]

    assert linear_element(x, w, b, 0, 0) == 11.0
    assert linear_element(x, w, b, 1, 1) == 28.0
    assert_close(linear_forward(x, w, b), [[11.0, 24.0], [13.0, 28.0]])

    x2 = [[1.0, -1.0, 2.0]]
    w2 = [[2.0, 1.0], [3.0, 0.0], [-1.0, 4.0]]
    b2 = [0.5, -0.5]
    assert_close(linear_forward(x2, w2, b2), [[-2.5, 8.5]])

    print("Lesson 0001: all 8 checks passed.")


if __name__ == "__main__":
    run_tests()
