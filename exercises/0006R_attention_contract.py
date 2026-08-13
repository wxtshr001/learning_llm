"""Independent contract correction for lesson 0006R."""

import torch


def validate_head_dim(head_dim: int) -> None:
    # TODO 1: raise ValueError unless D is strictly positive.
    if head_dim <= 0:
        raise ValueError("dim is not positive!")


def make_mask_like(sequence_length: int, query: torch.Tensor) -> torch.Tensor:
    # TODO 2: return a bool [S,S] future mask on query.device.
    retv = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=query.device), diagonal=1)
    return retv


def run_tests() -> None:
    validate_head_dim(4)
    try:
        validate_head_dim(0)
    except ValueError:
        pass
    else:
        raise AssertionError("D=0 must be rejected")

    query = torch.ones(1, 3, 2)
    mask = make_mask_like(3, query)
    expected = torch.tensor(
        [[False, True, True], [False, False, True], [False, False, False]]
    )
    assert mask.dtype == torch.bool
    assert mask.device == query.device
    torch.testing.assert_close(mask, expected)
    print("0006R attention code contracts passed.")


if __name__ == "__main__":
    run_tests()
