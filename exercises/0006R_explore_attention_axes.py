"""Guided numeric trace for lesson 0006R."""

import math

import torch


def main() -> None:
    query = torch.tensor([[[1.0, 2.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])

    scores = query @ key.transpose(-2, -1) / math.sqrt(2)
    weights = torch.tensor([[[0.25, 0.75]]])
    output = weights @ value

    torch.testing.assert_close(
        scores,
        torch.tensor([[[1 / math.sqrt(2), 2 / math.sqrt(2)]]]),
    )
    torch.testing.assert_close(output, torch.tensor([[[5.0, 7.0]]]))
    print("score sums feature axis d; shape:", tuple(scores.shape), scores)
    print("output sums key axis k; shape:", tuple(output.shape), output)
    print("0006R guided axis trace passed.")


if __name__ == "__main__":
    main()
