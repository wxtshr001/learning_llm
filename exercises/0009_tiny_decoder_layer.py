"""Independent assignment for lesson 0009: assemble one Tiny Decoder Layer."""

from __future__ import annotations

import torch
from torch import nn

from decoder_components import (
    GatedFFN,
    TinyGQAAttention,
    TinyRMSNorm,
    fill_deterministic_parameters,
)


class TinyDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
    ) -> None:
        super().__init__()
        # TODO 1: create two different RMSNorm modules.
        # TODO 2: create TinyGQAAttention and GatedFFN from the supplied components.
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        # TODO 3: first pre-norm branch:
        # residual_0 = hidden_states
        # hidden_after_attention = residual_0 + self_attn(input_norm(hidden_states))
        # TODO 4: second pre-norm branch:
        # residual_1 = hidden_after_attention
        # output = residual_1 + mlp(post_attention_norm(hidden_after_attention))
        raise NotImplementedError


def run_tests() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = dict(
        hidden_size=4,
        num_query_heads=2,
        num_kv_heads=1,
        head_dim=2,
        intermediate_size=6,
    )
    layer = TinyDecoderLayer(**config).to(device)
    fill_deterministic_parameters(layer)
    hidden_states = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0], [4.0, 1.0, 0.0, 2.0]]],
        device=device,
    )
    original = hidden_states.clone()
    position_ids = torch.tensor([[0, 1, 2]], device=device)

    candidate = layer(hidden_states, position_ids)
    attention_input = layer.input_norm(hidden_states)
    attention_output = layer.self_attn(attention_input, position_ids)
    hidden_after_attention = hidden_states + attention_output
    ffn_input = layer.post_attention_norm(hidden_after_attention)
    ffn_output = layer.mlp(ffn_input)
    expected = hidden_after_attention + ffn_output

    assert layer.input_norm is not layer.post_attention_norm
    assert candidate.shape == hidden_states.shape == (1, 3, 4)
    torch.testing.assert_close(candidate, expected)
    torch.testing.assert_close(hidden_states, original)

    changed = hidden_states.clone()
    changed[:, 2, :] += 1000
    changed_output = layer(changed, position_ids)
    torch.testing.assert_close(candidate[:, :2], changed_output[:, :2])
    assert not torch.allclose(candidate[:, 2], changed_output[:, 2])

    zero_layer = TinyDecoderLayer(**config).to(device)
    with torch.no_grad():
        zero_layer.self_attn.o_proj.weight.zero_()
        zero_layer.mlp.down_proj.weight.zero_()
    torch.testing.assert_close(zero_layer(hidden_states, position_ids), hidden_states)

    invalid_position_ids = torch.tensor([[0, 1]], device=device)
    try:
        layer(hidden_states, invalid_position_ids)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched position_ids must raise ValueError")

    print("Lesson 0009: all Tiny Decoder Layer checks passed.")
    print("device:", device)
    print("input/output shape:", tuple(hidden_states.shape), tuple(candidate.shape))
    print("attention branch shape:", tuple(attention_output.shape))
    print("FFN branch shape:", tuple(ffn_output.shape))
    print("changing future token leaves earlier outputs unchanged: True")
    print("zero branches make the layer an identity: True")


if __name__ == "__main__":
    run_tests()
