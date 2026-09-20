"""Guided trace for lesson 0009: assemble a pre-norm decoder layer."""

from __future__ import annotations

import torch

from decoder_components import (
    GatedFFN,
    TinyGQAAttention,
    TinyRMSNorm,
    fill_deterministic_parameters,
)


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch, sequence, hidden = 1, 3, 4
    num_query_heads, num_kv_heads, head_dim, intermediate = 2, 1, 2, 6
    hidden_states = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, 1.0, 3.0], [4.0, 1.0, 0.0, 2.0]]],
        device=device,
    )
    position_ids = torch.tensor([[0, 1, 2]], device=device)

    input_norm = TinyRMSNorm(hidden).to(device)
    attention = TinyGQAAttention(hidden, num_query_heads, num_kv_heads, head_dim).to(device)
    post_attention_norm = TinyRMSNorm(hidden).to(device)
    mlp = GatedFFN(hidden, intermediate).to(device)
    fill_deterministic_parameters(input_norm)
    fill_deterministic_parameters(attention)
    fill_deterministic_parameters(post_attention_norm)
    fill_deterministic_parameters(mlp)

    residual_0 = hidden_states
    attention_input = input_norm(residual_0)
    attention_trace = attention.trace(attention_input, position_ids)
    attention_output = attention_trace["output"]
    hidden_after_attention = residual_0 + attention_output

    residual_1 = hidden_after_attention
    ffn_input = post_attention_norm(residual_1)
    ffn_trace = mlp.trace(ffn_input)
    ffn_output = ffn_trace["output"]
    layer_output = residual_1 + ffn_output

    print("=== 1. Complete pre-norm layer flow ===")
    print("hidden_states / residual_0 [B,S,H]:", tuple(residual_0.shape))
    print("attention_input [B,S,H]:", tuple(attention_input.shape))
    print("Q split [B,Nq,S,D]:", tuple(attention_trace["query"].shape))
    print("K/V split [B,Nkv,S,D]:", tuple(attention_trace["key"].shape))
    print("weights [B,Nq,S,S]:", tuple(attention_trace["weights"].shape))
    print("attention_output [B,S,H]:", tuple(attention_output.shape))
    print("hidden_after_attention [B,S,H]:", tuple(hidden_after_attention.shape))
    print("ffn_input [B,S,H]:", tuple(ffn_input.shape))
    print("gate/up/mixed [B,S,I]:", tuple(ffn_trace["mixed"].shape))
    print("ffn_output [B,S,H]:", tuple(ffn_output.shape))
    print("layer_output [B,S,H]:", tuple(layer_output.shape))

    print("\n=== 2. One token through both residual additions ===")
    token = 1
    print("x0:", residual_0[0, token].round(decimals=4).tolist())
    print("attention branch:", attention_output[0, token].round(decimals=4).tolist())
    print("x1 = x0 + attention:", hidden_after_attention[0, token].round(decimals=4).tolist())
    print("ffn branch:", ffn_output[0, token].round(decimals=4).tolist())
    print("x2 = x1 + ffn:", layer_output[0, token].round(decimals=4).tolist())

    print("\n=== 3. Causal boundary ===")
    changed = hidden_states.clone()
    changed[:, 2, :] += 1000
    changed_attn_input = input_norm(changed)
    changed_attn_output = attention(changed_attn_input, position_ids)
    changed_x1 = changed + changed_attn_output
    changed_output = changed_x1 + mlp(post_attention_norm(changed_x1))
    print("changing future token 2 leaves outputs 0..1 unchanged:", bool(torch.allclose(layer_output[:, :2], changed_output[:, :2])))
    print("token 2 output changes:", bool(not torch.allclose(layer_output[:, 2], changed_output[:, 2])))

    assert residual_0.shape == attention_input.shape == attention_output.shape
    assert hidden_after_attention.shape == ffn_input.shape == ffn_output.shape == layer_output.shape
    assert attention_trace["query"].shape == (batch, num_query_heads, sequence, head_dim)
    assert attention_trace["key"].shape == (batch, num_kv_heads, sequence, head_dim)
    assert ffn_trace["mixed"].shape == (batch, sequence, intermediate)
    torch.testing.assert_close(hidden_after_attention, residual_0 + attention_output)
    torch.testing.assert_close(layer_output, hidden_after_attention + ffn_output)
    torch.testing.assert_close(layer_output[:, :2], changed_output[:, :2])
    assert not torch.allclose(layer_output[:, 2], changed_output[:, 2])
    print("Lesson 0009 guided decoder-layer trace passed.")


if __name__ == "__main__":
    main()
