"""Independent assignment for lesson 0005. Complete TODOs and run this file."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        # TODO 1: register a learnable weight of ones with shape [H], and save eps.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 2: compute the mean square over the last axis with keepdim=True,
        # normalize with rsqrt(mean_square + eps), apply weight, preserve shape.
        # Use float32 for the normalization calculation and return input dtype.
        raise NotImplementedError


class GatedFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        # TODO 3: create bias-free gate_proj/up_proj H->I and down_proj I->H.
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 4: down_proj(silu(gate_proj(x)) * up_proj(x)).
        raise NotImplementedError


def _fill_weights(ffn: GatedFFN) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(ffn.parameters(), start=1):
            values = torch.arange(parameter.numel(), dtype=parameter.dtype, device=parameter.device)
            parameter.copy_(values.reshape_as(parameter).div(17 + index).sub(0.25))


def run_tests() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    b, s, h, intermediate = 2, 3, 4, 7
    x = torch.linspace(-2.0, 2.0, steps=b * s * h, device=device).reshape(b, s, h)

    custom_norm = TinyRMSNorm(h, eps=1e-6).to(device)
    reference_norm = nn.RMSNorm(h, eps=1e-6).to(device)
    with torch.no_grad():
        norm_weight = torch.tensor([1.0, 0.5, 1.5, 2.0], device=device)
        custom_norm.weight.copy_(norm_weight)
        reference_norm.weight.copy_(norm_weight)
    candidate_norm = custom_norm(x)
    reference = reference_norm(x)
    assert candidate_norm.shape == (b, s, h)
    assert custom_norm.weight.shape == (h,)
    torch.testing.assert_close(candidate_norm, reference, rtol=1e-5, atol=1e-6)
    zero_output = custom_norm(torch.zeros_like(x))
    assert torch.isfinite(zero_output).all()

    if device.type == "cuda":
        half_x = x.to(torch.float16)
        half_custom = TinyRMSNorm(h, eps=1e-6).to(device=device, dtype=torch.float16)
        half_reference = nn.RMSNorm(h, eps=1e-6).to(device=device, dtype=torch.float16)
        with torch.no_grad():
            half_reference.weight.copy_(half_custom.weight)
        half_candidate = half_custom(half_x)
        half_expected = half_reference(half_x)
        assert half_candidate.dtype == half_x.dtype
        torch.testing.assert_close(half_candidate, half_expected, rtol=1e-3, atol=1e-3)

    ffn = GatedFFN(h, intermediate).to(device)
    _fill_weights(ffn)
    assert ffn.gate_proj.weight.shape == (intermediate, h)
    assert ffn.up_proj.weight.shape == (intermediate, h)
    assert ffn.down_proj.weight.shape == (h, intermediate)
    assert ffn.gate_proj.bias is None and ffn.up_proj.bias is None and ffn.down_proj.bias is None

    module_output = ffn(candidate_norm)
    gate = F.silu(F.linear(candidate_norm, ffn.gate_proj.weight))
    up = F.linear(candidate_norm, ffn.up_proj.weight)
    explicit_output = F.linear(gate * up, ffn.down_proj.weight)
    assert module_output.shape == (b, s, h)
    torch.testing.assert_close(module_output, explicit_output, rtol=1e-6, atol=1e-7)

    loss = module_output.square().mean()
    loss.backward()
    for parameter in ffn.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print("Lesson 0005: all checks passed.")
    print("device:", device)
    print("RMSNorm output shape:", tuple(candidate_norm.shape))
    print("RMSNorm max absolute error:", (candidate_norm - reference).abs().max().item())
    if device.type == "cuda":
        print("RMSNorm float16 output dtype preserved:", half_candidate.dtype)
    print("gate/up shape [B,S,I]:", tuple(gate.shape), tuple(up.shape))
    print("Gated FFN output shape:", tuple(module_output.shape))
    print("Gated FFN max absolute error:", (module_output - explicit_output).abs().max().item())
    print("all FFN parameter gradients finite: True")


if __name__ == "__main__":
    run_tests()
