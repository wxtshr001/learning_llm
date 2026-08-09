"""Guided parity exploration for RMSNorm and a gated FFN."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        normalized = x_float * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class GatedFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


def fill_weights(ffn: GatedFFN) -> None:
    """Install deterministic, non-symmetric weights for repeatable parity."""
    with torch.no_grad():
        ffn.gate_proj.weight.copy_(
            torch.arange(ffn.gate_proj.weight.numel(), dtype=torch.float32)
            .reshape_as(ffn.gate_proj.weight)
            .div(20)
            .sub(0.5)
        )
        ffn.up_proj.weight.copy_(
            torch.arange(ffn.up_proj.weight.numel(), dtype=torch.float32)
            .reshape_as(ffn.up_proj.weight)
            .flip(0)
            .div(25)
            .sub(0.3)
        )
        ffn.down_proj.weight.copy_(
            torch.arange(ffn.down_proj.weight.numel(), dtype=torch.float32)
            .reshape_as(ffn.down_proj.weight)
            .div(30)
            .sub(0.4)
        )


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size = 4
    intermediate_size = 6
    x = torch.tensor(
        [[[1.0, 2.0, -1.0, -2.0], [0.5, -0.5, 1.5, -1.5]]],
        device=device,
    )

    print("=== 1. RMSNorm shape path ===")
    custom_norm = TinyRMSNorm(hidden_size, eps=1e-6).to(device)
    reference_norm = nn.RMSNorm(hidden_size, eps=1e-6).to(device)
    with torch.no_grad():
        custom_norm.weight.copy_(torch.tensor([1.0, 2.0, 1.0, 0.5], device=device))
        reference_norm.weight.copy_(custom_norm.weight)
    norm_output = custom_norm(x)
    norm_reference = reference_norm(x)
    variance = x.float().pow(2).mean(dim=-1, keepdim=True)
    print("x shape [B,S,H]:", tuple(x.shape))
    print("variance shape [B,S,1]:", tuple(variance.shape))
    print("weight shape [H]:", tuple(custom_norm.weight.shape))
    print("output shape [B,S,H]:", tuple(norm_output.shape))
    print("first token output:", norm_output[0, 0].tolist())
    norm_error = (norm_reference - norm_output).abs().max().item()
    print("RMSNorm max absolute error:", norm_error)
    torch.testing.assert_close(norm_output, norm_reference, rtol=1e-5, atol=1e-6)

    print("\n=== 2. Gated FFN shape path ===")
    ffn = GatedFFN(hidden_size, intermediate_size).to(device)
    fill_weights(ffn)
    gate_raw = ffn.gate_proj(norm_output)
    gate = F.silu(gate_raw)
    up = ffn.up_proj(norm_output)
    mixed = gate * up
    output = ffn.down_proj(mixed)
    module_output = ffn(norm_output)
    print("gate_proj.weight [I,H]:", tuple(ffn.gate_proj.weight.shape))
    print("up_proj.weight [I,H]:", tuple(ffn.up_proj.weight.shape))
    print("down_proj.weight [H,I]:", tuple(ffn.down_proj.weight.shape))
    print("gate/up/mixed [B,S,I]:", tuple(gate.shape), tuple(up.shape), tuple(mixed.shape))
    print("output [B,S,H]:", tuple(output.shape))
    ffn_error = (module_output - output).abs().max().item()
    print("Gated FFN max absolute error:", ffn_error)
    torch.testing.assert_close(module_output, output, rtol=1e-6, atol=1e-7)

    print("\n=== 3. Backward evidence ===")
    loss = module_output.square().mean()
    loss.backward()
    for name, parameter in ffn.named_parameters():
        print(name, "grad shape=", tuple(parameter.grad.shape), "finite=", bool(torch.isfinite(parameter.grad).all()))
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    print("\nLesson 0005 guided exploration passed.")


if __name__ == "__main__":
    main()
