"""Provided building blocks for lesson 0009.

The learner has already implemented these algorithms in lessons 0005-0008.
Lesson 0009 reuses them so the new task stays focused on decoder-layer assembly.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class TinyRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0 or eps <= 0:
            raise ValueError("hidden_size and eps must be positive")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    head_dim = x.shape[-1]
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be positive and even")
    half = head_dim // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def build_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids.ndim != 2 or head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("position_ids must be [B,S] and head_dim must be positive/even")
    inv_freq = 1.0 / (
        10_000.0
        ** (torch.arange(0, head_dim, 2, device=position_ids.device).float() / head_dim)
    )
    angles = position_ids[..., None].float() * inv_freq
    embedding = torch.cat((angles, angles), dim=-1)
    return embedding.cos().to(dtype), embedding.sin().to(dtype)


def repeat_kv(x: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    num_kv_heads = x.shape[1]
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError("Nq must be divisible by positive Nkv")
    return x.repeat_interleave(num_query_heads // num_kv_heads, dim=1)


class TinyGQAAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_query_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if hidden_size != num_query_heads * head_dim:
            raise ValueError("H must equal Nq*D in this tiny lesson")
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("D must be positive and even for RoPE")
        if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
            raise ValueError("Nq must be divisible by positive Nkv")
        self.hidden_size = hidden_size
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_query_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_query_heads * head_dim, hidden_size, bias=False)

    def trace(self, x: torch.Tensor, position_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("hidden_states must be rank 3 [B,S,H]")
        batch, sequence, hidden = x.shape
        if hidden != self.hidden_size or position_ids.shape != (batch, sequence):
            raise ValueError("hidden_states and position_ids shapes do not match config")

        q_raw = self.q_proj(x)
        k_raw = self.k_proj(x)
        v_raw = self.v_proj(x)
        query = q_raw.view(batch, sequence, self.num_query_heads, self.head_dim).transpose(1, 2)
        key = k_raw.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = v_raw.view(batch, sequence, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = build_rope_cos_sin(position_ids, self.head_dim, x.dtype)
        cos_heads = cos.unsqueeze(1)
        sin_heads = sin.unsqueeze(1)
        query_rope = query * cos_heads + rotate_half(query) * sin_heads
        key_rope = key * cos_heads + rotate_half(key) * sin_heads

        logical_key = repeat_kv(key_rope, self.num_query_heads)
        logical_value = repeat_kv(value, self.num_query_heads)
        scores = query_rope @ logical_key.transpose(-2, -1) / math.sqrt(self.head_dim)
        mask = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=-1)
        head_output = weights @ logical_value
        merged = head_output.transpose(1, 2).contiguous().view(batch, sequence, -1)
        output = self.o_proj(merged)
        return {
            "q_raw": q_raw,
            "k_raw": k_raw,
            "v_raw": v_raw,
            "query": query,
            "key": key,
            "value": value,
            "cos": cos,
            "sin": sin,
            "query_rope": query_rope,
            "key_rope": key_rope,
            "logical_key": logical_key,
            "logical_value": logical_value,
            "scores": scores,
            "weights": weights,
            "head_output": head_output,
            "merged": merged,
            "output": output,
        }

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        return self.trace(x, position_ids)["output"]


class GatedFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        if hidden_size <= 0 or intermediate_size <= 0:
            raise ValueError("H and I must be positive")
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def trace(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        gate_raw = self.gate_proj(x)
        gate = F.silu(gate_raw)
        up = self.up_proj(x)
        mixed = gate * up
        output = self.down_proj(mixed)
        return {
            "gate_raw": gate_raw,
            "gate": gate,
            "up": up,
            "mixed": mixed,
            "output": output,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trace(x)["output"]


def fill_deterministic_parameters(module: nn.Module) -> None:
    """Fill parameters with small reproducible values for lesson traces."""
    with torch.no_grad():
        for index, parameter in enumerate(module.parameters(), start=1):
            values = torch.arange(
                1,
                parameter.numel() + 1,
                device=parameter.device,
                dtype=parameter.dtype,
            )
            parameter.copy_(values.reshape_as(parameter).div(50 + 7 * index))
