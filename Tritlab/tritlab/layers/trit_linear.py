"""BitNet 风格层封装：无参 RMSNorm -> int8 激活量化 -> 三值线性。

替换 nn.Linear 的两个开关（EX-001 逐层消融就靠它们）：
- weight_mode: 'none' | 'absmean' | 'perchannel' | 'learnable'
- quantize_input: bool（是否对输入做 int8 absmax 量化）
"""
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..quant.ternary import (
    Int8Absmax,
    TernaryLearnableDelta,
    ternarize_absmean,
    ternarize_perchannel,
)


class RMSNorm(nn.Module):
    """无参 RMSNorm（BitNet 要求归一化无 gamma/beta）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(var + self.eps)).type_as(x)


class TritLinear(nn.Module):
    """接口与 nn.Linear 一致，weight shape = (out_features, in_features)。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_mode: str = "absmean",
        delta_coeff: float = 0.7,
        quantize_input: bool = True,
        norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mode = weight_mode
        self.delta_coeff = delta_coeff

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self.norm = RMSNorm(in_features, eps=norm_eps)
        self.input_quant = Int8Absmax() if quantize_input else nn.Identity()
        self.delta_learn = (
            TernaryLearnableDelta(delta_coeff) if weight_mode == "learnable" else None
        )

    def _quant_weight(self) -> torch.Tensor:
        if self.weight_mode == "none":
            return self.weight
        if self.weight_mode == "absmean":
            return ternarize_absmean(self.weight, self.delta_coeff)
        if self.weight_mode == "perchannel":
            return ternarize_perchannel(self.weight, self.delta_coeff)
        if self.weight_mode == "learnable":
            return self.delta_learn(self.weight)
        raise ValueError(f"unknown weight_mode: {self.weight_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.input_quant(x)
        return F.linear(x, self._quant_weight(), self.bias)

    @torch.no_grad()
    def stats(self) -> Dict[str, float]:
        """量化后权重的 {+1,0,-1} 比例与 Delta（L3-B 课题的数据来源）。"""
        w = self.weight
        delta = self.delta_coeff * w.abs().mean().item()
        t = (w / delta).round().clamp(-1, 1)
        return {
            "delta": delta,
            "frac_plus": (t == 1).float().mean().item(),
            "frac_zero": (t == 0).float().mean().item(),
            "frac_minus": (t == -1).float().mean().item(),
        }


class TritQKV(nn.Module):
    """融合 QKV 投影：内部三个 TritLinear，输出 cat([q,k,v], -1)。

    用于改造 GPT-2 的 c_attn（Conv1D），逐部件控制是否三值化。
    """

    def __init__(
        self,
        dim: int,
        modes: Dict[str, str],  # {'q': 'absmean', 'k': 'none', ...}
        bias: bool = True,
        quantize_input: bool = True,
        delta_coeff: float = 0.7,
    ):
        super().__init__()
        self.q_proj = TritLinear(dim, dim, bias, modes.get("q", "none"),
                                 delta_coeff, quantize_input)
        self.k_proj = TritLinear(dim, dim, bias, modes.get("k", "none"),
                                 delta_coeff, quantize_input)
        self.v_proj = TritLinear(dim, dim, bias, modes.get("v", "none"),
                                 delta_coeff, quantize_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.q_proj(x), self.k_proj(x), self.v_proj(x)], dim=-1
        )
