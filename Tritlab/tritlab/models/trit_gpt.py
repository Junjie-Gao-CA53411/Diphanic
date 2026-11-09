"""TritGPT：原生三值 LM（从零训练）。

与改造路线（GPT-2 retrofit）的本质区别：这里的设计变量全是我们的——
激活（ReLU²）、sub-norm 位置、GQA 比例、RoPE θ、词表、权重绑定。
遵循 BitNet 训练配方：无参 RMSNorm + 大学习率 + 长 warmup + grad clip。

数据流（和微软 gpu/model.py 的 2B4T 对齐）：
    x -> RMSNorm -> int8 激活量化 -> 三值线性 -> ...（残差、sub-norm 同 2B4T）
embedding 与 lm_head 保持高精度（且默认绑定）。
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from ..layers.trit_linear import RMSNorm, TritLinear


@dataclass
class TritGPTConfig:
    vocab_size: int = 8192
    dim: int = 384
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_dim: int = 1024
    max_seq_len: int = 512
    rope_theta: float = 500000.0
    norm_eps: float = 1e-6
    act: str = "relu2"            # 'relu2'（BitNet a4.8 配方）| 'gelu'
    use_sub_norm: bool = True      # a4.8 的 attn / ffn 子层 norm
    tie_weights: bool = True
    weight_mode: str = "absmean"   # 三值化开关：'none' = FP 基线
    delta_coeff: float = 0.7
    quantize_input: bool = True
    initializer_std: float = 0.02
    grad_ckpt: bool = False        # 显存紧张时开（速度换显存）


class Rotary(nn.Module):
    """interleaved RoPE（q/k 共用一份 cos/sin 表）。"""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (T, D/2)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        # q, k: (B, H, T, D)
        T = q.shape[-2]
        cos = self.cos[:T][None, None, :, :]      # (1,1,T,D/2)
        sin = self.sin[:T][None, None, :, :]

        def rot(x):
            e, o = x[..., 0::2], x[..., 1::2]
            return torch.stack([e * cos - o * sin, e * sin + o * cos], dim=-1).flatten(-2)

        return rot(q), rot(k)


class Attention(nn.Module):
    def __init__(self, cfg: TritGPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        kw = dict(weight_mode=cfg.weight_mode, delta_coeff=cfg.delta_coeff,
                  quantize_input=cfg.quantize_input)
        self.q_proj = TritLinear(cfg.dim, cfg.n_heads * self.head_dim, **kw)
        self.k_proj = TritLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, **kw)
        self.v_proj = TritLinear(cfg.dim, cfg.n_kv_heads * self.head_dim, **kw)
        self.o_proj = TritLinear(cfg.n_heads * self.head_dim, cfg.dim, **kw)
        self.rotary = Rotary(self.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.sub_norm = RMSNorm(cfg.dim, cfg.norm_eps) if cfg.use_sub_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q, k)
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, -1)
        return self.o_proj(self.sub_norm(y))


class MLP(nn.Module):
    def __init__(self, cfg: TritGPTConfig):
        super().__init__()
        kw = dict(weight_mode=cfg.weight_mode, delta_coeff=cfg.delta_coeff,
                  quantize_input=cfg.quantize_input)
        self.w13 = TritLinear(cfg.dim, 2 * cfg.ffn_dim, **kw)
        self.w2 = TritLinear(cfg.ffn_dim, cfg.dim, **kw)
        self.sub_norm = (RMSNorm(cfg.ffn_dim, cfg.norm_eps)
                         if cfg.use_sub_norm else nn.Identity())
        self.act = cfg.act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        if self.act == "relu2":
            h = F.relu(a).square() * b       # BitNet 配方：ReLU^2 激活天然稀疏
        elif self.act == "gelu":
            h = F.gelu(a) * b
        else:
            raise ValueError(self.act)
        return self.w2(self.sub_norm(h))


class Block(nn.Module):
    def __init__(self, cfg: TritGPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x


class TritGPT(nn.Module):
    def __init__(self, cfg: TritGPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.weight   # 高精度 + 绑定（省一份大矩阵）

        self.apply(self._init)
        # 残差出口的投影用缩放初始化，稳定深网络
        for blk in self.blocks:
            for lin in (blk.attn.o_proj, blk.mlp.w2):
                nn.init.normal_(lin.weight, mean=0.0,
                                std=cfg.initializer_std / math.sqrt(2 * cfg.n_layers))

    def _init(self, m):
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.initializer_std)

    def forward(self, input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None):
        x = self.embed(input_ids)
        for blk in self.blocks:
            if self.cfg.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        logits = self.lm_head(self.norm(x))
        out = SimpleNamespace(logits=logits, loss=None)
        if labels is not None:
            out.loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1))
        return out

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
