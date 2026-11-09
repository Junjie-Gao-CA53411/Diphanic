"""量化器家族：三值权重（STE）+ int8 激活（absmax）。

统一约定
--------
- 所有函数/模块输入输出 shape 不变，可直接替换 nn.Linear 里的权重。
- 训练期（QAT）：返回的是"量化-反量化"后的浮点值，梯度经 STE 回传。
- 推理期：把这里的逻辑替换成整数 kernel（bitnet.cpp 的 I2_S / gpu 版 int2 打包）。
"""
from typing import Tuple

import torch
from torch import nn


class STE(torch.autograd.Function):
    """直通估计器：前向用量化值，反向梯度原样传给潜权重。"""

    @staticmethod
    def forward(ctx, w_q: torch.Tensor, w_ref: torch.Tensor) -> torch.Tensor:
        return w_q

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        return grad_out, None  # 第二个返回值对应 w_ref，不需要梯度


def ste(w_q: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return STE.apply(w_q, w)


def ternarize_absmean(w: torch.Tensor, delta_coeff: float = 0.7) -> torch.Tensor:
    """TWN / BitNet b1.58 标准三值化：Delta = delta_coeff * mean|W|。

    返回量化-反量化后的权重（值为 -Delta/0/+Delta），梯度经 STE。
    """
    delta = delta_coeff * w.abs().mean()
    w_q = (w / delta).round().clamp(-1, 1) * delta
    return ste(w_q, w)


def ternarize_perchannel(w: torch.Tensor, delta_coeff: float = 0.7) -> torch.Tensor:
    """逐输出通道三值化：每行一个 Delta。精度更高，每行多存一个 scale。"""
    delta = delta_coeff * w.abs().mean(dim=1, keepdim=True)
    w_q = (w / delta).round().clamp(-1, 1) * delta
    return ste(w_q, w)


class TernaryLearnableDelta(nn.Module):
    """L1 课题 A：可学习 Delta。

    delta = delta_raw * mean|W|.detach()  —— 可学习的是相对系数（量纲稳定），
    训练后可以直接看每层学到的系数，回答"最优 Delta 是否随深度变化"。
    """

    def __init__(self, init_coeff: float = 0.7):
        super().__init__()
        self.delta_raw = nn.Parameter(torch.tensor(float(init_coeff)))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        delta = self.delta_raw * w.abs().mean().detach()
        w_q = (w / delta).round().clamp(-1, 1) * delta
        return ste(w_q, w)

    @torch.no_grad()
    def current_delta(self, w: torch.Tensor) -> torch.Tensor:
        return self.delta_raw * w.abs().mean()


class Int8Absmax(nn.Module):
    """激活量化：逐 token（最后一维）absmax 定标到 int8，再反量化。

    对应 BitNet 的 8-bit 激活；第二课 gamma 扫描的完整版。
    forward 里加可学习 gamma 即可复现你 hidden=256 时的 gamma 实验。
    """

    def __init__(self, num_bits: int = 8):
        super().__init__()
        self.qmax = 2 ** (num_bits - 1) - 1  # 127

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.qmax / x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_q = (x * s).round().clamp(-self.qmax - 1, self.qmax) / s
        return ste(x_q, x)


# tritlab/quant/ternary.py 追加
class TernaryLowRank(nn.Module):
    def __init__(self, rank=8, delta_coeff=0.7):
        super().__init__()
        self.rank, self.delta_coeff = rank, delta_coeff
    def forward(self, w):
        delta = self.delta_coeff * w.abs().mean().detach()
        t = (w / delta).round().clamp(-1, 1)
        R = w - t * delta                      # 残差
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        r = min(self.rank, S.shape[0])
        R_low = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
        return ste(t * delta + 0.1 * R_low, w)  # α=0.1 也可学习
    