#!/usr/bin/env python
"""骨架自检：不需要 transformers/datasets，本地 CPU 几十秒跑完。

用法：
  python experiments/smoke_test.py            # 自检
  python experiments/smoke_test.py --sweep    # L1-B 预实验：per-tensor vs per-channel
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tritlab.layers import TritLinear, TritQKV, RMSNorm        # noqa: E402
from tritlab.quant import (Int8Absmax, TernaryLearnableDelta,   # noqa: E402
                           ternarize_absmean, ternarize_perchannel)
from tritlab.train import QATConfig, train_qat                  # noqa: E402
from tritlab.eval import evaluate_ppl                           # noqa: E402
from tritlab.utils import ExperimentLogger, plot_sweep, set_seed  # noqa: E402


class TinyLM(torch.nn.Module):
    """GPT-2 式结构（ln->attn->res; ln->mlp->res），用于无依赖自检。"""

    def __init__(self, vocab=256, dim=64, n_inner=128, n_head=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.ln1 = RMSNorm(dim)
        self.qkv = TritQKV(dim, {"q": "absmean", "k": "absmean", "v": "absmean"})
        self.o = TritLinear(dim, dim, weight_mode="absmean")
        self.ln2 = RMSNorm(dim)
        self.fc = TritLinear(dim, n_inner, weight_mode="absmean")
        self.proj = TritLinear(n_inner, dim, weight_mode="absmean")
        self.head = torch.nn.Linear(dim, vocab, bias=False)
        self.n_head = n_head
        self.dim = dim

    def forward(self, input_ids, labels=None):
        h = self.emb(input_ids)
        B, T, D = h.shape
        qkv = self.qkv(self.ln1(h)).view(B, T, 3, self.n_head, D // self.n_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        att = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        att = att.transpose(1, 2).reshape(B, T, D)
        h = h + self.o(att)
        h = h + self.proj(self.fc(self.ln2(h)))
        logits = self.head(h)
        out = type("O", (), {})()
        out.logits = logits
        out.loss = None
        if labels is not None:
            out.loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1))
        return out


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    set_seed(0)

    print("== 1. 量化器 ==")
    w = torch.randn(64, 32, requires_grad=True)
    wq = ternarize_absmean(w, 0.7)
    uniq = wq.detach().unique()
    check("absmean 输出只有 {-d,0,+d}", len(uniq) <= 3)
    wpc = ternarize_perchannel(w.detach(), 0.7)
    check("perchannel shape 不变", wpc.shape == w.shape)
    check("STE: 梯度流到潜权重", wq.requires_grad and torch.autograd.grad(wq.sum(), w, retain_graph=False)[0] is not None)

    ld = TernaryLearnableDelta(0.7)
    out = ld(w)
    g = torch.autograd.grad(out.sum(), ld.delta_raw, retain_graph=True)[0]
    check("可学习 Delta 能收到梯度", g is not None and torch.isfinite(g))

    iq = Int8Absmax()
    x = torch.randn(4, 16)
    xq = iq(x)
    check("int8 absmax 值域被夹", xq.abs().max() <= x.abs().max() * 1.01)

    print("== 2. 层 ==")
    lin = TritLinear(32, 16, weight_mode="absmean")
    y = lin(torch.randn(2, 5, 32))
    check("TritLinear 输出 shape", y.shape == (2, 5, 16))
    st = lin.stats()
    print(f"     权重三态比例: +{st['frac_plus']:.2f} 0:{st['frac_zero']:.2f} -{st['frac_minus']:.2f} (L3-B 的数据)")
    qkv = TritQKV(32, {"q": "absmean", "k": "none", "v": "none"})
    check("TritQKV 输出 3*dim", qkv(torch.randn(2, 32)).shape == (2, 96))

    print("== 3. 训练 + 评测 ==")
    model = TinyLM()
    ids = [torch.randint(0, 256, (64,)) for _ in range(16)]

    def batches():
        yield from ({"input_ids": torch.stack(ids[i:i + 4])} for i in range(0, 16, 4))

    hist = train_qat(model, batches, QATConfig(max_steps=3, warmup_steps=2,
                                               log_every=1, device="cpu", bf16=False))
    check("训练循环跑通", len(hist["loss"]) == 3 and all(torch.isfinite(torch.tensor(hist["loss"]))))

    res = evaluate_ppl(model, ids[:4], device="cpu", max_len=32, stride=16)
    check("ppl 有限", res["ppl"] == res["ppl"] and res["ppl"] > 0)
    print(f"     随机权重 tiny LM ppl = {res['ppl']:.1f}（应约等于 ln(256)≈5.5×e... 即 ~256 附近）")

    print("== 4. 登记簿 ==")
    import tempfile, os
    tmp_log = os.path.join(tempfile.mkdtemp(), "smoke.jsonl")
    logger = ExperimentLogger(tmp_log)
    logger.log("smoke", {"a": 1}, {"ppl": res["ppl"]})
    check("jsonl 登记", len(logger.load()) == 1)

    if args.sweep:
        print("== L1-B 预实验: per-tensor vs per-channel (3 步训练 loss) ==")
        losses = {"per-tensor": [], "per-channel": []}
        for name, mode in [("per-tensor", "absmean"), ("per-channel", "perchannel")]:
            set_seed(0)
            m = TinyLM()
            for mod in m.modules():
                if isinstance(mod, TritLinear):
                    mod.weight_mode = mode
            h = train_qat(m, batches, QATConfig(max_steps=3, warmup_steps=2,
                                                log_every=99, device="cpu", bf16=False))
            losses[name] = [round(v, 3) for v in h["loss"]]
        print("     3 步 loss:", losses)
        plot_sweep([1, 2, 3], losses, "step", "loss", "L1-B pre-sweep",
                   save_path="/tmp/l1b_sweep.png")

    print("\n全部通过。骨架可用。")


if __name__ == "__main__":
    main()
