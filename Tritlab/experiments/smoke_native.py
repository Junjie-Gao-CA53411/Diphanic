#!/usr/bin/env python
"""TritGPT 自检：从零模型前向/反向/训练/评测链路（无需 transformers）。

用法：python experiments/smoke_native.py
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tritlab.data import load_eval_chunks, make_loader      # noqa: E402
from tritlab.eval import evaluate_ppl                       # noqa: E402
from tritlab.models import TritGPT, TritGPTConfig           # noqa: E402
from tritlab.train import QATConfig, train_qat              # noqa: E402
from tritlab.utils import set_seed                          # noqa: E402


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    assert cond, name


def main():
    set_seed(0)
    tmpdir = tempfile.mkdtemp()
    train_bin = os.path.join(tmpdir, "train.bin")
    valid_bin = os.path.join(tmpdir, "valid.bin")
    rng = np.random.default_rng(0)
    rng.integers(0, 256, size=20_000, dtype=np.uint16).tofile(train_bin)
    rng.integers(0, 256, size=4_000, dtype=np.uint16).tofile(valid_bin)

    print("== 1. TritGPT 构造 ==")
    cfg = TritGPTConfig(vocab_size=256, dim=64, n_layers=2, n_heads=4,
                        n_kv_heads=2, ffn_dim=128, max_seq_len=64,
                        rope_theta=10000.0)
    model = TritGPT(cfg)
    n_params = model.count_params()
    print(f"     参数量 = {n_params/1e3:.1f}K (tiny 自检配置)")
    check("参数量 > 50K", n_params > 50_000)
    check("权重绑定生效", model.lm_head.weight is model.embed.weight)

    print("== 2. 前向 / 反向 ==")
    ids = torch.randint(0, 256, (2, 32))
    out = model(input_ids=ids, labels=ids)
    check("loss 有限", torch.isfinite(out.loss))
    out.loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    check("全部参数收到有限梯度", grads_ok)

    print("== 3. 三值开关（FP 基线路径） ==")
    cfg_fp = TritGPTConfig(**{**cfg.__dict__, "weight_mode": "none",
                              "quantize_input": False, "use_sub_norm": False,
                              "act": "gelu"})
    m_fp = TritGPT(cfg_fp)
    out_fp = m_fp(input_ids=ids, labels=ids)
    check("FP 基线也能跑", torch.isfinite(out_fp.loss))

    print("== 4. 训练 + 评测链路 ==")
    loader = make_loader(train_bin, seq_len=64, batch_size=8, num_workers=0)
    hist = train_qat(model, lambda: iter(loader),
                     QATConfig(max_steps=3, warmup_steps=2, log_every=1,
                               device="cpu", bf16=False, lr=1e-3,
                               schedule="cosine", min_lr_ratio=0.1))
    check("3 步训练 loss 有限", all(torch.isfinite(torch.tensor(hist["loss"]))))

    chunks = load_eval_chunks(valid_bin, seq_len=64, max_chunks=4)
    res = evaluate_ppl(model, chunks, device="cpu", max_len=64, stride=32)
    check("ppl 有限且为正", res["ppl"] > 0)
    print(f"     随机数据 ppl = {res['ppl']:.1f}（应 ≈ 256 = e^ln(256)，均匀分布）")

    print("== 5. 三态比例采集（L3-B 数据管道） ==")
    stats = [m.stats() for m in model.modules() if hasattr(m, "stats")]
    z = [s["frac_zero"] for s in stats]
    print(f"     层数={len(z)} 首层 0 比例={z[0]:.3f} 末层={z[-1]:.3f}")
    check("stats 可用", len(z) == 12)  # 2层 x (attn: q/k/v/o 4 + mlp: w13/w2 2)

    print("\nTritGPT 自检全部通过。")


if __name__ == "__main__":
    main()
