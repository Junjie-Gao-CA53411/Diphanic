#!/usr/bin/env python
"""demo_first_model.py —— 端到端：你自己的第一个三值模型（30 分钟版）。

不需要 GPU / transformers / 任何下载。喂给它任意 .txt（建议 ≥2 万字符），
流程：字符级分词 -> 从零训练小 TritGPT（三值权重）-> 评测 ppl -> 打印三态比例 -> 续写。

用法：
  python experiments/demo_first_model.py --text data/my_corpus.txt --steps 500
  python experiments/demo_first_model.py --text 诗经.txt --steps 800 --dim 96 --layers 3
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tritlab.eval import evaluate_ppl                 # noqa: E402
from tritlab.models import TritGPT, TritGPTConfig     # noqa: E402
from tritlab.train import QATConfig, train_qat        # noqa: E402
from tritlab.utils import set_seed                    # noqa: E402


def char_tokenize(text: str):
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return ids, stoi, itos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="语料 .txt 路径（utf-8）")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--gen", type=int, default=120, help="续写字符数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    text = Path(args.text).read_text(encoding="utf-8")
    ids, stoi, itos = char_tokenize(text)
    V = len(stoi)
    print(f"[data] 语料 {len(ids)} 字符, 字符级词表 {V}")

    cfg = TritGPTConfig(vocab_size=V, dim=args.dim, n_layers=args.layers,
                        n_heads=4, n_kv_heads=2, ffn_dim=2 * args.dim,
                        max_seq_len=args.seq, rope_theta=10000.0)
    model = TritGPT(cfg)
    n_tri = sum(m.weight.numel() for m in model.modules()
                if m.__class__.__name__ == "TritLinear")
    print(f"[model] TritGPT {model.count_params()/1e3:.0f}K params "
          f"(其中 {n_tri/1e3:.0f}K 是三值权重, 目标 1.58 bit/权重)")

    def batches():  # 无限随机裁剪流，max_steps 处自动停
        while True:
            ix = torch.randint(0, len(ids) - args.seq, (16,))
            yield {"input_ids": torch.stack([ids[i:i + args.seq] for i in ix])}

    print("[train] 开始 QAT（CPU 即可）...")
    train_qat(model, batches, QATConfig(
        lr=args.lr, max_steps=args.steps, warmup_steps=max(args.steps // 10, 1),
        schedule="cosine", min_lr_ratio=0.05,
        log_every=max(args.steps // 10, 1), device="cpu", bf16=False))

    n_chunks = min(20, (len(ids) - args.seq) // args.seq)
    eval_ids = [ids[i * args.seq:(i + 1) * args.seq] for i in range(n_chunks)]
    res = evaluate_ppl(model, eval_ids, device="cpu",
                       max_len=args.seq, stride=args.seq // 2)
    print(f"[eval] 字符级 ppl = {res['ppl']:.2f} "
          f"(随机基线 ≈ {V}; 越小说明模型真的学到了结构)")

    stats = [m.stats() for m in model.modules() if hasattr(m, "stats")]
    z = sum(s["frac_zero"] for s in stats) / len(stats)
    print(f"[stats] 三值权重中 0 的平均比例 = {z:.3f}  ← 训练自发涌现的稀疏度")

    model.eval()
    prompt = text[:min(16, len(text))]
    ctx = torch.tensor([stoi[c] for c in prompt], dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        for _ in range(args.gen):
            window = ctx[:, -args.seq:]
            logits = model(input_ids=window).logits[0, -1] / 0.8  # temperature=0.8
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1)
            ctx = torch.cat([ctx, nxt.unsqueeze(0)], dim=1)
    print("[gen] 三值模型从语料开头续写：")
    print("----")
    print("".join(itos[i] for i in ctx[0].tolist()))
    print("----")
    print("提示：loss 应从 ~ln(词表) 持续下降；续写应出现语料里的重复模式。")


if __name__ == "__main__":
    main()
