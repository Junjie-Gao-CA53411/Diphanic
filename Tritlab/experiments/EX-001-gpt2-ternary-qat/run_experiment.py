#!/usr/bin/env python
"""EX-001 runner：按 config.yaml 跑 组 × 种子，登记到 logs/experiments.jsonl。

用法：
  python run_experiment.py                       # 全部组 × 全部种子
  python run_experiment.py --group b_all --seed 0 --max-steps 20   # 烟雾测试
"""
import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

# 让脚本无论从哪运行都能找到 tritlab 包
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tritlab.layers import TritLinear, TritQKV            # noqa: E402
from tritlab.train import QATConfig, train_qat            # noqa: E402
from tritlab.eval import evaluate_ppl                     # noqa: E402
from tritlab.utils import ExperimentLogger, set_seed      # noqa: E402

PARTS = ("q", "k", "v", "o", "fc", "proj")


def retrofit_gpt2(model, quantize, keep_fp, quant_cfg):
    """把 GPT-2 指定部件换成 TritLinear/TritQKV，并搬运权重。

    不在 quantize 集合内的部件保持原样（原汁原味的 FP 对照）。
    """
    quantize = set(quantize) - set(keep_fp)
    cfg = model.config
    nx, n_inner = cfg.n_embd, cfg.n_inner

    def mode(p):
        return quant_cfg["weight_mode"] if p in quantize else "none"

    for block in model.transformer.h:
        attn, mlp = block.attn, block.mlp

        if {"q", "k", "v"} & quantize:
            c = attn.c_attn  # Conv1D: weight (nx, 3*nx), bias (3*nx,)
            qkv = TritQKV(nx, {p: mode(p) for p in "qkv"},
                          bias=True,
                          quantize_input=quant_cfg["quantize_input"],
                          delta_coeff=quant_cfg["delta_coeff"])
            with torch.no_grad():
                for i, p in enumerate("qkv"):
                    lin = getattr(qkv, f"{p}_proj")
                    sl = slice(i * nx, (i + 1) * nx)
                    lin.weight.copy_(c.weight[:, sl].T.contiguous())
                    lin.bias.copy_(c.bias[sl])
            attn.c_attn = qkv

        if "o" in quantize:
            c = attn.c_proj
            lin = TritLinear(nx, nx, bias=True, weight_mode=mode("o"),
                             delta_coeff=quant_cfg["delta_coeff"],
                             quantize_input=quant_cfg["quantize_input"])
            with torch.no_grad():
                lin.weight.copy_(c.weight.T.contiguous())
                lin.bias.copy_(c.bias)
            attn.c_proj = lin

        if "fc" in quantize:
            c = mlp.c_fc
            lin = TritLinear(nx, n_inner, bias=True, weight_mode=mode("fc"),
                             delta_coeff=quant_cfg["delta_coeff"],
                             quantize_input=quant_cfg["quantize_input"])
            with torch.no_grad():
                lin.weight.copy_(c.weight.T.contiguous())
                lin.bias.copy_(c.bias)
            mlp.c_fc = lin

        if "proj" in quantize:
            c = mlp.c_proj
            lin = TritLinear(n_inner, nx, bias=True, weight_mode=mode("proj"),
                             delta_coeff=quant_cfg["delta_coeff"],
                             quantize_input=quant_cfg["quantize_input"])
            with torch.no_grad():
                lin.weight.copy_(c.weight.T.contiguous())
                lin.bias.copy_(c.bias)
            mlp.c_proj = lin

    return model


def build_batches(encoded, max_len, stride, batch_size):
    """把 token ids 流切成 (input_ids) batch 的迭代器工厂。"""
    def make_chunks():
        chunks = []
        for ids in encoded:
            for b in range(0, len(ids) - max_len + 1, stride):
                chunks.append(ids[b:b + max_len])
                if len(chunks) == batch_size:
                    yield {"input_ids": torch.stack(chunks)}
                    chunks = []
        if chunks:
            yield {"input_ids": torch.stack(chunks)}
    return make_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--group", default=None, help="只跑某一组，如 b_all")
    ap.add_argument("--seed", type=int, default=None, help="只跑某一种子")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖训练步数（烟雾测试）")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset

    tok = GPT2TokenizerFast.from_pretrained(cfg["model"])

    def encode(split, limit):
        ds = load_dataset("wikitext", cfg["dataset"], split=split)
        texts = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = torch.tensor(tok(texts, return_tensors="pt").input_ids[0])
        out, limit = [], limit if limit > 0 else 10**9
        for b in range(0, min(len(ids), limit * cfg["data"]["max_len"]),
                       cfg["data"]["max_len"]):
            out.append(ids[b:b + cfg["data"]["max_len"]])
        return out

    print("[data] tokenizing wikitext-2 ...")
    train_ids = encode("train", cfg["data"]["max_train_samples"])
    valid_ids = encode("validation", cfg["eval"].get("max_eval_samples", -1))
    print(f"[data] train chunks: {len(train_ids)}, valid chunks: {len(valid_ids)}")

    logger = ExperimentLogger(str(Path(__file__).resolve().parents[2] / "logs" / "experiments.jsonl"))
    groups = {args.group: cfg["groups"][args.group]} if args.group else cfg["groups"]
    seeds = [args.seed] if args.seed is not None else cfg["training"]["seeds"]

    for gname, gcfg in groups.items():
        for seed in seeds:
            print(f"\n=== group={gname} seed={seed} ===")
            set_seed(seed)
            model = GPT2LMHeadModel.from_pretrained(cfg["model"])
            model = retrofit_gpt2(model, gcfg["quantize"], gcfg["keep_fp"], cfg["quant"])

            tcfg = copy.deepcopy(cfg["training"])
            tcfg.pop("seeds")
            if args.max_steps is not None:
                tcfg["max_steps"] = args.max_steps
            tcfg = QATConfig(**{k: v for k, v in tcfg.items() if k in QATConfig.__dataclass_fields__})

            batches = build_batches(train_ids, cfg["data"]["max_len"],
                                    cfg["data"]["stride"], batch_size=4)
            train_qat(model, batches, tcfg)

            res = evaluate_ppl(model, valid_ids,
                               max_len=cfg["eval"]["max_len"],
                               stride=cfg["eval"]["stride"])
            print(f"=== {gname} seed={seed} -> ppl {res['ppl']:.2f} ===")

            logger.log(
                experiment=cfg["experiment"],
                config={"group": gname, "seed": seed,
                        "quantize": gcfg["quantize"], "keep_fp": gcfg["keep_fp"],
                        "quant": cfg["quant"], "lr": tcfg.lr},
                results=res,
                notes="",
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
