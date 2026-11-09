#!/usr/bin/env python
"""EX-002 runner：从零训练 TritGPT（原生三值 LM）。

用法：
  python train_from_scratch.py                       # 按 config 跑全部组
  python train_from_scratch.py --group n_all --max-steps 20   # 烟雾测试
"""
import argparse
import copy
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tritlab.data import load_eval_chunks, make_loader      # noqa: E402
from tritlab.eval import evaluate_ppl                       # noqa: E402
from tritlab.models import TritGPT, TritGPTConfig           # noqa: E402
from tritlab.train import QATConfig, train_qat              # noqa: E402
from tritlab.utils import ExperimentLogger, set_seed        # noqa: E402


def build_model(mcfg: dict, qcfg: dict, gcfg: dict) -> TritGPT:
    cfg = TritGPTConfig(**{**mcfg, **qcfg, **gcfg})
    model = TritGPT(cfg)
    n = model.count_params()
    print(f"[model] TritGPT dim={cfg.dim} L={cfg.n_layers} "
          f"mode={cfg.weight_mode} -> {n/1e6:.1f}M params "
          f"(三值部分有效位宽 ~1.58-bit, embedding/head 高精度)")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Tritlab\\experiments\\EX-002-native-ternary-v0\\config.yaml")
    ap.add_argument("--group", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    logger = ExperimentLogger(str(root / "logs" / "experiments.jsonl"))

    groups = {args.group: cfg["groups"][args.group]} if args.group else cfg["groups"]
    seeds = [args.seed] if args.seed is not None else cfg["training"]["seeds"]

    for gname, gcfg in groups.items():
        for seed in seeds:
            print(f"\n=== group={gname} seed={seed} ===")
            set_seed(seed)
            model = build_model(cfg["model"], cfg.get("quant", {}), gcfg)

            loader = make_loader(cfg["data"]["train_bin"],
                                 cfg["data"]["seq_len"],
                                 cfg["training"]["batch_size"],
                                 shuffle=True)
            batch_iter = lambda: iter(loader)   # 每个 epoch 一个新迭代器

            tcfg = copy.deepcopy(cfg["training"])
            tcfg.pop("seeds")
            tcfg.pop("batch_size")
            if args.max_steps is not None:
                tcfg["max_steps"] = args.max_steps
            tcfg = QATConfig(**{k: v for k, v in tcfg.items()
                                if k in QATConfig.__dataclass_fields__})

            train_qat(model, batch_iter, tcfg)

            # 训练后记录三态比例（L3-B 数据管道，顺手就采了）
            stats = [m.stats() for m in model.modules() if hasattr(m, "stats")]
            if stats:
                z = [s["frac_zero"] for s in stats]
                print(f"[stats] 0 比例: 首层 {z[0]:.3f} 末层 {z[-1]:.3f} 平均 {sum(z)/len(z):.3f}")

            eval_chunks = load_eval_chunks(cfg["data"]["valid_bin"],
                                           cfg["eval"]["max_len"],
                                           cfg["eval"]["max_eval_samples"])
            res = evaluate_ppl(model, eval_chunks,
                               max_len=cfg["eval"]["max_len"],
                               stride=cfg["eval"]["stride"])
            print(f"=== {gname} seed={seed} -> ppl {res['ppl']:.2f} ===")

            logger.log(cfg["experiment"],
                       {"group": gname, "seed": seed, "model": cfg["model"],
                        "quant": gcfg, "training": {k: v for k, v in tcfg.__dict__.items()}},
                       {"ppl": res["ppl"], "frac_zero_first": z[0] if stats else None,
                        "frac_zero_last": z[-1] if stats else None})

            # 保存 checkpoint（模型线的产品！命名带实验号）
            out = Path(__file__) / ".." / "ckpt" / f"{gname}_s{seed}.pt"
            out = Path(out).resolve()
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "config": dict(cfg["model"], **gcfg)}, out)
            print(f"[ckpt] saved -> {out}")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
