#!/usr/bin/env python
"""把语料 tokenize 成 uint16 bin（vocab 必须 < 65536）。

用法：
  python tools/prepare_tokens.py --tokenizer gpt2 \
      --out data/tinystories_train.bin data/raw/tinystories_train.txt
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="原始文本文件")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=-1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    all_ids = []
    for f in args.files:
        text = Path(f).read_text(encoding="utf-8")
        # 按空行切条目逐条编码，避免超长单行
        for para in text.split("\n\n"):
            para = para.strip()
            if para:
                all_ids.extend(tok(para, add_special_tokens=False).input_ids)
        print(f"[prep] {f}: 累计 {len(all_ids)} tokens")

    ids = np.array(all_ids[:args.max_tokens] if args.max_tokens > 0 else all_ids,
                   dtype=np.uint16)
    assert int(ids.max()) < 65536, "vocab >= 65536 会溢出 uint16，换 uint32"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(args.out)
    print(f"[prep] wrote {len(ids)} tokens -> {args.out}")


if __name__ == "__main__":
    main()
