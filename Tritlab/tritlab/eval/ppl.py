"""固定评测：滑动窗口 perplexity（Henly-Ney 风格）。

铁规矩：这个文件永远不改。所有实验用同一把尺子。
输入是【已经 tokenize 好的】token ids 列表，与 tokenizer 解耦，
方便 smoke test 用随机数据跑通。
"""
import math
from typing import Dict, List

import torch
from torch import nn


@torch.no_grad()
def evaluate_ppl(
    model: nn.Module,
    encoded: List[torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_len: int = 1024,
    stride: int = 512,
    max_samples: int = -1,
) -> Dict[str, float]:
    model.to(device)
    model.eval()

    nll_sum, n_tokens = 0.0, 0
    seqs = encoded[:max_samples] if max_samples > 0 else encoded

    for ids in seqs:
        ids = ids.to(device)
        L = len(ids)
        prev_end = 0
        for begin in range(0, L, stride):
            end = min(begin + max_len, L)
            chunk = ids[begin:end].unsqueeze(0)
            target_ids = chunk.clone()
            # 与前一个窗口重叠的部分不算 loss（避免重复计数）
            target_ids[0, :prev_end - begin] = -100
            logits = model(input_ids=chunk).logits
            shift_logits = logits[0, :-1].float()
            shift_target = target_ids[0, 1:]
            mask = shift_target != -100
            if mask.sum() == 0:
                prev_end = end
                continue
            logp = torch.log_softmax(shift_logits, dim=-1)
            tgt = shift_target.masked_fill(~mask, 0).unsqueeze(-1)
            nll_sum += -logp.gather(-1, tgt).squeeze(-1)[mask].sum().item()
            n_tokens += mask.sum().item()
            prev_end = end
            if end == L:
                break

    ppl = math.exp(nll_sum / max(n_tokens, 1))
    return {"ppl": ppl, "n_tokens": n_tokens}
