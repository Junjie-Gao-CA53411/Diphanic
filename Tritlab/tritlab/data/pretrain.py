"""从零预训练的数据管道：uint16 bin 文件 -> memmap -> DataLoader。

v0 用"不重叠定长切块"（GPT-2 经典做法）：简单、可复现、无随机性歧义。
语料 tokenize 成 bin 用 tools/prepare_tokens.py。
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MemmapTokens(Dataset):
    def __init__(self, bin_path: str, seq_len: int):
        self.arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.arr) // self.seq_len

    def __getitem__(self, i: int):
        chunk = torch.from_numpy(
            self.arr[i * self.seq_len:(i + 1) * self.seq_len].astype(np.int64))
        return {"input_ids": chunk}


def make_loader(bin_path: str, seq_len: int, batch_size: int,
                shuffle: bool = True, num_workers: int = 2,
                max_samples: int = -1) -> DataLoader:
    ds = MemmapTokens(bin_path, seq_len)
    if max_samples > 0:
        from torch.utils.data import Subset
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=True, num_workers=num_workers)


def load_eval_chunks(bin_path: str, seq_len: int,
                     max_chunks: int = 200) -> list:
    """给 evaluate_ppl 用的定长 chunks 列表（不打乱，保证评测可复现）。"""
    ds = MemmapTokens(bin_path, seq_len)
    n = min(max_chunks, len(ds))
    return [ds[i]["input_ids"] for i in range(n)]
