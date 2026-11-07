import math
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = Path(__file__).resolve().parent

block_size = 1024
d_model = 512
n_head = 8
n_layer = 12
batch_size = 48
lr = 6e-4
max_iters = 5000
warmup = 200
eval_every = 250
data_file = BASE / "data" / "sample.jsonl"
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)

text = data.get("text")
print(text)

