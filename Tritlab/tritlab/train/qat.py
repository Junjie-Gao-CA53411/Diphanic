"""QAT 训练循环：AdamW + 线性 warmup + 梯度裁剪 + bf16 autocast。

训练器只吃 (model, batch_iter, QATConfig)，不关心模型结构——
EX-001 之后换实验，只需要换模型和数据，这里一行都不用动。
"""
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional

import torch
from torch import nn


@dataclass
class QATConfig:
    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 100
    epochs: int = 1
    max_steps: int = -1          # -1 = 跑完 epochs；正数 = 提前停（烟雾测试用）
    grad_clip: float = 1.0
    schedule: str = "constant"     # 'constant' | 'cosine'（从零预训练用 cosine）
    min_lr_ratio: float = 0.1      # cosine 衰减到的最低 lr 比例
    bf16: bool = True
    amp_dtype: str = "bf16"        # 'bf16'（A100/T4）| 'fp16'（P100 等无 bf16 加速的卡）
    log_every: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _make_lr_lambda(cfg: "QATConfig"):
    """warmup 线性爬升；之后按 cfg.schedule 保持常数或 cosine 衰减到 min_lr_ratio。"""
    def fn(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        if cfg.schedule == "cosine" and cfg.max_steps > 0:
            p = min((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
            c = 0.5 * (1 + math.cos(math.pi * p))
            return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * c
        return 1.0
    return fn


def train_qat(
    model: nn.Module,
    batch_iter: Callable[[], Iterator[Dict[str, torch.Tensor]]],
    cfg: QATConfig,
    eval_hook: Optional[Callable[[nn.Module, int], None]] = None,
) -> Dict[str, List[float]]:
    """batch_iter 是一个【返回迭代器】的函数，每个 epoch 调用一次。

    eval_hook(model, step) 可选，用于训练中途评估。
    返回 history: {'loss': [...], 'step': [...]}。
    """
    model.to(cfg.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    if cfg.schedule == "cosine" and cfg.max_steps <= 0:
        print("[warn] schedule='cosine' 需要 max_steps>0，本次退化为 constant")
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _make_lr_lambda(cfg))

    history: Dict[str, List[float]] = {"loss": [], "step": []}
    global_step = 0
    t0 = time.time()
    done = False

    for epoch in range(cfg.epochs):
        if done:
            break
        for batch in batch_iter():
            ids = batch["input_ids"].to(cfg.device)
            _AMP = {"bf16": torch.bfloat16, "fp16": torch.float16}
            amp_dtype = _AMP[cfg.amp_dtype]
            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=(cfg.bf16 and cfg.device == "cuda")):
                out = model(input_ids=ids, labels=ids)
            loss = out.loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()

            history["loss"].append(loss.item())
            history["step"].append(global_step)

            if global_step % cfg.log_every == 0:
                print(f"  step {global_step:5d} | loss {loss.item():.4f} | "
                      f"lr {sched.get_last_lr()[0]:.2e} | "
                      f"{time.time() - t0:.0f}s")

            global_step += 1
            if 0 < cfg.max_steps <= global_step:
                done = True
                break

            if eval_hook is not None:
                eval_hook(model, global_step)

    return history
