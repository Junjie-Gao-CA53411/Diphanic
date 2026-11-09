"""扫描实验画图（Delta 系数 / gamma / lr 等一维扫描）。"""
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_sweep(
    x: Sequence[float],
    series: Dict[str, Sequence[float]],
    xlabel: str,
    ylabel: str,
    title: str = "",
    save_path: str = "",
) -> None:
    plt.figure(figsize=(6, 4))
    for name, ys in series.items():
        plt.plot(x, ys, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] saved -> {save_path}")
    plt.show()
