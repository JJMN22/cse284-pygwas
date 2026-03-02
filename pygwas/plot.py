"""
pygwas/plot.py — Manhattan and QQ plots using qqman, matching notebook output.
"""

import matplotlib.pyplot as plt
import pandas as pd
from qqman import qqman


def plot_results(assoc_path: str, out_prefix: str) -> None:
    data = pd.read_csv(assoc_path, sep=r"\s+")
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    fig.set_size_inches((15, 5))
    qqman.manhattan(data, ax=ax0)
    qqman.qqplot(data, ax=ax1)
    path = f"{out_prefix}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}")
