#!/usr/bin/env python3
"""Three-method hardware-search wall-clock figure.

Reads update/wallclock_three_methods_data.json and writes
fig3/wallclock_three_methods_v1.png (and .pdf).

Bars per workload, log-scale hours:
  Vizier+Timeloop+LP  (static method, 50 trials, one H100)
  Vizier+LP           (V1 joint-tile formulation, 50 trials, one H100)
  Master LP (OCS)     (one solve of the finite master program, A100 host)

A missing master_lp_h is drawn as no bar. A static 'inf' is drawn as a red
cross. Annotations give the Vizier+LP speedup over the static method.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "wallclock_three_methods_data.json")
OUT = os.path.join(ROOT, "fig3", "wallclock_three_methods_v1")

METHODS = [
    ("static_h", "Vizier + Timeloop + LP (static)", "#f28e2b"),
    ("v1_h", "Vizier + LP (V1, LLM-guided)", "#4e79a7"),
    ("master_lp_h", "Master LP (OCS, one solve)", "#59a14f"),
]


def main():
    d = json.load(open(DATA))
    wl = d["workloads"]
    n = len(wl)
    w = 0.26
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(16, 6.2))
    ymin, ymax = 1e-3, 3e2
    for j, (key, label, color) in enumerate(METHODS):
        xs = x + (j - 1) * w
        labeled = False
        for i, row in enumerate(wl):
            v = row[key]
            if v is None:
                continue
            if v == "inf":
                ax.text(xs[i], 0.035, "✗", ha="center", va="center",
                        color="#c00000", fontsize=18, fontweight="bold")
                ax.text(xs[i], 0.012, "static\ninfeasible", ha="center",
                        va="center", color="#c00000", fontsize=7)
                continue
            ax.bar(xs[i], v, width=w, color=color, bottom=ymin,
                   label=None if labeled else label, zorder=3)
            labeled = True
            txt = f"{v:.2f}h" if v >= 0.1 else f"{v*60:.1f}min"
            ax.text(xs[i], v * 1.25, txt, ha="center", va="bottom", fontsize=7)

    # speedup annotation: static / v1
    for i, row in enumerate(wl):
        s, v1 = row["static_h"], row["v1_h"]
        if s == "inf":
            lab = "∞×"
        else:
            lab = f"{s / v1:.1f}×"
        ax.text(x[i], ymax * 0.55, lab, ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="#1f4e9c" if row["family"] == "vision" else "#b00000")

    # family divider
    nv = sum(1 for r in wl if r["family"] == "vision")
    ax.axvline(nv - 0.5, color="gray", ls="--", lw=1)
    ax.text((nv - 1) / 2, ymax * 1.3, "EfficientNet (vision)  —  Vizier+LP speedup vs static",
            ha="center", color="#1f4e9c", fontsize=11)
    ax.text(nv + (n - nv - 1) / 2, ymax * 1.3, "LLMs  —  Vizier+LP speedup vs static",
            ha="center", color="#b00000", fontsize=11)

    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax * 3)
    ax.set_ylabel("Hardware-search wall-clock (hours, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n{r['params']}" for r in wl], fontsize=9)
    ax.grid(axis="y", which="major", ls=":", alpha=0.6, zorder=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=9, frameon=True)
    ax.set_xlim(-0.6, n - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=220)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    sys.exit(main())
