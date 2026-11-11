#!/usr/bin/env python3
"""Three-method hardware-search wall-clock figure with per-bar breakdown.

Reads update/wallclock_three_methods_data.json and writes
fig3/wallclock_three_methods_v1.png (and .pdf).

Per workload, three stacked bars on a log-hours axis:
  Vizier+Timeloop+LP  (static method, 50 trials, one H100)
                      segments: Timeloop / Vizier + LP build / LP solve
  Vizier+LP           (LLM-guided joint-tile formulation, 50 trials, one H100)
                      segments: Vizier + LP build / LP solve
  Master LP (OCS)     (recorded solve stages of the retained design, A100 host)
                      segments: joint + hardware-screen solves /
                                fixed-chip controls / branch and bound + certificate

A missing master bar means no such solve exists for the workload. A static
'inf' is drawn as a red cross. Annotations give the Vizier+LP speedup over
the static method.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "wallclock_three_methods_data.json")
OUT = os.path.join(ROOT, "fig3", "wallclock_three_methods_v1")

YMIN = 1e-3
BARW = 0.26

# (data key, legend label, color), stacked bottom-up in this order
TRIAL_COMPONENTS = [
    ("lp_solve_h", "LP solve", "#1f77b4"),
    ("vizier_lp_build_h", "Vizier + LP build", "#2ca02c"),
    ("timeloop_h", "Timeloop", "#ff7f0e"),
]
MASTER_COMPONENTS = [
    ("joint_hardware_screen_s", "Joint + hardware-screen solves", "#6a51a3"),
    ("fixed_chip_controls_s", "Fixed-chip controls", "#9e9ac8"),
    ("bnb_certificate_s", "Branch and bound + certificate", "#cbc9e2"),
]
METHOD_LABELS = ["static", "LLM", "master"]


def stack(ax, x, comps, spec):
    """Draw a stacked bar from a components dict; return the total height."""
    bottom = YMIN
    for key, _label, color in spec:
        v = comps.get(key, 0.0) or 0.0
        if v <= 0:
            continue
        ax.bar(x, v, width=BARW, bottom=bottom, color=color, zorder=3,
               edgecolor="white", linewidth=0.3)
        bottom += v
    return bottom


def fmt(h):
    return f"{h:.2f}h" if h >= 0.1 else f"{h*60:.1f}min"


def main():
    d = json.load(open(DATA))
    wl = d["workloads"]
    n = len(wl)
    x = np.arange(n)
    ymax = 3e2

    fig, ax = plt.subplots(figsize=(16, 6.6))
    for i, row in enumerate(wl):
        xs = [x[i] - BARW, x[i], x[i] + BARW]
        # static
        if row["static_h"] == "inf":
            ax.text(xs[0], 0.035, "✗", ha="center", va="center",
                    color="#c00000", fontsize=18, fontweight="bold")
            ax.text(xs[0], 0.012, "static\ninfeasible", ha="center", va="center",
                    color="#c00000", fontsize=7)
        else:
            top = stack(ax, xs[0], row["static_components"], TRIAL_COMPONENTS)
            ax.text(xs[0], top * 1.25, fmt(row["static_h"]), ha="center",
                    va="bottom", fontsize=7)
        # LLM-guided
        top = stack(ax, xs[1], row["llm_guided_components"], TRIAL_COMPONENTS)
        ax.text(xs[1], top * 1.25, fmt(row["v1_h"]), ha="center", va="bottom",
                fontsize=7)
        # master LP
        mc = row.get("master_lp_components")
        if mc:
            top = stack(ax, xs[2], mc, MASTER_COMPONENTS)
            ax.text(xs[2], top * 1.25, fmt(row["master_lp_h"]), ha="center",
                    va="bottom", fontsize=7)
        # speedup annotation (static / LLM-guided)
        s, v1 = row["static_h"], row["v1_h"]
        lab = "∞×" if s == "inf" else f"{s / v1:.1f}×"
        ax.text(x[i], ymax * 0.55, lab, ha="center", va="bottom", fontsize=10,
                fontweight="bold",
                color="#1f4e9c" if row["family"] == "vision" else "#b00000")
        # per-bar method labels
        for xx, ml in zip(xs, METHOD_LABELS):
            if ml == "master" and not mc:
                continue
            ax.text(xx, YMIN * 0.72, ml, ha="center", va="top", fontsize=7,
                    color="#444444")

    nv = sum(1 for r in wl if r["family"] == "vision")
    ax.axvline(nv - 0.5, color="gray", ls="--", lw=1)
    ax.text((nv - 1) / 2, ymax * 1.3,
            "EfficientNet (vision)  —  Vizier+LP speedup vs static",
            ha="center", color="#1f4e9c", fontsize=11)
    ax.text(nv + (n - nv - 1) / 2, ymax * 1.3,
            "LLMs  —  Vizier+LP speedup vs static",
            ha="center", color="#b00000", fontsize=11)

    ax.set_yscale("log")
    ax.set_ylim(YMIN, ymax * 3)
    ax.set_ylabel("Hardware-search wall-clock (hours, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"\n\n{r['label']}\n{r['params']}" for r in wl], fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", which="major", ls=":", alpha=0.6, zorder=0)
    ax.set_xlim(-0.6, n - 0.4)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    handles = [Patch(color=c, label=l) for _, l, c in reversed(TRIAL_COMPONENTS)]
    handles += [Patch(color=c, label=l) for _, l, c in MASTER_COMPONENTS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.13),
              ncol=6, fontsize=8.5, frameon=True,
              title="50-trial search components (left two bars)          master LP solve stages (right bar)",
              title_fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT + ".png", dpi=220)
    fig.savefig(OUT + ".pdf")
    print("wrote", OUT + ".png")


if __name__ == "__main__":
    sys.exit(main())
