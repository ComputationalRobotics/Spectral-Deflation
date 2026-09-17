#!/usr/bin/env python3
"""eta vs cumulative projection time, in the style of a Fig.-3 panel of Kang et al. (arXiv:2507.09165):
log-log, thin solid lines, grid.  Both arms of one instance share a panel: red = the reference
factorization-free FP16 filter, blue = the same filter behind deflation.  y-limits cover both arms.
usage: scripts/plot_convergence.py [instances...]  -> fig/eta_grid.pdf (1 x n panels) and fig/<instance>_eta.pdf"""
import os, re, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, "results"); FIG = os.path.join(ROOT, "fig")
INST = sys.argv[1:] or ["G55mc", "G59mc", "G60mc", "G60_mb"]
ARMS = [("baseline16", "Factorization-free ADMM", "#c0392b"), ("deflated16", "Deflated ADMM", "#1f4e9c")]
def load(fn):
    t, e = [], []                                   # cumulative projection time (s), eta
    for l in open(fn):
        if re.match(r"^\s+\d+\s", l):
            p = l.split()
            try: v = float(p[4]); tt = float(p[8]); t.append(tt); e.append(v)
            except (ValueError, IndexError): pass
    return np.array(t), np.array(e)
def nice_limits(vals):
    v = vals[np.isfinite(vals) & (vals > 0)]
    return 10 ** math.floor(math.log10(v.min()) - 0.05), 10 ** math.ceil(math.log10(v.max()) + 0.05)
def panel(ax, g, show_y):
    series = [(lbl, col, *load(f"{RES}/{g}_{arm}.txt")) for arm, lbl, col in ARMS if os.path.exists(f"{RES}/{g}_{arm}.txt")]
    if not series: ax.axis("off"); return
    for lbl, col, t, e in series:
        ax.plot(t, e, color=col, lw=0.9, ls="-", label=lbl)
    tmax = max(t.max() for *_, t, _ in series)
    ylim = nice_limits(np.concatenate([e[t >= 1.0] for *_, t, e in series]))   # y-range over the plotted window
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1.0, 1.1 * tmax); ax.set_ylim(*ylim)   # from 1 s on
    ax.grid(True, which="both", lw=0.4, alpha=0.6); ax.set_title(g, fontsize=10)
    ax.tick_params(labelsize=7.5); ax.set_xlabel("Projection time (s)", fontsize=8.5)
    if show_y: ax.set_ylabel(r"$\eta$", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right", handlelength=2.2)
fig, axes = plt.subplots(1, len(INST), figsize=(3.1 * len(INST), 2.7), squeeze=False)
for c, g in enumerate(INST): panel(axes[0][c], g, show_y=(c == 0))
fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
fig.savefig(f"{FIG}/eta_grid.pdf", bbox_inches="tight"); print("wrote", f"{FIG}/eta_grid.pdf")
for g in INST:
    fig, ax = plt.subplots(figsize=(3.4, 2.8)); panel(ax, g, show_y=True); fig.tight_layout()
    fig.savefig(f"{FIG}/{g}_eta.pdf", bbox_inches="tight"); print("wrote", f"{FIG}/{g}_eta.pdf")
