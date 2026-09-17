#!/usr/bin/env python3
"""Compact panel(s) for a paper figure: eta vs cumulative projection time, red = factorization-free,
blue = deflated.  usage: [instance[,instance...]] [out.pdf]  -- several comma-separated instances are drawn
side by side (one panel each, shared y-axis, legend in the first panel), e.g. G55mc,G59mc for Fig. 1(b)."""
import os, re, sys, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); RES = os.path.join(ROOT, "results")
INST = (sys.argv[1] if len(sys.argv) > 1 else "G55mc").split(",")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "fig", f"{'_'.join(INST)}_eta_small.pdf")
ARMS = [("baseline16", "Factorization-free ADMM", "#c0392b"), ("deflated16", "Deflated ADMM", "#1f4e9c")]
def load(fn):
    t, e = [], []
    for l in open(fn):
        if re.match(r"^\s+\d+\s", l):
            p = l.split()
            try: v = float(p[4]); tt = float(p[8]); t.append(tt); e.append(v)
            except (ValueError, IndexError): pass
    return np.array(t), np.array(e)
plt.rcParams.update({"font.size": 11})
n = len(INST)
fig, axes = plt.subplots(1, n, figsize=(2.6 + 2.0 * (n - 1), 2.35), sharey=(n > 1), squeeze=False)
ymin, ymax = np.inf, -np.inf
for c, (g, ax) in enumerate(zip(INST, axes[0])):
    series = [(lbl, col, *load(f"{RES}/{g}_{arm}.txt")) for arm, lbl, col in ARMS]
    vals = np.concatenate([e[t >= 1.0] for *_, t, e in series]); vals = vals[np.isfinite(vals) & (vals > 0)]
    ymin, ymax = min(ymin, vals.min()), max(ymax, vals.max())
    for lbl, col, t, e in series: ax.plot(t, e, color=col, lw=1.1, label=lbl)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.0, 1.1 * max(t.max() for *_, t, _ in series))
    ax.grid(True, which="both", lw=0.4, alpha=0.6); ax.tick_params(labelsize=9)
    ax.set_xlabel("Projection time (s)", fontsize=10)
    if n > 1: ax.set_title(g, fontsize=10)
    if c == 0:
        ax.set_ylabel(r"$\eta$", fontsize=11)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right", handlelength=1.8, borderaxespad=0.2)
axes[0][0].set_ylim(10 ** math.floor(math.log10(ymin) - 0.05), 10 ** math.ceil(math.log10(ymax) + 0.05))
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(OUT, bbox_inches="tight"); print("wrote", OUT)
