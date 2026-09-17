"""Parse the "# track" checkpoint lines of results/<inst>_deflated16_track.txt and draw
fig/tracking_accuracy.pdf (Appendix D): the relative error of the deflated Ritz values against
the exact eigenvalues, at the checkpoints where the gate is active, one panel per instance.
Also writes results/summary.csv (median / max per instance).
Usage: python analyze.py [--results DIR] [--fig DIR]
"""
import argparse
import csv
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, LogLocator

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42,
                     "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.7, "grid.linewidth": 0.5,
                     "grid.alpha": 0.35, "lines.linewidth": 1.4})
ORDER = ["G55mc", "G59mc", "G60mc", "G60_mb"]
NAME = {"G55mc": "G55mc ($n=5000$)", "G59mc": "G59mc ($n=5000$)",
        "G60mc": "G60mc ($n=7000$)", "G60_mb": "G60\\_mb ($n=7000$)"}
KV = re.compile(r"(\w+)=([-+0-9.eE]+)")


def parse(path):
    return [{k: float(v) for k, v in KV.findall(line)} for line in open(path) if line.startswith("# track it=")]


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--fig", default=os.path.join(here, "fig"))
    args = ap.parse_args()
    os.makedirs(args.fig, exist_ok=True)
    data = {inst: parse(os.path.join(args.results, f"{inst}_deflated16_track.txt")) for inst in ORDER
            if os.path.exists(os.path.join(args.results, f"{inst}_deflated16_track.txt"))}
    insts = [i for i in ORDER if i in data]
    fig, axes = plt.subplots(1, len(insts), figsize=(2.2 * len(insts) + 0.6, 1.9), sharey="row", squeeze=False)
    summary = []
    for j, inst in enumerate(insts):
        r = data[inst]
        it = np.array([x["it"] for x in r])
        eig = np.array([x["eig_err"] for x in r])
        on = (np.array([x["fired"] for x in r]) > 0) & (eig > 0)
        ax = axes[0][j]
        ax.plot(it[on], eig[on], ".-", color="#1c5fb0", ms=3.5)
        ax.set_yscale("log"); ax.set_ylim(1e-9, 1e-6); ax.grid(True, which="major")
        ax.yaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ax.set_title(NAME[inst]); ax.set_xlabel("ADMM iteration")
        if j == 0:
            ax.set_ylabel(r"$\max_{i\in I}|\lambda_i-\lambda_i^\star|/|\lambda_i^\star|$")
        summary.append(dict(instance=inst, n_ckpt=len(r), n_gate_on=int(on.sum()),
                            eig_err_med=float(np.median(eig[on])), eig_err_max=float(eig[on].max())))
    fig.tight_layout()
    out = os.path.join(args.fig, "tracking_accuracy.pdf")
    fig.savefig(out); print("wrote", out)
    with open(os.path.join(args.results, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    for d in summary:
        print(f"{d['instance']:7s} checkpoints {d['n_ckpt']} (gate on {d['n_gate_on']}) "
              f"eig_err median {d['eig_err_med']:.2e} max {d['eig_err_max']:.2e}")


if __name__ == "__main__":
    main()
