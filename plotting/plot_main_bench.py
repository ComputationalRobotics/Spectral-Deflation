"""Paper figures + data for the four-arm main benchmark.

Per model, two figures (PDF for the paper + PNG preview) and two CSVs holding
exactly the plotted series (so the paper numbers are re-derivable without
re-parsing the JSONs):

    <model>_loss.{pdf,png}         all four arms: validation | training loss,
                                   each vs iterations and vs time
    <model>_ns_val_loss.{pdf,png}  Newton Schulz arms only (plain + deflated):
                                   training loss vs iterations | validation
                                   loss vs iterations | vs time
    <model>_pe_val_loss.{pdf,png}  the same for the Polar Express arms
    <model>_{ns,pe}_train_loss     training loss vs iterations | vs time
    <model>_val_data.csv           arm, step, val_loss, train_time_s, wall_time_s
    <model>_train_data.csv         arm, step, loss, cum_train_time_s

Iterations = optimizer steps (the 1B benchmark is 1905 steps = one epoch).
The default time axis is cumulative TRAINING time in seconds (sum of
step_times, recorded at every val point as val_train_times) -- eval overhead
excluded, so arms are comparable even if their val cadence ever differs.
Pass --time-axis wall for raw wall-clock time since training start instead
(val_wall_times; includes eval/logging overhead, identical across arms at
equal cadence).  The CSVs always carry both.

The standalone figures are drawn PER polar method: one file for the Newton
Schulz pair (plain + deflated), one for the Polar Express pair, so each
shows the deflation effect for one method without four curves in one panel.
The combined <model>_loss figure keeps all four arms.

All three figures are drawn by plotting/curve_layout.py (combined_figure /
split_figure), the same code the per-tier ablation scripts use, so the main
benchmark and the ablations share one look: x-axes from --xmin-iter (default
1000) onward, y-window auto-tightened to the displayed curves, 16in x 4
panels at aspect 0.55.

Usage:
    python plotting/plot_main_bench.py gpt-large
    python plotting/plot_main_bench.py gpt-large --root outputs/main_bench --out figures/main_bench
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# shared curve layout, also used by every per-tier ablation figure -- the
# main benchmark and the ablations must look identical
from curve_layout import combined_figure, family_figures

DEFAULT_ROOT = "outputs/main_bench"
DEFAULT_OUT = "figures/main_bench"

# (label, list of candidate run-dir names, color, linestyle, polar family)
# Fixed color per arm; linestyles differ so the figure survives grayscale print.
# The family is the polar method the arm shares with one other arm; it groups
# the column pairs of the standalone val/train figures.
ARMS = [
    ("Muon (Newton Schulz)",          ["ns5-ns5"],             "#7ab0e2", (0, (5, 2)),           "Newton Schulz"),
    ("Muon (Polar Express)",          ["polarexpress-ns5"],    "#f2a06c", (0, (1.5, 1.5)),       "Polar Express"),
    ("Deflated Muon (Newton Schulz)", ["deflated_ns_cut-ns5"], "#1c5fb0", "-",                   "Newton Schulz"),
    ("Deflated Muon (Polar Express)", ["deflated_pe_cut-ns5"], "#d95f02", "-",                   "Polar Express"),
]

# left-to-right order of the column pairs in the standalone figures
FAMILY_ORDER = ["Newton Schulz", "Polar Express"]
_FAMILY = {label: fam for label, _, _, _, fam in ARMS}

TRAIN_SMOOTH = 100  # steps; the raw series is drawn faintly underneath

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "pdf.fonttype": 42, "ps.fonttype": 42,   # embed TrueType, no Type-3
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "grid.linewidth": 0.5, "grid.alpha": 0.35,
    "legend.frameon": False, "lines.linewidth": 1.4,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def load(root, model):
    runs = []
    for label, names, color, ls, _fam in ARMS:
        for name in names:
            hits = sorted(glob.glob(os.path.join(
                root, f"{model}-std0.01", name, "logs_jobid_*.json")))
            if hits:
                runs.append((label, json.load(open(hits[0])), color, ls))
                break
        else:
            print(f"  [warn] {model}: no run found for '{label}' "
                  f"(tried {names}) -- skipped")
    return runs



def fig_curves(runs, out, model, val_interval, time_axis,
               xmin_iter=1000, figwidth=16, aspect=0.55, ytop=None):
    """The combined figure with all four arms plus, per polar method, a
    standalone validation and a training figure (Newton Schulz and Polar
    Express in separate files) -- the same five-figure set, from the same
    layout code, as the ablation tiers."""
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    time_label = ("wall-clock time (s)" if time_axis == "wall"
                  else "training time (s)")
    series = []
    for label, j, color, ls in runs:
        series.append(dict(
            label=label, color=color, ls=ls, family=_FAMILY.get(label),
            val_steps=j["val_steps"], val_losses=j["val_losses"],
            val_times=j[tkey], train_losses=j["losses"],
            train_times=np.cumsum(np.asarray(j["step_times"], float))))
    common = dict(xmin_iter=xmin_iter, figwidth=figwidth, aspect=aspect,
                  val_interval=val_interval, train_smooth=TRAIN_SMOOTH,
                  time_label=time_label)
    figs = [("loss", combined_figure(series, **common))]
    for kind in ("val", "train"):
        figs += [(f"{slug}_{kind}_loss", fig) for slug, fig in
                 family_figures(series, kind, family_order=FAMILY_ORDER,
                                ytop=ytop, **common)]
    for tag, fig in figs:
        if fig is None:
            continue
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"{model}_{tag}.{ext}"),
                        dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}/{model}_{tag}.{{pdf,png}}")


def write_csvs(runs, out, model):
    """Dump exactly the raw plotted series (no smoothing, no trimming)."""
    p = os.path.join(out, f"{model}_val_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "val_loss", "train_time_s", "wall_time_s"])
        for label, j, _, _ in runs:
            for s, v, tt, wt in zip(j["val_steps"], j["val_losses"],
                                    j["val_train_times"], j["val_wall_times"]):
                w.writerow([label, s, v, tt, wt])
    print(f"  wrote {p}")
    p = os.path.join(out, f"{model}_train_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "loss", "cum_train_time_s"])
        for label, j, _, _ in runs:
            t = np.cumsum(np.asarray(j["step_times"], float))
            for i, (y, tt) in enumerate(zip(j["losses"], t), 1):
                w.writerow([label, i, y, tt])
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="gpt-large")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--time-axis", choices=["train", "wall"], default="train",
                    help="time panel x-axis: 'train' = cumulative training "
                         "compute (eval excluded; the like-for-like default), "
                         "'wall' = raw wall-clock time since training start")
    ap.add_argument("--val-interval", type=int, default=100,
                    help="plot validation points every this many optimizer "
                         "steps (must be a multiple of the recorded cadence)")
    # layout knobs: same defaults as the ablation scripts
    ap.add_argument("--xmin-iter", type=float, default=1000,
                    help="start the x-axes at this optimizer step (the time "
                         "panel starts at the same fraction of the run); "
                         "crops the early cliff so the arms separate")
    ap.add_argument("--figwidth", type=float, default=16)
    ap.add_argument("--ytop", type=float, default=None,
                    help="fixed y-axis top for the per-family figures "
                         "(default: auto-tightened to the displayed curves)")
    ap.add_argument("--aspect", type=float, default=0.55,
                    help="panel box height/width ratio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for model in args.models:
        runs = load(args.root, model)
        if not runs:
            print(f"{model}: nothing to plot")
            continue
        write_csvs(runs, args.out, model)
        fig_curves(runs, args.out, model, args.val_interval, args.time_axis,
                   args.xmin_iter, args.figwidth, args.aspect, args.ytop)


if __name__ == "__main__":
    main()
