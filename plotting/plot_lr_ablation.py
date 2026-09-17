"""Paper figures + data for the learning-rate ablation.

Reads the runs written by experiments/lr_ablation.sbatch (three muon polar
methods x lr in {0.01, 0.009, ..., 0.002} (step 0.001), main-experiment protocol
otherwise) and writes, per model:

    <model>_lr_final_val.{pdf,png}         final validation loss vs lr, one
                                           panel per polar method (plain +
                                           deflated in each)
    <model>_lr<lr>_val_loss.{pdf,png}      per-lr method comparison, validation
                                           loss -- left vs iterations, right vs
                                           wall time (main-figure style)
    <model>_lr<lr>_train_loss.{pdf,png}    same for training loss (smoothed)
    <model>_lr_final_val.csv               arm, lr, final_val_loss,
                                           total_train_time_s, total_wall_time_s
    <model>_lr_val_data.csv                arm, lr, step, val_loss,
                                           train_time_s, wall_time_s
    <model>_lr_train_data.csv              arm, lr, step, loss, cum_train_time_s

The CSVs hold exactly the raw recorded series (no smoothing, no trimming).
Final val loss is the last recorded eval.  The time axis default is cumulative
training time (eval excluded); pass --time-axis wall for raw wall-clock time.

Usage:
    python plotting/plot_lr_ablation.py gpt-large
    python plotting/plot_lr_ablation.py gpt-large --root outputs/lr_ablation --out figures/lr_ablation
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
from curve_layout import combined_figure

DEFAULT_ROOT = "outputs/lr_ablation"
DEFAULT_OUT = "figures/lr_ablation"
LR_GRID = ["0.01", "0.009", "0.008", "0.007", "0.006", "0.005",
           "0.004", "0.003", "0.002"]   # dir-name strings

# (label, run-dir prefix, color, linestyle) -- identical to the main figure
# the 5th field is the polar family: it pairs each plain arm with its
# deflated counterpart and gives the summary figure one panel per method
ARMS = [
    ("Muon (Newton Schulz)",          "ns5",             "#7ab0e2", (0, (5, 2)),     "Newton Schulz"),
    ("Muon (Polar Express)",          "polarexpress",    "#f2a06c", (0, (1.5, 1.5)), "Polar Express"),
    ("Deflated Muon (Newton Schulz)", "deflated_ns_cut", "#1c5fb0", "-",             "Newton Schulz"),
    ("Deflated Muon (Polar Express)", "deflated_pe_cut", "#d95f02", "-",             "Polar Express"),
]

# left-to-right panel order of the summary figure
FAMILY_ORDER = ["Newton Schulz", "Polar Express"]

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
    """-> {lr_str: [(label, json, color, ls), ...]}, missing runs warned."""
    runs = {}
    for lr in LR_GRID:
        tier = []
        for label, prefix, color, ls, _fam in ARMS:
            hits = sorted(glob.glob(os.path.join(
                root, f"{model}-std0.01", f"{prefix}-ns5-lr{lr}",
                "logs_jobid_*.json")))
            if hits:
                tier.append((label, json.load(open(hits[0])), color, ls))
            else:
                print(f"  [warn] {model}: no run for '{label}' lr={lr} -- skipped")
        if tier:
            runs[lr] = tier
    return runs


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return np.asarray(y, float)
    return np.convolve(np.asarray(y, float), np.ones(w) / w, mode="valid")


def _panels(ylabel, time_axis):
    fig, (ax_i, ax_t) = plt.subplots(1, 2, figsize=(6.75, 3.4), sharey=True)
    ax_i.set_xlabel("iterations")
    ax_t.set_xlabel("wall-clock time (s)" if time_axis == "wall"
                    else "training time (s)")
    ax_i.set_ylabel(ylabel)
    for a in (ax_i, ax_t):
        a.grid(True, axis="y")
        a.set_box_aspect(1)
    return fig, ax_i, ax_t


def _short(label, family):
    """Legend label with the family suffix dropped -- the panel title already
    names it ("Deflated Muon (Polar Express)" -> "Deflated Muon")."""
    return label.replace(f" ({family})", "")


def fig_final(runs, out, model):
    """Final val loss vs lr, ONE PANEL PER POLAR METHOD: Newton Schulz
    (plain + deflated) left, Polar Express right, so each panel shows the
    deflation gap across the whole lr grid without four curves in one axes.
    Shared y-axis, so the panels are directly comparable.  The CSV is
    unsplit and still carries all four arms."""
    # collect once per arm, then lay the arms out by family
    series, rows = {}, []
    for label, _, color, ls, fam in ARMS:
        xs, fv = [], []
        for lr in LR_GRID:
            for lab, j, _, _ in runs.get(lr, []):
                if lab == label:
                    xs.append(float(lr))
                    fv.append(j["val_losses"][-1])
                    rows.append([label, lr, j["val_losses"][-1],
                                 float(np.sum(j["step_times"])),
                                 j["val_wall_times"][-1]])
        if xs:
            order = np.argsort(xs)
            series[label] = (np.asarray(xs)[order], np.asarray(fv)[order],
                             color, ls, fam)
    families = [(fam, [a[0] for a in ARMS if a[4] == fam and a[0] in series])
                for fam in FAMILY_ORDER]
    families = [(fam, labels) for fam, labels in families if labels]
    if not families:
        print(f"  [warn] {model}: no arm loaded -- final-val figure skipped")
        return
    # panel size matches the old single-panel figure (3.6 x 3.2 in), so the
    # two-panel version is simply twice as wide
    fig, axes = plt.subplots(1, len(families),
                             figsize=(3.3 * len(families), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (fam, labels) in zip(axes, families):
        for label in labels:
            xs, fv, color, ls, _f = series[label]
            ax.plot(xs, fv, color=color, linestyle=ls, marker="o",
                    markersize=3.5, label=_short(label, fam))
        ax.set_title(fam)
        # the grid is uniform in lr (step 0.001), so a linear axis places the
        # points evenly; rotated labels keep ten ticks legible
        ax.set_xticks([float(lr) for lr in LR_GRID])
        ax.set_xticklabels(LR_GRID, rotation=45, ha="right")
        ax.set_xlabel("learning rate")
        ax.grid(True, axis="y")
        ax.legend(loc="best")
    axes[0].set_ylabel("final validation loss")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{model}_lr_final_val.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    p = os.path.join(out, f"{model}_lr_final_val.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "lr", "final_val_loss",
                    "total_train_time_s", "total_wall_time_s"])
        w.writerows(rows)
    print(f"  wrote {out}/{model}_lr_final_val.{{pdf,png,csv}}")


def fig_curves(runs, out, model, ymax, val_interval, time_axis,
               xmin_iter=None, figwidth=16, aspect=0.55):
    """Per-tier combined figure, main-benchmark paradigm (see curve_layout)."""
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    for v, tier in runs.items():
        series = []
        for label, j, color, ls in tier:
            series.append(dict(
                label=label, color=color, ls=ls,
                val_steps=j["val_steps"], val_losses=j["val_losses"],
                val_times=j[tkey], train_losses=j["losses"],
                train_times=np.cumsum(np.asarray(j["step_times"], float))))
        fig = combined_figure(
            series, xmin_iter=xmin_iter, figwidth=figwidth, aspect=aspect,
            val_interval=val_interval, train_smooth=TRAIN_SMOOTH,
            time_label=("wall-clock time (s)" if time_axis == "wall"
                        else "training time (s)"),
            tag=f"lr = {v}")
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(out, f"{model}_lr{v}_loss.{ext}"),
                        dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}/{model}_lr{v}_loss.{{pdf,png}}")


def write_csvs(runs, out, model):
    """Dump exactly the raw recorded series (no smoothing, no trimming)."""
    p = os.path.join(out, f"{model}_lr_val_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "lr", "step", "val_loss",
                    "train_time_s", "wall_time_s"])
        for lr, tier in runs.items():
            for label, j, _, _ in tier:
                for s, v, tt, wt in zip(j["val_steps"], j["val_losses"],
                                        j["val_train_times"],
                                        j["val_wall_times"]):
                    w.writerow([label, lr, s, v, tt, wt])
    print(f"  wrote {p}")
    p = os.path.join(out, f"{model}_lr_train_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "lr", "step", "loss", "cum_train_time_s"])
        for lr, tier in runs.items():
            for label, j, _, _ in tier:
                t = np.cumsum(np.asarray(j["step_times"], float))
                for i, (y, tt) in enumerate(zip(j["losses"], t), 1):
                    w.writerow([label, lr, i, y, tt])
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="gpt-large")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--time-axis", choices=["train", "wall"], default="train")
    ap.add_argument("--val-interval", type=int, default=100)
    ap.add_argument("--xmin-iter", type=float, default=1000,
                    help="start the x-axes at this optimizer step (time "
                         "panels at the same run fraction); 0 = full run")
    ap.add_argument("--figwidth", type=float, default=16)
    ap.add_argument("--aspect", type=float, default=0.55)
    ap.add_argument("--ymax", type=float, default=4.0,
                    help="y-axis top for the curve figures; the smallest lr "
                         "may sit higher -- raise if it must be visible")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for model in args.models:
        runs = load(args.root, model)
        if not runs:
            print(f"{model}: nothing to plot")
            continue
        write_csvs(runs, args.out, model)
        fig_final(runs, args.out, model)
        fig_curves(runs, args.out, model, args.ymax,
                   args.val_interval, args.time_axis,
                   args.xmin_iter, args.figwidth, args.aspect)


if __name__ == "__main__":
    main()
