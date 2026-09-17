"""Paper figures + data for the momentum ablation.

Reads the runs written by experiments/momentum_ablation.sbatch (three muon
polar methods x momentum in {0.90, 0.95, 0.98}, main-experiment protocol
otherwise) and writes, per model:

    <model>_mom_final_val.{pdf,png}         final validation loss vs momentum,
                                            one panel per polar method
                                            (plain + deflated in each)
    <model>_mom<m>_loss.{pdf,png}           per-momentum method comparison,
                                            all four arms: validation then
                                            training loss, each vs iterations
                                            and vs time (combined layout)
    <model>_mom<m>_val_loss.{pdf,png}       same tier, validation loss only,
                                            split by polar method: Newton
                                            Schulz pair | Polar Express pair
    <model>_mom<m>_train_loss.{pdf,png}     same for training loss (smoothed)
    <model>_mom_final_val.csv               arm, momentum, final_val_loss,
                                            total_train_time_s, total_wall_time_s
    <model>_mom_val_data.csv                arm, momentum, step, val_loss,
                                            train_time_s, wall_time_s
    <model>_mom_train_data.csv              arm, momentum, step, loss,
                                            cum_train_time_s

The CSVs hold exactly the raw recorded series (no smoothing, no trimming).
Final val loss is the last recorded eval.  The time axis default is cumulative
training time (eval excluded); pass --time-axis wall for raw wall-clock time.

Usage:
    python plotting/plot_momentum_ablation.py gpt-large
    python plotting/plot_momentum_ablation.py gpt-large --root outputs/momentum_ablation --out figures/momentum_ablation
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
from curve_layout import combined_figure, family_figures, short_label

DEFAULT_ROOT = "outputs/momentum_ablation"
DEFAULT_OUT = "figures/momentum_ablation"
MOM_GRID = ["0.90", "0.95", "0.98"]   # dir-name strings

# (label, run-dir prefix, color, linestyle, polar family) -- colors and
# styles identical to the main figure; the family pairs each plain arm with
# its deflated counterpart and gives the summary figure one panel per method
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
    """-> {mom_str: [(label, json, color, ls), ...]}, missing runs warned."""
    runs = {}
    for mom in MOM_GRID:
        tier = []
        for label, prefix, color, ls, _fam in ARMS:
            hits = sorted(glob.glob(os.path.join(
                root, f"{model}-std0.01", f"{prefix}-ns5-mom{mom}",
                "logs_jobid_*.json")))
            if hits:
                tier.append((label, json.load(open(hits[0])), color, ls))
            else:
                print(f"  [warn] {model}: no run for '{label}' momentum={mom} -- skipped")
        if tier:
            runs[mom] = tier
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


def fig_final(runs, out, model):
    """Final val loss vs momentum, ONE PANEL PER POLAR METHOD: Newton Schulz
    (plain + deflated) left, Polar Express right, so each panel shows the
    deflation gap at every momentum without four curves in one axes.  Shared
    y-axis, so the panels are directly comparable.  The CSV is unsplit and
    still carries all four arms."""
    # collect once per arm, then lay the arms out by family
    series, rows = {}, []
    for label, _, color, ls, fam in ARMS:
        xs, fv = [], []
        for mom in MOM_GRID:
            for lab, j, _, _ in runs.get(mom, []):
                if lab == label:
                    xs.append(float(mom))
                    fv.append(j["val_losses"][-1])
                    rows.append([label, mom, j["val_losses"][-1],
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
                    markersize=3.5, label=short_label(label, fam))
        ax.set_title(fam)
        ax.set_xticks([float(m) for m in MOM_GRID])
        ax.set_xticklabels(MOM_GRID)
        ax.set_xlabel("momentum")
        ax.grid(True, axis="y")
        ax.legend(loc="best")
    axes[0].set_ylabel("final validation loss")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{model}_mom_final_val.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    p = os.path.join(out, f"{model}_mom_final_val.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "momentum", "final_val_loss",
                    "total_train_time_s", "total_wall_time_s"])
        w.writerows(rows)
    print(f"  wrote {out}/{model}_mom_final_val.{{pdf,png,csv}}")


def fig_curves(runs, out, model, ymax, val_interval, time_axis,
               xmin_iter=None, figwidth=16, aspect=0.55):
    """Per tier: the combined figure with all four arms (main-benchmark
    paradigm, see curve_layout) plus a standalone validation and training
    figure with the arms split by polar method -- the same three-figure set
    the main benchmark produces."""
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    time_label = ("wall-clock time (s)" if time_axis == "wall"
                  else "training time (s)")
    fam = {label: f for label, _, _, _, f in ARMS}
    for v, tier in runs.items():
        series = []
        for label, j, color, ls in tier:
            series.append(dict(
                label=label, color=color, ls=ls, family=fam.get(label),
                val_steps=j["val_steps"], val_losses=j["val_losses"],
                val_times=j[tkey], train_losses=j["losses"],
                train_times=np.cumsum(np.asarray(j["step_times"], float))))
        common = dict(xmin_iter=xmin_iter, figwidth=figwidth, aspect=aspect,
                      val_interval=val_interval, train_smooth=TRAIN_SMOOTH,
                      time_label=time_label)
        figs = [(f"mom{v}_loss", combined_figure(series, **common))]
        for kind in ("val", "train"):
            figs += [(f"mom{v}_{slug}_{kind}_loss", fig) for slug, fig in
                     family_figures(series, kind, family_order=FAMILY_ORDER,
                                    **common)]
        for tag, fig in figs:
            if fig is None:
                continue
            for ext in ("pdf", "png"):
                fig.savefig(os.path.join(out, f"{model}_{tag}.{ext}"),
                            dpi=600, bbox_inches="tight")
            plt.close(fig)
        print(f"  wrote {out}/{model}_mom{v}_{{loss,{{ns,pe}}_val_loss,"
              f"{{ns,pe}}_train_loss}}.{{pdf,png}}")


def write_csvs(runs, out, model):
    """Dump exactly the raw recorded series (no smoothing, no trimming)."""
    p = os.path.join(out, f"{model}_mom_val_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "momentum", "step", "val_loss",
                    "train_time_s", "wall_time_s"])
        for mom, tier in runs.items():
            for label, j, _, _ in tier:
                for s, v, tt, wt in zip(j["val_steps"], j["val_losses"],
                                        j["val_train_times"],
                                        j["val_wall_times"]):
                    w.writerow([label, mom, s, v, tt, wt])
    print(f"  wrote {p}")
    p = os.path.join(out, f"{model}_mom_train_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "momentum", "step", "loss", "cum_train_time_s"])
        for mom, tier in runs.items():
            for label, j, _, _ in tier:
                t = np.cumsum(np.asarray(j["step_times"], float))
                for i, (y, tt) in enumerate(zip(j["losses"], t), 1):
                    w.writerow([label, mom, i, y, tt])
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
                    help="y-axis top for the curve figures")
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
