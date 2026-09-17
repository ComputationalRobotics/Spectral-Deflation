"""Paper figures + data for the deflation-cutoff ablation.

Reads the runs written by experiments/cutoff_ablation.sbatch (both deflated
arms x cutoff in {10, 30, 50, 100}% of the step budget, main-experiment
protocol otherwise; 100% = full-run deflation) plus the same-protocol Muon
baselines (ns5, polarexpress) from the main experiment, and writes, per model:

    <model>_cut_final_val.{pdf,png}          final validation loss vs cutoff %,
                                             one line per deflated arm, with the
                                             Muon (NS) / Muon (PE) finals as
                                             horizontal reference lines
    <model>_cut<C>_vs_muon_loss.{pdf,png}    both deflated arms at cutoff C%
                                             (default 10) against the two plain
                                             Muon arms, in the main-benchmark
                                             combined layout (curve_layout.py):
                                             val | val-vs-time | train |
                                             train-vs-time, x from --xmin-iter
    <model>_cut_<arm>_val_loss.{pdf,png}     per-arm validation loss, one line
                                             per cutoff -- left vs iterations,
                                             right vs wall time
    <model>_cut_<arm>_train_loss.{pdf,png}   same for training loss (smoothed)
    <model>_cut_final_val.csv                arm, cutoff_pct, final_val_loss,
                                             total_train_time_s, total_wall_time_s
    <model>_cut_val_data.csv                 arm, cutoff_pct, step, val_loss,
                                             train_time_s, wall_time_s
    <model>_cut_train_data.csv               arm, cutoff_pct, step, loss,
                                             cum_train_time_s

The plain Muon baselines have no cutoff; they appear in the CSVs with
cutoff_pct = "none".  They are read from --baseline-root (default
outputs/main_bench), i.e. the main-experiment runs -- NOT re-run here, so they
were trained on a different node than the cutoff runs (same protocol, same
seed; run-to-run drift from non-deterministic kernels is ~0.005 at 1B).  The
CSVs hold exactly the raw recorded series (no smoothing, no trimming).  The
time axis default is cumulative training time (eval excluded); pass
--time-axis wall for raw wall-clock time.

Usage:
    python plotting/plot_cutoff_ablation.py gpt-large
    python plotting/plot_cutoff_ablation.py gpt-large --compare-cutoff 30
    python plotting/plot_cutoff_ablation.py gpt-large --xmin-iter 1000 --figwidth 16 --aspect 0.55
    python plotting/plot_cutoff_ablation.py gpt-large --root outputs/cutoff_ablation --out figures/cutoff_ablation
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

DEFAULT_ROOT = "outputs/cutoff_ablation"
DEFAULT_BASELINE_ROOT = "outputs/main_bench"
DEFAULT_OUT = "figures/cutoff_ablation"
CUT_GRID = [10, 30, 50, 100]   # % of the step budget

# (label, run-dir prefix, color, linestyle, filename slug)
ARMS = [
    ("Deflated Muon (Newton Schulz)", "deflated_ns_cut", "#1c5fb0", "-", "ns"),
    ("Deflated Muon (Polar Express)", "deflated_pe_cut", "#d95f02",
     "-", "pe"),
]
# Same-protocol plain-Muon baselines from the main experiment; colors and
# linestyles match plotting/plot_main_bench.py so the two figure sets agree.
# (label, run-dir name, color, linestyle)
BASELINES = [
    ("Muon (Newton Schulz)", "ns5-ns5",          "#7ab0e2", (0, (5, 2))),
    ("Muon (Polar Express)", "polarexpress-ns5", "#f2a06c", (0, (1.5, 1.5))),
]
TRAIN_SMOOTH = 100             # steps; the raw series is drawn faintly underneath

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

# one color per cutoff, shared by both curve figures (dark = larger cutoff)
CUT_CMAP = plt.get_cmap("viridis")
CUT_COLOR = {c: CUT_CMAP(0.9 - 0.85 * i / (len(CUT_GRID) - 1))
             for i, c in enumerate(CUT_GRID)}


def load(root, model):
    """-> {arm_label: {cutoff_pct: run_json}}, missing runs warned."""
    runs = {}
    for label, prefix, _, _, _ in ARMS:
        per_c = {}
        for c in CUT_GRID:
            hits = sorted(glob.glob(os.path.join(
                root, f"{model}-std0.01", f"{prefix}-ns5-cut{c}",
                "logs_jobid_*.json")))
            if hits:
                per_c[c] = json.load(open(hits[0]))
            else:
                print(f"  [warn] {model}: no run for '{label}' cutoff={c}% -- skipped")
        if per_c:
            runs[label] = per_c
    return runs


def load_baselines(root, model):
    """-> {label: run_json} for the plain-Muon main-experiment arms."""
    base = {}
    for label, name, _, _ in BASELINES:
        hits = sorted(glob.glob(os.path.join(
            root, f"{model}-std0.01", name, "logs_jobid_*.json")))
        if hits:
            base[label] = json.load(open(hits[0]))
        else:
            print(f"  [warn] {model}: no baseline run for '{label}' "
                  f"under {root} -- skipped")
    return base


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


def fig_final(runs, base, out, model):
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for label, _, color, ls, _ in ARMS:
        per_c = runs.get(label, {})
        cs = sorted(per_c)
        if not cs:
            continue
        fv = [per_c[c]["val_losses"][-1] for c in cs]
        ax.plot(cs, fv, color=color, linestyle=ls, marker="o", markersize=3.5,
                label=label)
    # plain-Muon finals: no cutoff, so a horizontal reference line each
    for label, _, color, ls in BASELINES:
        if label in base:
            ax.axhline(base[label]["val_losses"][-1], color=color,
                       linestyle=ls, label=label)
    ax.set_xticks(CUT_GRID)
    ax.set_xlabel("deflation cutoff (% of steps)")
    ax.set_ylabel("final validation loss")
    ax.grid(True, axis="y")
    # headroom above the highest line so the legend never covers a curve
    ys = [per_c[c]["val_losses"][-1] for per_c in runs.values() for c in per_c]
    ys += [j["val_losses"][-1] for j in base.values()]
    lo, hi = min(ys), max(ys)
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.75 * (hi - lo))
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{model}_cut_final_val.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    p = os.path.join(out, f"{model}_cut_final_val.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cutoff_pct", "final_val_loss",
                    "total_train_time_s", "total_wall_time_s"])
        for label, per_c in runs.items():
            for c in sorted(per_c):
                j = per_c[c]
                w.writerow([label, c, j["val_losses"][-1],
                            float(np.sum(j["step_times"])),
                            j["val_wall_times"][-1]])
        for label, j in base.items():
            w.writerow([label, "none", j["val_losses"][-1],
                        float(np.sum(j["step_times"])),
                        j["val_wall_times"][-1]])
    print(f"  wrote {out}/{model}_cut_final_val.{{pdf,png,csv}}")


def _val_series(j, tkey, val_interval):
    s = np.asarray(j["val_steps"], float)
    v = np.asarray(j["val_losses"], float)
    t = np.asarray(j[tkey], float)
    m = (s % val_interval) == 0
    return s[m], v[m], t[m]


def _train_series(j):
    y = np.asarray(j["losses"], float)
    it = np.arange(1, len(y) + 1)
    t = np.cumsum(np.asarray(j["step_times"], float))
    return it, y, t


def fig_curves(runs, out, model, ymax, val_interval, time_axis):
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    slug = {label: sl for label, _, _, _, sl in ARMS}
    for label, per_c in runs.items():
        # validation loss, one line per cutoff
        fig, ax_i, ax_t = _panels("validation loss", time_axis)
        for c in sorted(per_c):
            s, v, t = _val_series(per_c[c], tkey, val_interval)
            ax_i.plot(s, v, color=CUT_COLOR[c], label=f"cutoff {c}%")
            ax_t.plot(t, v, color=CUT_COLOR[c], label=f"cutoff {c}%")
        lo = min(np.asarray(j["val_losses"], float).min()
                 for j in per_c.values())
        ax_i.set_ylim(lo - 0.02, ymax)
        ax_i.set_title(label)
        ax_i.legend(loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                out, f"{model}_cut_{slug[label]}_val_loss.{ext}"),
                dpi=600, bbox_inches="tight")
        plt.close(fig)
        # training loss, one line per cutoff (smoothed over raw)
        fig, ax_i, ax_t = _panels("training loss", time_axis)
        for c in sorted(per_c):
            it, y, t = _train_series(per_c[c])
            ys = smooth(y, TRAIN_SMOOTH)
            off = TRAIN_SMOOTH - 1
            for a, x in ((ax_i, it), (ax_t, t)):
                a.plot(x, y, color=CUT_COLOR[c], linewidth=0.4, alpha=0.08)
                a.plot(x[off:], ys, color=CUT_COLOR[c], label=f"cutoff {c}%")
        lo = min(smooth(np.asarray(j["losses"], float), TRAIN_SMOOTH).min()
                 for j in per_c.values())
        ax_i.set_ylim(lo - 0.02, ymax)
        ax_i.set_title(label)
        ax_i.legend(loc="lower left")
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                out, f"{model}_cut_{slug[label]}_train_loss.{ext}"),
                dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}/{model}_cut_{slug[label]}_{{val,train}}_loss.{{pdf,png}}")


def fig_compare(runs, base, out, model, cut, val_interval, time_axis,
                xmin_iter=None, figwidth=16, aspect=0.55):
    """Both deflated arms at ONE cutoff vs the two plain-Muon arms, in the
    main-benchmark combined layout (see curve_layout.combined_figure)."""
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    # baselines first so the deflated (solid) curves draw on top
    arms = [(label, base[label], color, ls)
            for label, _, color, ls in BASELINES if label in base]
    arms += [(f"{label}, cutoff {cut}%", runs[label][cut], color, ls)
             for label, _, color, ls, _ in ARMS
             if label in runs and cut in runs[label]]
    if len(arms) < 2:
        print(f"  [warn] {model}: fewer than two arms for the cutoff-{cut}% "
              f"comparison -- skipped")
        return
    series = [dict(label=label, color=color, ls=ls,
                   val_steps=j["val_steps"], val_losses=j["val_losses"],
                   val_times=j[tkey], train_losses=j["losses"],
                   train_times=np.cumsum(np.asarray(j["step_times"], float)))
              for label, j, color, ls in arms]
    fig = combined_figure(
        series, xmin_iter=xmin_iter, figwidth=figwidth, aspect=aspect,
        val_interval=val_interval, train_smooth=TRAIN_SMOOTH,
        time_label=("wall-clock time (s)" if time_axis == "wall"
                    else "training time (s)"),
        tag=f"deflation cutoff = {cut}%")
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{model}_cut{cut}_vs_muon_loss.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}/{model}_cut{cut}_vs_muon_loss.{{pdf,png}}")


def write_csvs(runs, base, out, model):
    """Dump exactly the raw recorded series (no smoothing, no trimming)."""
    rows = [(label, c, per_c[c]) for label, per_c in runs.items()
            for c in sorted(per_c)]
    rows += [(label, "none", j) for label, j in base.items()]
    p = os.path.join(out, f"{model}_cut_val_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cutoff_pct", "step", "val_loss",
                    "train_time_s", "wall_time_s"])
        for label, c, j in rows:
            for s, v, tt, wt in zip(j["val_steps"], j["val_losses"],
                                    j["val_train_times"], j["val_wall_times"]):
                w.writerow([label, c, s, v, tt, wt])
    print(f"  wrote {p}")
    p = os.path.join(out, f"{model}_cut_train_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "cutoff_pct", "step", "loss", "cum_train_time_s"])
        for label, c, j in rows:
            t = np.cumsum(np.asarray(j["step_times"], float))
            for i, (y, tt) in enumerate(zip(j["losses"], t), 1):
                w.writerow([label, c, i, y, tt])
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="gpt-large")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--baseline-root", default=DEFAULT_BASELINE_ROOT,
                    help="main-experiment root holding the plain-Muon runs")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--time-axis", choices=["train", "wall"], default="train")
    ap.add_argument("--val-interval", type=int, default=100)
    ap.add_argument("--ymax", type=float, default=4.0,
                    help="y-axis top for the curve figures")
    ap.add_argument("--compare-cutoff", type=int, default=10,
                    choices=CUT_GRID,
                    help="cutoff %% shown in the deflated-vs-Muon comparison")
    ap.add_argument("--xmin-iter", type=float, default=1000,
                    help="comparison figure: start the x-axes at this "
                         "optimizer step (time panels at the same fraction); "
                         "0 = full run")
    ap.add_argument("--figwidth", type=float, default=16,
                    help="comparison figure width in inches")
    ap.add_argument("--aspect", type=float, default=0.55,
                    help="comparison figure panel height/width ratio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for model in args.models:
        runs = load(args.root, model)
        if not runs:
            print(f"{model}: nothing to plot")
            continue
        base = load_baselines(args.baseline_root, model)
        write_csvs(runs, base, args.out, model)
        fig_final(runs, base, args.out, model)
        fig_curves(runs, args.out, model, args.ymax,
                   args.val_interval, args.time_axis)
        fig_compare(runs, base, args.out, model, args.compare_cutoff,
                    args.val_interval, args.time_axis, args.xmin_iter,
                    args.figwidth, args.aspect)


if __name__ == "__main__":
    main()
