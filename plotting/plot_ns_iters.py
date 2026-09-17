"""Paper figures + data for the NS-iterations ablation.

Reads the runs written by experiments/ns_iters_ablation.sbatch (three muon
polar methods x ns_steps in {0..5}, main-experiment protocol otherwise) and
writes, per model:

    <model>_ns_final_val.{pdf,png}          final validation loss vs ns_steps,
                                            one panel per polar method
                                            (plain + deflated in each)
    <model>_ns_<arm>_val_loss.{pdf,png}     per-method validation-loss curves,
                                            one line per depth -- left vs
                                            iterations, right vs wall time
    <model>_ns_<arm>_train_loss.{pdf,png}   same for training loss (smoothed)
    <model>_ns_final_val.csv                arm, ns_steps, final_val_loss,
                                            total_train_time_s, total_wall_time_s
    <model>_ns_val_data.csv                 arm, ns_steps, step, val_loss,
                                            train_time_s, wall_time_s
    <model>_ns_train_data.csv               arm, ns_steps, step, loss,
                                            cum_train_time_s

The CSVs hold exactly the raw recorded series (no smoothing, no trimming), so
every paper number is re-derivable without re-parsing the JSONs.  Final val
loss is the last recorded eval (the end-of-run eval at step 1905).  The time
axis default is cumulative training time (eval excluded); pass
--time-axis wall for raw wall-clock time.

Usage:
    python plotting/plot_ns_iters.py gpt-large
    python plotting/plot_ns_iters.py gpt-large --root outputs/ns_iters_ablation --out figures/ns_iters_ablation
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

DEFAULT_ROOT = "outputs/ns_iters_ablation"
DEFAULT_OUT = "figures/ns_iters_ablation"
NS_GRID = list(range(6))   # ns_steps 0..5

# (label, run-dir prefix, method color for the summary figure, linestyle,
#  polar family) -- the family pairs each plain arm with its deflated
# counterpart and gives the summary figure one panel per method
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

# one color per depth, shared by every per-method figure (dark = deep)
DEPTH_CMAP = plt.get_cmap("viridis")
DEPTH_COLOR = {k: DEPTH_CMAP(0.9 - 0.85 * k / max(NS_GRID)) for k in NS_GRID}


def load(root, model):
    """-> {label: {ns_steps: run_json}}, missing runs warned and skipped."""
    runs = {}
    for label, prefix, _, _, _fam in ARMS:
        per_k = {}
        for k in NS_GRID:
            hits = sorted(glob.glob(os.path.join(
                root, f"{model}-std0.01", f"{prefix}-ns{k}",
                "logs_jobid_*.json")))
            if hits:
                per_k[k] = json.load(open(hits[0]))
            else:
                print(f"  [warn] {model}: no run for '{label}' ns_steps={k} -- skipped")
        if per_k:
            runs[label] = per_k
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
    """Final val loss vs ns_steps, ONE PANEL PER POLAR METHOD: Newton Schulz
    (plain + deflated) left, Polar Express right, so each panel shows the
    deflation gap at every depth without four curves in one axes.  Shared
    y-axis, so the panels are directly comparable.  The CSV is unsplit and
    still carries all four arms."""
    families = [(fam, [a for a in ARMS if a[4] == fam and runs.get(a[0])])
                for fam in FAMILY_ORDER]
    families = [(fam, arms) for fam, arms in families if arms]
    if not families:
        print(f"  [warn] {model}: no arm loaded -- final-val figure skipped")
        return
    # panel size matches the old single-panel figure (3.6 x 3.2 in), so the
    # two-panel version is simply twice as wide
    fig, axes = plt.subplots(1, len(families), figsize=(3.3 * len(families), 3.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    rows = []
    for ax, (fam, arms) in zip(axes, families):
        for label, _, color, ls, _f in arms:
            per_k = runs[label]
            ks = sorted(per_k)
            fv = [per_k[k]["val_losses"][-1] for k in ks]
            ax.plot(ks, fv, color=color, linestyle=ls, marker="o",
                    markersize=3.5, label=_short(label, fam))
            for k in ks:
                j = per_k[k]
                rows.append([label, k, j["val_losses"][-1],
                             float(np.sum(j["step_times"])),
                             j["val_wall_times"][-1]])
        ax.set_title(fam)
        ax.set_xlabel("NS iterations")
        ax.set_xticks(NS_GRID)
        ax.grid(True, axis="y")
        ax.legend(loc="best")
    axes[0].set_ylabel("final validation loss")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{model}_ns_final_val.{ext}"),
                    dpi=600, bbox_inches="tight")
    plt.close(fig)
    p = os.path.join(out, f"{model}_ns_final_val.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ns_steps", "final_val_loss",
                    "total_train_time_s", "total_wall_time_s"])
        w.writerows(rows)
    print(f"  wrote {out}/{model}_ns_final_val.{{pdf,png,csv}}")


def _slug(label):
    return {"Muon (Newton Schulz)": "ns5", "Muon (Polar Express)": "polarexpress",
            "Deflated Muon (Newton Schulz)": "deflated",
            "Deflated Muon (Polar Express)": "deflated_pe"}[label]


def fig_curves(runs, out, model, ymax, val_interval, time_axis):
    tkey = "val_wall_times" if time_axis == "wall" else "val_train_times"
    for label, _, _, _, _fam in ARMS:
        per_k = runs.get(label, {})
        if not per_k:
            continue
        # validation curves, one line per depth
        fig, ax_i, ax_t = _panels("validation loss", time_axis)
        for k in sorted(per_k):
            j = per_k[k]
            s = np.asarray(j["val_steps"], float)
            v = np.asarray(j["val_losses"], float)
            t = np.asarray(j[tkey], float)
            m = (s % val_interval) == 0
            ax_i.plot(s[m], v[m], color=DEPTH_COLOR[k], label=f"$k={k}$")
            ax_t.plot(t[m], v[m], color=DEPTH_COLOR[k], label=f"$k={k}$")
        lo = min(np.asarray(j["val_losses"], float).min()
                 for j in per_k.values())
        ax_i.set_ylim(lo - 0.02, ymax)
        ax_i.set_title(label)
        ax_i.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                out, f"{model}_ns_{_slug(label)}_val_loss.{ext}"),
                dpi=600, bbox_inches="tight")
        plt.close(fig)
        # training curves, one line per depth (smoothed over raw)
        fig, ax_i, ax_t = _panels("training loss", time_axis)
        for k in sorted(per_k):
            j = per_k[k]
            y = np.asarray(j["losses"], float)
            it = np.arange(1, len(y) + 1)
            t = np.cumsum(np.asarray(j["step_times"], float))
            ys = smooth(y, TRAIN_SMOOTH)
            off = TRAIN_SMOOTH - 1
            for a, x in ((ax_i, it), (ax_t, t)):
                a.plot(x, y, color=DEPTH_COLOR[k], linewidth=0.4, alpha=0.08)
                a.plot(x[off:], ys, color=DEPTH_COLOR[k], label=f"$k={k}$")
        lo = min(smooth(np.asarray(j["losses"], float), TRAIN_SMOOTH).min()
                 for j in per_k.values())
        ax_i.set_ylim(lo - 0.02, ymax)
        ax_i.set_title(label)
        ax_i.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(
                out, f"{model}_ns_{_slug(label)}_train_loss.{ext}"),
                dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}/{model}_ns_{_slug(label)}_{{val,train}}_loss.{{pdf,png}}")


def write_csvs(runs, out, model):
    """Dump exactly the raw recorded series (no smoothing, no trimming)."""
    p = os.path.join(out, f"{model}_ns_val_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ns_steps", "step", "val_loss",
                    "train_time_s", "wall_time_s"])
        for label, per_k in runs.items():
            for k in sorted(per_k):
                j = per_k[k]
                for s, v, tt, wt in zip(j["val_steps"], j["val_losses"],
                                        j["val_train_times"],
                                        j["val_wall_times"]):
                    w.writerow([label, k, s, v, tt, wt])
    print(f"  wrote {p}")
    p = os.path.join(out, f"{model}_ns_train_data.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "ns_steps", "step", "loss", "cum_train_time_s"])
        for label, per_k in runs.items():
            for k in sorted(per_k):
                j = per_k[k]
                t = np.cumsum(np.asarray(j["step_times"], float))
                for i, (y, tt) in enumerate(zip(j["losses"], t), 1):
                    w.writerow([label, k, i, y, tt])
    print(f"  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="gpt-large")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--time-axis", choices=["train", "wall"], default="train")
    ap.add_argument("--val-interval", type=int, default=100)
    ap.add_argument("--ymax", type=float, default=4.0,
                    help="y-axis top for the curve figures; shallow depths may "
                         "sit far above it -- raise if k=0/1 must be visible")
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
                   args.val_interval, args.time_axis)


if __name__ == "__main__":
    main()
