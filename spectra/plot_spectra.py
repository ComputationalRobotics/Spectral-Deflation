"""Offline analysis/figures for the Section 2.1 spectral study.

Reads the momentum dumps written by spectra/collect.py, computes FULL SVDs
offline (never the rSVD -- using the front-end's own sketch to justify the
front-end would be circular), and produces:

  <tag>_spectrum_step<N>.pdf  the Fig.-6 layout of arXiv:2606.04058: columns =
                              matrix types (Q, K, V, O, MLP in, MLP out), top row
                              the full spectrum of sigma_i/||M||_F, bottom row the
                              same with the largest singular value removed (the
                              bulk).  Linear x, log counts, one color per matrix.
  <tag>_after.pdf             histogram of f^{o5}(sigma_i/||M||_F): where the
                              singular values land after 5 NS5 steps
  <tag>_quantiles.pdf         quantile-stabilization curves from quantiles.json
  <tag>_summary.csv           per (step, matrix): srank, gate decision at the
                              production operating point (window 2.5% of m, ratio
                              0.10), k, R = ||M||_F/||W||_F, ||W||_F/sigma_max
                              (deflation beats plain spectral normalization iff
                              < 1), and the implied NS-iteration saving
                              log(R)/log(3.4445)

Q, K, V are the three row blocks of the c_attn dump (Q=[0:E], K=[E:2E],
V=[2E:3E]); each is normalized by its own Frobenius norm, the divisor Muon
applies to it, which is also the convention of arXiv:2606.04058.

Usage:
    python spectra/plot_spectra.py [--dumps spectra/dumps] [--out figures/spectra]
                                  [--tags gpt-large ...]
"""

import argparse
import csv
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MaxNLocator, NullFormatter

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# NS5 quintic, exactly muon.zeropower_via_newtonschulz5's coefficients
NS_ABC = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
# production gate operating point (deflation_batched.py defaults)
GATE_WINDOW, GATE_RATIO = 0.025, 0.10
# production constants, imported so figures track the promoted operating point
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
try:
    from defmuon.optim.deflation_batched import DIVISOR_PAD
except ImportError:
    DIVISOR_PAD = 1.01      # restore height is 1/PAD (DEFMUON_DEFL_PAD default)

ROW_ORDER = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_fc", "mlp_proj"]
ROW_NAME = {"attn_q": "Attn $Q$", "attn_k": "Attn $K$", "attn_v": "Attn $V$",
            "attn_qkv": "Attn $QKV$", "attn_o": "Attn $O$",
            "mlp_fc": "MLP in", "mlp_proj": "MLP out"}
# one fixed muted color per matrix type (identity follows the entity)
MAT_COLOR = {"attn_q": "#4C72B0", "attn_k": "#DD8452", "attn_v": "#55A868",
             "attn_qkv": "#64B5CD", "attn_o": "#C44E52",
             "mlp_fc": "#8172B3", "mlp_proj": "#937860"}
# the six matrices Muon projects, each with its own Frobenius entry
PANEL_COLS = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_fc", "mlp_proj"]


def f5(x):
    a, b, c = NS_ABC
    x = np.asarray(x, dtype=np.float64)
    for _ in range(NS_STEPS):
        x = a * x + b * x**3 + c * x**5
    return x


def load_tag(dump_dir):
    """-> {step: {label: dict(s=descending sigmas, den=NS divisor, n, m)}}"""
    out = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "step*_*.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=False)
        A = d["A"].float()
        step, lbl = d["step"], d["label"]
        ent = out.setdefault(step, {})
        fro = A.norm().item()
        s = torch.linalg.svdvals(A).numpy()
        ent[lbl] = dict(s=s, den=fro + 1e-7, n=max(A.shape), m=min(A.shape))
        if lbl == "attn_qkv":                       # rows: Q | K | V
            E = A.shape[1]
            for i, sub in enumerate(["attn_q", "attn_k", "attn_v"]):
                blk = A[i * E:(i + 1) * E]
                den = blk.norm().item() + 1e-7
                ent[sub] = dict(s=torch.linalg.svdvals(blk).numpy(), den=den,
                                n=E, m=E)
    return out


def summarize(data, tag, out_csv):
    rows = []
    for step in sorted(data):
        for lbl in ROW_ORDER:
            if lbl not in data[step]:
                continue
            e = data[step][lbl]
            s, m = e["s"], e["m"]
            fro = float(np.sqrt((s ** 2).sum()))
            srank = fro ** 2 / s[0] ** 2
            kw = min(max(1, math.ceil(GATE_WINDOW * m)), len(s))
            k = int((s > GATE_RATIO * s[0]).sum())
            fired = bool(s[kw - 1] < GATE_RATIO * s[0]) and k < len(s)
            w_fro = float(np.sqrt((s[k:] ** 2).sum())) if fired else fro
            R = fro / max(w_fro, 1e-30)
            rows.append(dict(
                tag=tag, step=step, matrix=ROW_NAME[lbl].replace("$", ""),
                n=e["n"], m=m,
                sigma_max=f"{s[0]:.4g}", fro=f"{fro:.4g}",
                srank=f"{srank:.2f}", x_min=f"{s[-1] / e['den']:.3g}",
                gate_fired=fired, k=k if fired else 0,
                R=f"{R:.3f}", W_over_sigmax=f"{w_fro / s[0]:.3f}",
                iters_saved=f"{math.log(R) / math.log(NS_ABC[0]):.2f}"))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


def _hist(ax, v, color, bins=60):
    edge = tuple(0.55 * c for c in matplotlib.colors.to_rgb(color))
    lo = max(float(v.min()), 1e-8)
    b = np.logspace(math.log10(lo), math.log10(float(v.max()) * 1.1), bins)
    ax.hist(v, bins=b, color=color, edgecolor=edge, linewidth=0.15)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(b[0], b[-1])
    ax.tick_params(labelsize=6.5)


def plot_spectrum_pair(data, tag, step, out_pdf, cols=PANEL_COLS):
    """Full spectrum of the normalized momentum, one panel per projected matrix."""
    cols = [c for c in cols if c in data[step]]
    fig, axes = plt.subplots(1, len(cols), figsize=(1.95 * len(cols), 1.85),
                             squeeze=False)
    for j, lbl in enumerate(cols):
        e = data[step][lbl]
        v = e["s"] / e["den"]
        _hist(axes[0][j], v, MAT_COLOR[lbl])
        axes[0][j].set_title(ROW_NAME[lbl], fontsize=9)
        axes[0][j].set_xlabel(r"$\sigma_i\,/\,\|M\|_F$", fontsize=8)
        if j == 0:
            axes[0][j].set_ylabel("Count", fontsize=8)
    fig.suptitle(f"{tag}, mid layer, step {step}", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def plot_spectrum_iters(data, tag, steps, out_pdf, cols=PANEL_COLS):
    """Standalone: sigma_i/||M||_F distributions, one horizontal row of panels
    (matrix x iteration)."""
    steps = [t for t in steps if t in data]
    panels = [(lbl, t) for lbl in cols for t in steps]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.1 * len(panels), 1.9),
                             squeeze=False)
    for j, (lbl, step) in enumerate(panels):
        ax = axes[0][j]
        e = data[step][lbl]
        _hist(ax, e["s"] / e["den"], MAT_COLOR[lbl])
        ax.set_title(f"{ROW_NAME[lbl]}, Iteration {step}", fontsize=8.5)
        ax.set_xlabel(r"$\sigma_i\,/\,\|M\|_F$", fontsize=8)
        if j == 0:
            ax.set_ylabel("Count", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def plot_ns5_iters(data, tag, steps, out_pdf, cols=PANEL_COLS):
    """Standalone: the spectra after Muon's 5-step NS projection.  Columns =
    (matrix x iteration), matching plot_spectrum_iters; top row the plain
    Frobenius entry, bottom row the deflated remainder (remove every
    sigma_i > 0.1*sigma_1 where the gate fires, renormalize by ||W||_F), so each
    deflated panel sits directly under its undeflated counterpart."""
    steps = [t for t in steps if t in data]
    panels = [(lbl, t) for lbl in cols for t in steps]
    fig, axes = plt.subplots(2, len(panels), figsize=(2.1 * len(panels), 3.4),
                             squeeze=False)
    for j, (lbl, step) in enumerate(panels):
        e = data[step][lbl]
        spec, fired, _ = _deflated_entry(e["s"])
        for i, y in enumerate([f5(e["s"] / e["den"]), f5(spec)]):
            ax = axes[i][j]
            ax.hist(y, bins=np.linspace(0, 1.25, 50), color=MAT_COLOR[lbl],
                    edgecolor=tuple(0.55 * c for c in
                                    matplotlib.colors.to_rgb(MAT_COLOR[lbl])),
                    linewidth=0.15)
            ax.axvline(1.0, color="black", ls=":", lw=0.9)
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0, 1.25)
            ax.tick_params(labelsize=6.5)
            note = f"{100 * float((y < 0.5).mean()):.0f}% < 0.5"
            if i == 1 and not fired:
                note = "gate: off"
            ax.text(0.04, 0.92, note, transform=ax.transAxes, fontsize=6.5,
                    va="top", color="0.25", zorder=5,
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none",
                              pad=1.0))
            if i == 0:
                ax.set_title(f"{ROW_NAME[lbl]}, Iteration {step}", fontsize=8.5)
            else:
                ax.set_xlabel(r"$f^{\circ 5}(\sigma)$", fontsize=8)
            if j == 0:
                ax.set_ylabel(("Deflated" if i else "Original") + "\nCount",
                              fontsize=8)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}")


def plot_after(data, tag, out_pdf, rows):
    steps = sorted(data)
    fig, axes = plt.subplots(len(rows), len(steps),
                             figsize=(2.35 * len(steps), 1.55 * len(rows)),
                             squeeze=False)
    for i, lbl in enumerate(rows):
        for j, step in enumerate(steps):
            ax = axes[i][j]
            e = data[step].get(lbl)
            if e is None:
                ax.set_axis_off()
                continue
            y = f5(e["s"] / e["den"])
            ax.hist(y, bins=np.linspace(0, 1.2, 45), color=MAT_COLOR[lbl],
                    edgecolor="0.25", linewidth=0.25)
            ax.axvline(1.0, color="black", ls=":", lw=0.9)
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0, 1.2)
            frac = float((y < 0.5).mean())
            ax.text(0.03, 0.92, f"{100 * frac:.0f}% < 0.5",
                    transform=ax.transAxes, fontsize=6.5, va="top", color="0.25")
            if i == 0:
                ax.set_title(f"step {step}", fontsize=8)
            if j == 0:
                ax.set_ylabel(ROW_NAME[lbl], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"{tag}: singular values after 5 NS5 steps, "
                 r"$f^{\circ 5}(\sigma_i/\|M\|_F)$ (dotted: target 1)", fontsize=9)
    fig.supxlabel(r"$f^{\circ 5}(\sigma_i/\|M\|_F)$", fontsize=8)
    fig.supylabel("count", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.97))
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def _deflated_entry(s):
    """Deflated-remainder spectrum at the production operating point: remove every
    s_i > 0.1*s_1 (when the gate fires) and Frobenius-normalize the remainder W.
    The restored head is NOT included -- this shows what NS5 does to W alone.
    Returns (spectrum, fired, k)."""
    m = len(s)
    kw = min(max(1, math.ceil(GATE_WINDOW * m)), m)
    k = int((s > GATE_RATIO * s[0]).sum())
    fired = bool(s[kw - 1] < GATE_RATIO * s[0]) and 1 <= k < m
    if not fired:
        return s / (np.sqrt((s ** 2).sum()) + 1e-7), False, 0
    w_fro = np.sqrt((s[k:] ** 2).sum()) + 1e-7
    return s[k:] / w_fro, True, k


def plot_after_deflated(data, tag, out_pdf, rows):
    """f^{o5} of the deflated entry vs the plain Frobenius entry, per panel."""
    steps = sorted(data)
    fig, axes = plt.subplots(len(rows), len(steps),
                             figsize=(2.35 * len(steps), 1.55 * len(rows)),
                             squeeze=False)
    bins = np.linspace(0, 1.25, 50)
    for i, lbl in enumerate(rows):
        for j, step in enumerate(steps):
            ax = axes[i][j]
            e = data[step].get(lbl)
            if e is None:
                ax.set_axis_off()
                continue
            base = f5(e["s"] / e["den"])
            spec, fired, k = _deflated_entry(e["s"])
            defl = f5(spec)
            ax.hist(base, bins=bins, color="0.82", zorder=2,
                    label=r"plain entry $M/\|M\|_F$")
            ax.hist(defl, bins=bins, histtype="step", color=MAT_COLOR[lbl],
                    linewidth=1.2, zorder=3, label="deflated $W/\\|W\\|_F$")
            ax.axvline(1.0, color="black", ls=":", lw=0.9)
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlim(0, 1.25)
            note = (f"{100 * float((base < 0.5).mean()):.0f}% → "
                    f"{100 * float((defl < 0.5).mean()):.0f}% < 0.5"
                    if fired else "gate: off")
            ax.text(0.03, 0.92, note, transform=ax.transAxes, fontsize=6.5,
                    va="top", color="0.25")
            if i == 0:
                ax.set_title(f"step {step}", fontsize=8)
            if j == 0:
                ax.set_ylabel(ROW_NAME[lbl], fontsize=8)
            ax.tick_params(labelsize=6)
            if i == 0 and j == len(steps) - 1:
                ax.legend(fontsize=5.5, frameon=False, loc="upper left",
                          bbox_to_anchor=(0.0, 0.82))
    fig.suptitle(f"{tag}: singular values after 5 NS5 steps -- deflated remainder "
                 r"$W/\|W\|_F$ (remove $\sigma_i > 0.1\,\sigma_1$, renormalize) "
                 "vs plain entry", fontsize=9)
    fig.supxlabel(r"$f^{\circ 5}(\sigma)$", fontsize=8)
    fig.supylabel("count", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.965))
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def plot_quantiles(qpath, tag, out_pdf):
    with open(qpath) as f:
        recs = json.load(f)
    labels = [l for l in ROW_ORDER + ["attn_qkv"]
              if any(r["label"] == l for r in recs)]
    qs = sorted({q for r in recs for q in r["q"]}, key=float)
    shades = [str(g) for g in np.linspace(0.75, 0.0, len(qs))]  # light->dark = q up
    fig, axes = plt.subplots(1, len(labels), figsize=(2.6 * len(labels), 2.2),
                             squeeze=False)
    for j, lbl in enumerate(labels):
        ax = axes[0][j]
        sub = [r for r in recs if r["label"] == lbl]
        stepsq = [r["step"] for r in sub]
        for q, sh in zip(qs, shades):
            ax.plot(stepsq, [r["q"][q] for r in sub], color=sh, lw=1.0,
                    label=f"q={q}")
        ax.set_yscale("log")
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(ROW_NAME[lbl], fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel(r"$\sigma_q(M/\|M\|_F)$", fontsize=8)
        if j == len(labels) - 1:
            ax.legend(fontsize=6, frameon=False, loc="lower right")
    fig.suptitle(f"{tag}: singular-value quantiles of the normalized momentum, "
                 "mid layer", fontsize=9)
    fig.supxlabel("optimizer step", fontsize=8)
    fig.tight_layout(rect=(0.01, 0.03, 1, 0.95))
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="spectra/dumps")
    ap.add_argument("--out", default="figures/spectra")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="default: every subfolder of --dumps")
    args = ap.parse_args()

    tags = args.tags or sorted(
        d for d in os.listdir(args.dumps)
        if os.path.isdir(os.path.join(args.dumps, d)))
    os.makedirs(args.out, exist_ok=True)
    for tag in tags:
        ddir = os.path.join(args.dumps, tag)
        data = load_tag(ddir)
        if data:
            summarize(data, tag, os.path.join(args.out, f"{tag}_summary.csv"))
            for step in sorted(data):
                plot_spectrum_pair(data, tag, step,
                                   os.path.join(args.out,
                                                f"{tag}_spectrum_step{step}.pdf"))
            if all(t in data for t in (1, 1000)):
                iter_cols = ["attn_v", "mlp_fc"]
                plot_spectrum_iters(data, tag, [1, 1000],
                                    os.path.join(args.out,
                                                 f"{tag}_spectrum_iter1_1000.pdf"),
                                    cols=iter_cols)
                plot_ns5_iters(data, tag, [1, 1000],
                               os.path.join(args.out,
                                            f"{tag}_ns5_iter1_1000.pdf"),
                               cols=iter_cols)
            plot_after(data, tag, os.path.join(args.out, f"{tag}_after.pdf"),
                       PANEL_COLS)
            plot_after_deflated(data, tag,
                                os.path.join(args.out, f"{tag}_after_deflated.pdf"),
                                PANEL_COLS)
        qpath = os.path.join(ddir, "quantiles.json")
        if os.path.exists(qpath):
            plot_quantiles(qpath, tag, os.path.join(args.out,
                                                    f"{tag}_quantiles.pdf"))
        if not data and not os.path.exists(qpath):
            print(f"{tag}: nothing to plot in {ddir}")


if __name__ == "__main__":
    main()
