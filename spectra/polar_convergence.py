"""Matrix-function accuracy study for the paper (Sec. 3.1 and Appendix F).

For every collected momentum matrix (spectra/collect.py dumps; Q, K, V are
the three row blocks of the c_attn dump), run
the library's four polar methods with their PRODUCTION entries and record, at
every iteration k = 0..5, the projection accuracy

    relative Frobenius error   ||X_k - UV^T||_F / ||UV^T||_F

against the exact polar factor UV^T, taken from an fp64 SVD of the dumped
matrix.  The curve is written in full (k = 0..5, columns rel_err_k0..k5) to
polar_convergence_summary.csv, so the figures can be redrawn or re-tabulated
without re-running anything.

Methods, with their entries exactly as in defmuon/optim:

    ns5            X0 = M/(||M||_F + 1e-7);  quintic (3.4445, -4.7750, 2.0315) x5
                   [muon.zeropower_via_newtonschulz5]
    polarexpress   X0 = M/(1.01*||M||_F + 1e-7);  the safety-factored Polar
                   Express schedule, first 5 polynomials  [polar_express.PolarExpress]
    deflated_ns    deflation_batched.batched_rmfro_clip -- the production batched
                   randomized SVD (Gaussian sketch of ceil(2.5% m) + ceil(2.5% m)
                   oversampling columns, one subspace iteration, CholeskyQR, thin
                   SVD of the reduced matrix), then the production gate/clip
                   (window 2.5% of m, fire ratio 0.10, clip sigma_i > 0.10 sigma_1):
                   remove the ESTIMATED head, renormalize by ||W||_F, restore the
                   head at 1/DIVISOR_PAD; then plain NS5.  Where the gate declines
                   this is exactly the ns5 entry.
    deflated_pe    the SAME front-end (DeflatedPEPolar.PAD restore height, 1.1 by
                   default -- PE's steep early slope amplifies overshoot), then
                   the Polar Express schedule (its 1.01 safety factor lives in
                   the coefficients).  Where the gate declines the entry is
                   M/(||M||_F + 1e-7) -- the production fallback, which is NOT
                   the polarexpress arm's padded entry.

The deflated head is therefore estimated by the same randomized SVD the
optimizer runs, never read off the exact SVD: the study measures the gap the
theory leaves open (estimated components and retuned mappings) on matrices
from real training.  The front-end runs in fp32, as in training.  The
polynomial iteration is evaluated in fp32 by default (--iter-dtype fp32) so
that estimation and mapping effects are not mixed with bf16 rounding, which
every production arm shares; --iter-dtype bf16 reproduces the training
precision instead.  The Gaussian sketch is seeded per matrix, so the study is
exactly reproducible.  Wide matrices are transposed before the front-end and
the polar factor transposed back, polar(M^T) = polar(M)^T, as in muon.py.

Usage:
    python spectra/polar_convergence.py [--dumps DIR] [--out DIR] [--tags ...]
                                        [--iter-dtype fp32|bf16]
"""

import argparse
import csv
import glob
import os
import sys
import zlib

# The iteration kernel is torch.compile-wrapped for training; here it runs eagerly
# (CPU-sized batches, no warm-up to amortize).  Must be set before the import.
os.environ["DEFMUON_COMPILE_DEFL_ITER"] = "0"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from defmuon.optim.deflation_batched import (DIVISOR_PAD, GATE_RATIO,          # noqa: E402
                                             GATE_WINDOW, DEFLATE_THRESH,
                                             SKETCH_FRAC, OVERSAMPLE_FRAC,
                                             RSVD_NITER, DeflatedPEPolar,
                                             _horner_iterate,
                                             batched_rmfro_clip)
from defmuon.optim.polar_express import coeffs_list as PE_COEFFS               # noqa: E402
from plot_spectra import PANEL_COLS, ROW_NAME                                  # noqa: E402

NS_ABC = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5
PE_PAD = DeflatedPEPolar.PAD    # tracks production incl. DEFMUON_DEFL_PE_PAD
SEED = 20260915                 # sketch seed base; per matrix, see sketch_seed()

METHODS = ["ns5", "polarexpress", "deflated_ns", "deflated_pe"]
# Same names / colors / linestyles as plotting/plot_main_bench.py ARMS, so the
# accuracy figures read as the same arms as the benchmark curves.
METHOD_NAME = {"ns5": "Muon (Newton Schulz)",
               "polarexpress": "Muon (Polar Express)",
               "deflated_ns": "Deflated Muon (Newton Schulz)",
               "deflated_pe": "Deflated Muon (Polar Express)"}
METHOD_COLOR = {"ns5": "#7ab0e2", "polarexpress": "#f2a06c",
                "deflated_ns": "#1c5fb0", "deflated_pe": "#d95f02"}
METHOD_LS = {"ns5": (0, (5, 2)), "polarexpress": (0, (1.5, 1.5)),
             "deflated_ns": "-", "deflated_pe": "-"}

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


# ---------------------------------------------------------------------------
# data: the dumped matrices, tall (n >= m), fp32
# ---------------------------------------------------------------------------

def load_matrices(dump_dir):
    """-> {step: {label: A}} with A a tall fp32 tensor (n >= m).  The attn_qkv
    dump is taken apart by rows into attn_q / attn_k / attn_v, exactly as
    plot_spectra.load_tag does; wide matrices are transposed."""
    out = {}
    for path in sorted(glob.glob(os.path.join(dump_dir, "step*_*.pt"))):
        d = torch.load(path, map_location="cpu", weights_only=False)
        A = d["A"].float()
        step, lbl = d["step"], d["label"]
        ent = out.setdefault(step, {})
        blocks = {lbl: A}
        if lbl == "attn_qkv":                       # rows: Q | K | V
            E = A.shape[1]
            blocks = {sub: A[i * E:(i + 1) * E]
                      for i, sub in enumerate(["attn_q", "attn_k", "attn_v"])}
        for name, M in blocks.items():
            if name in PANEL_COLS:
                ent[name] = (M.mT if M.shape[0] < M.shape[1] else M).contiguous()
    return out


def exact_polar(A):
    """UV^T from an fp64 SVD -- the reference every method is measured against."""
    U, _, Vh = torch.linalg.svd(A.double(), full_matrices=False)
    return U @ Vh


def sketch_seed(step, lbl):
    return SEED + 7919 * int(step) + zlib.crc32(lbl.encode()) % 1000


# ---------------------------------------------------------------------------
# the four methods, production entries
# ---------------------------------------------------------------------------

def entry(method, A, seed):
    """-> (X0, k): the method's fp32 entry for the tall matrix A, and the
    deflation width the front-end chose (0 where the gate declined; 0 for the
    plain arms)."""
    fro = A.norm()
    if method == "ns5":
        return A / (fro + 1e-7), 0
    if method == "polarexpress":
        return A / (1.01 * fro + 1e-7), 0
    pad = DIVISOR_PAD if method == "deflated_ns" else PE_PAD
    gen = torch.Generator(device=A.device).manual_seed(seed)
    X0, k, _ = batched_rmfro_clip(A[None], generator=gen, pad=pad)
    return X0[0], int(k[0])


def schedule(method):
    if method in ("polarexpress", "deflated_pe"):
        return PE_COEFFS[:NS_STEPS]
    return [NS_ABC] * NS_STEPS


def rel_err(X, P):
    return float((X.double() - P).norm() / P.norm())


def run(method, A, P, seed, dtype):
    """-> (rel_err[k] for k = 0..NS_STEPS, deflation width k)."""
    X0, k = entry(method, A, seed)
    X = X0.to(dtype)
    rel = [rel_err(X, P)]
    for coef in schedule(method):
        X = _horner_iterate(X, [coef])
        rel.append(rel_err(X, P))
    return rel, k


def compute_all(data, dtype):
    """-> {(step, label): {method: (rel, k)}}, every matrix once."""
    results = {}
    for step in sorted(data):
        for lbl in [c for c in PANEL_COLS if c in data[step]]:
            A = data[step][lbl]
            P = exact_polar(A)
            seed = sketch_seed(step, lbl)
            res = {m: run(m, A, P, seed, dtype) for m in METHODS}
            results[(step, lbl)] = res
            print(f"  step {step:>4} {lbl:9s} {tuple(A.shape)}  "
                  + "  ".join(f"{m}: k5={res[m][0][-1]:.4f}"
                              + (f" (k={res[m][1]})" if m.startswith("deflated") else "")
                              for m in METHODS), flush=True)
    return results


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

YLABEL = r"$\|X_k - UV^\top\|_F / \|UV^\top\|_F$"


def _fit_yaxis(axes):
    """Shared y-range hugging the plotted curves: the error only decreases from
    ~1 at k=0, so the axis runs from just under the lowest curve to just above 1
    instead of a fixed [0, 1.05] that leaves dead space under the curves."""
    lo = min(min(l.get_ydata()) for ax in axes for l in ax.get_lines())
    hi = max(max(l.get_ydata()) for ax in axes for l in ax.get_lines())
    pad = 0.04 * (hi - lo)
    for ax in axes:
        ax.set_ylim(lo - pad, hi + pad)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, axis="y")


def _style(ax, j, title):
    ax.set_xticks(list(range(NS_STEPS + 1)))
    ax.set_title(title, pad=3)
    ax.set_xlabel("iteration $k$", labelpad=1.5)
    if j == 0:
        ax.set_ylabel(YLABEL, fontsize=9)
    else:
        ax.tick_params(labelleft=False)


def _plot(ax, ks, res):
    for meth in METHODS:
        rel, _ = res[meth]
        # markers at the integer iterations: k is discrete, the line is a guide
        ax.plot(ks, rel, color=METHOD_COLOR[meth], linestyle=METHOD_LS[meth],
                marker="o", markersize=3.2, label=METHOD_NAME[meth])


def _finish(fig, axes, out_pdf):
    """Shared data-hugging y-range, arm legend across the top, tight bbox."""
    _fit_yaxis(axes)
    handles = [Line2D([], [], color=METHOD_COLOR[m], linestyle=METHOD_LS[m],
                      marker="o", markersize=3.2)
               for m in METHODS]
    fig.legend(handles, [METHOD_NAME[m] for m in METHODS], ncol=len(METHODS),
               loc="upper center", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.1,
               handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=0.6)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out_pdf}")


def plot_per_matrix(results, lbl, out_pdf):
    """One matrix type across its dump iterations: columns = training steps,
    relative Frobenius error vs NS iteration."""
    steps = sorted({s for (s, l) in results if l == lbl})
    ks = list(range(NS_STEPS + 1))
    # taller panels (aspect ~3.3 for four steps, not the 4.7 of a 2.0in strip):
    # at text width the curves keep readable vertical separation
    fig, axes = plt.subplots(1, len(steps), figsize=(2.25 * len(steps), 2.6),
                             squeeze=False)
    for j, step in enumerate(steps):
        _plot(axes[0][j], ks, results[(step, lbl)])
        _style(axes[0][j], j, f"training step {step}")
    _finish(fig, axes[0], out_pdf)


def plot_per_step(results, step, out_pdf):
    """One training step across the matrix types: columns = matrices."""
    cols = [c for c in PANEL_COLS if (step, c) in results]
    ks = list(range(NS_STEPS + 1))
    fig, axes = plt.subplots(1, len(cols), figsize=(2.1 * len(cols), 2.0),
                             squeeze=False)
    for j, lbl in enumerate(cols):
        _plot(axes[0][j], ks, results[(step, lbl)])
        _style(axes[0][j], j, ROW_NAME[lbl])
    _finish(fig, axes[0], out_pdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="spectra/dumps")
    ap.add_argument("--out", default="figures/spectra")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--iter-dtype", choices=["fp32", "bf16"], default="fp32",
                    help="precision of the polynomial iteration (fp32 isolates "
                         "estimation + mapping effects; bf16 is the training precision)")
    args = ap.parse_args()
    dtype = torch.float32 if args.iter_dtype == "fp32" else torch.bfloat16
    tags = args.tags or sorted(d for d in os.listdir(args.dumps)
                               if os.path.isdir(os.path.join(args.dumps, d)))
    os.makedirs(args.out, exist_ok=True)
    print(f"front-end: sketch {SKETCH_FRAC}+{OVERSAMPLE_FRAC} of m, {RSVD_NITER} subspace "
          f"iteration(s), window {GATE_WINDOW}, ratio {GATE_RATIO}, clip {DEFLATE_THRESH}, "
          f"pad NS {DIVISOR_PAD} / PE {PE_PAD}; iteration in {args.iter_dtype}")
    for tag in tags:
        data = load_matrices(os.path.join(args.dumps, tag))
        print(f"[{tag}] {sum(len(v) for v in data.values())} matrices")
        results = compute_all(data, dtype)
        for lbl in [c for c in PANEL_COLS if any((s, c) in results for s in data)]:
            plot_per_matrix(results, lbl,
                            os.path.join(args.out, f"{tag}_convergence_{lbl}.pdf"))
        csv_rows = []
        for step in sorted(data):
            plot_per_step(results, step, os.path.join(
                args.out, f"{tag}_polar_convergence_step{step}.pdf"))
            for lbl in [c for c in PANEL_COLS if (step, c) in results]:
                for meth in METHODS:
                    rel, k = results[(step, lbl)][meth]
                    row = dict(tag=tag, step=step,
                               matrix=ROW_NAME[lbl].replace("$", ""),
                               method=meth)
                    row.update({f"rel_err_k{i}": f"{v:.6g}"
                                for i, v in enumerate(rel)})
                    row["k_deflated"] = k if meth.startswith("deflated") else ""
                    row["iter_dtype"] = args.iter_dtype
                    csv_rows.append(row)
        tag_csv = os.path.join(args.out, f"{tag}_polar_convergence.csv")
        with open(tag_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"wrote {tag_csv}")

    # combined view, rebuilt from every per-tag file present
    merged = []
    for fn in sorted(os.listdir(args.out)):
        if fn.endswith("_polar_convergence.csv"):
            with open(os.path.join(args.out, fn)) as f:
                merged.extend(csv.DictReader(f))
    if merged:
        out_csv = os.path.join(args.out, "polar_convergence_summary.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
            w.writeheader()
            w.writerows(merged)
        print(f"wrote {out_csv} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
