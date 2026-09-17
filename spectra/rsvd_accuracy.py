"""Accuracy of the batched randomized SVD head estimate (Appendix D, Table 3).

For every recorded momentum matrix (spectra/dumps, loaded exactly as in
polar_convergence.py) the production front-end (defmuon/optim/deflation_batched._rsvd_b:
Gaussian sketch of ceil(2.5% m) + ceil(2.5% m) columns, CholeskyQR, one subspace
iteration, thin SVD of the reduced matrix) is run in fp32 for several sketch seeds, and
the k deflated singular values -- k being the width chosen by the production gate -- are
compared with the leading k singular values of an fp64 SVD:

    sv_err = max_{i<=k} |s_i - sigma_i| / sigma_i

Matrices on which the gate does not fire are skipped.  Nothing in the training or the
matrix-function-accuracy pipeline is touched.

Usage: python spectra/rsvd_accuracy.py [--dumps DIR] [--out DIR] [--seeds 20]
Writes <out>/rsvd_accuracy.csv (one row per matrix and seed) and
<out>/rsvd_accuracy_table.tex (median / min / max per momentum matrix).
"""
import argparse
import csv
import math
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from polar_convergence import load_matrices, sketch_seed, PANEL_COLS                   # noqa: E402
from defmuon.optim.deflation_batched import (_rsvd_b, RSVD_NITER, SKETCH_FRAC,          # noqa: E402
                                             OVERSAMPLE_FRAC, GATE_WINDOW, GATE_RATIO,
                                             DEFLATE_THRESH)

ROWS = [("attn_q", "Attention $\\mathrm{Q}$"), ("attn_k", "Attention $\\mathrm{K}$"),
        ("attn_v", "Attention $\\mathrm{V}$"), ("attn_o", "Attention $\\mathrm{O}$"),
        ("mlp_fc", "MLP (first)"), ("mlp_proj", "MLP (second)")]


def one(A, se, seed):
    """-> (fired, k, sv_err) for one sketch of the tall fp32 matrix A with exact
    singular values se (fp64, descending)."""
    n, m = A.shape
    r = min(math.ceil(SKETCH_FRAC * m) + math.ceil(OVERSAMPLE_FRAC * m), m)
    kw = min(max(1, math.ceil(GATE_WINDOW * m)), r)
    gen = torch.Generator().manual_seed(seed)
    Omega = torch.randn(1, m, r, dtype=A.dtype, generator=gen)
    _, s, _ = _rsvd_b(A[None], Omega, RSVD_NITER)
    s = s[0].double()
    k = int((s > DEFLATE_THRESH * s[0]).sum())
    fired = bool(s[kw - 1] < GATE_RATIO * s[0]) and 1 <= k < r
    if not fired:
        return False, 0, float("nan")
    return True, k, float(((s[:k] - se[:k]).abs() / se[:k]).max())


def write_table(csv_path, out_path):
    d = pd.read_csv(csv_path)
    d = d[d.fired == 1]
    fmt = lambda x: f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e")
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Momentum matrix & median & min & max \\\\", "\\midrule"]
    for key, label in ROWS:
        v = d[d.label == key].sv_err
        if len(v):
            lines.append(f"{label} & {fmt(v.median())} & {fmt(v.min())} & {fmt(v.max())} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(out_path, "w").write("\n".join(lines) + "\n")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=os.path.join(os.path.dirname(__file__), "dumps", "gpt-large"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "rsvd_accuracy"))
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--table-only", action="store_true", help="rebuild the table from the existing csv")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "rsvd_accuracy.csv")
    if not args.table_only:
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        data = load_matrices(args.dumps)
        rows = []
        for step in sorted(data):
            for lbl in [c for c in PANEL_COLS if c in data[step]]:
                A = data[step][lbl]
                se = torch.linalg.svdvals(A.double())
                for sidx in range(args.seeds):
                    fired, k, err = one(A, se, sketch_seed(step, lbl) + 100003 * sidx)
                    rows.append(dict(step=step, label=lbl, seed=sidx, fired=int(fired), k=k, sv_err=err))
                fr = [r["sv_err"] for r in rows if r["step"] == step and r["label"] == lbl and r["fired"]]
                print(f"step {step:>4} {lbl:9s} {tuple(A.shape)}  fired {len(fr)}/{args.seeds}"
                      + (f"  median sv_err {pd.Series(fr).median():.2e}" if fr else ""), flush=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "label", "seed", "fired", "k", "sv_err"])
            w.writeheader(); w.writerows(rows)
        print("wrote", csv_path)
    write_table(csv_path, os.path.join(args.out, "rsvd_accuracy_table.tex"))


if __name__ == "__main__":
    main()
