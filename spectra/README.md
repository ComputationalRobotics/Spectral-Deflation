# Section 2.1: spectral study of Muon's momentum matrices

Methodology follows the spectral study of *Spectral Scaling Laws of Muon*
(arXiv:2606.04058, Sec. 3): track the Frobenius-normalized momentum matrix
`A = M/||M||_F` -- exactly the input NS5 sees -- at a fixed relative depth, and
plot its singular-value distribution against the 5-step NS5 transfer map.

## What is collected

Plain **ns5** Muon (unchanged baseline arm), GPT-2 {medium, large}, full
1905-step benchmark budget.  At the mid layer `h.{n_layer//2}` (medium: 12,
large: 18):

- **Dumps** at optimizer steps **{1, 500, 1000, 1500}**: the exact post-Nesterov
  fp32 matrix the polar factorizer consumes, for `attn.c_attn` (dumped whole;
  its three row blocks are Q, K, V), `attn.c_proj` (O), `mlp.c_fc`, `mlp.c_proj`.
  Step 1 has an empty
  momentum buffer, so it is the pure gradient direction ("step 0").
- **Quantiles** every 10 steps for the whole run: sigma_q(A) for
  q in {0.1, 0.25, 0.5, 0.75, 0.9} (full `svdvals`, not the rSVD), into
  `quantiles.json`.

The patch (`collect.py`) is read-only: it computes A out-of-place before the real
step runs, touches no RNG, and saves on rank 0 only -- the training trajectory is
bit-identical to an unpatched ns5 run.  We run the full budget rather than
`max_steps=1500` because capping would move the constant-linear warmup end
(0.4 * total) and change the trajectory; the finished runs double as ns5 replicas.
Do NOT reuse these runs' step timings: the quantile SVDs bill into `step_times`.

## Running

```bash
sbatch experiments/spectra.sbatch gpt-large
```

Dumps land in `spectra/dumps/<tag>/` (~80 MB/step for gpt-large; gitignored).
Knobs: `DEFMUON_SPECTRA_{DIR,TAG,STEPS,QSTEP}` -- see `collect.py`.

## Figures / tables

The offline analysis lives in this folder too: `plot_spectra.py` (spectrum /
post-NS5 / quantile figures) and `polar_convergence.py` (per-iteration
projection accuracy -- relative Frobenius error to the exact polar factor,
||X_k - UV^T||_F / ||UV^T||_F, k = 0..5 -- of ns5 / polarexpress / deflated_ns /
deflated_pe on the dumped matrices).  The four methods use their production
entries: the deflated arms run `deflation_batched.batched_rmfro_clip`, i.e. the
same batched randomized SVD, gate and clip as the optimizer (the head is
*estimated*, never read off an exact SVD; the sketch is seeded per matrix), and
the reference UV^T comes from an fp64 SVD.  The polynomial iteration runs in
fp32 by default so that estimation and mapping effects are not mixed with bf16
rounding, which all production arms share; `--iter-dtype bf16` reproduces the
training precision.

```bash
python spectra/plot_spectra.py
python spectra/polar_convergence.py
```

Both write into `figures/spectra/`:

| file | content |
|---|---|
| `<tag>_spectrum_step<N>.pdf` | the Fig.-6 layout of arXiv:2606.04058: columns Q,K,V,O,MLP in/out; top row the full sigma_i/||M||_F spectrum, bottom row with the largest singular value removed (linear x, log counts, one color per matrix) |
| `<tag>_after.pdf` | histogram of f^(o5)(sigma_i/||M||_F): where singular values land after 5 NS steps |
| `<tag>_quantiles.pdf` | quantile-stabilization curves over training |
| `<tag>_summary.csv` | per (step, matrix): srank, gate decision + k at the production operating point, R = ||M||_F/||W||_F, ||W||_F/sigma_max, implied NS-iteration saving log(R)/log 3.4445 |

Normalization convention: each of Q, K, V is divided by its own Frobenius norm,
the divisor Muon applies to it, matching arXiv:2606.04058.  Full SVDs are computed
offline -- never the production rSVD, whose top-5% view would make the argument
circular.
