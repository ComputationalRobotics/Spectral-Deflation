# Results

Benchmark results (Kang et al. Sec. 4.2 instances, FP16 throughout, 10000 iterations, sigma = 1):
`<instance>_<mode>.txt` for `G55mc`, `G59mc`, `G60mc`, `G60_mb` x `baseline16`, `deflated16`.

All files: Mittelmann's G55mc, G59mc, G60mc, G60_mb; sigma = 1; 10000 ADMM iterations; FP16 throughout; one NVIDIA H200;
`FINAL` = a-posteriori lambda_min(X), lambda_min(S), rank(X).
Runs of 2026-09-16; the projection output is symmetrized, `P <- (P + P^T)/2` (`symmetrize=1`), and the
Lanczos start vectors are seeded with `DEFADMM_LANCZOS_SEED=20260916`, so a run is reproducible.
The two trailing log columns are the symmetry defect `||P-P^T||_F/||P||_F` before and after symmetrization.

| file | produced by |
|---|---|
| `<instance>_baseline16.txt` | the reference filter (Kang et al.), FP16 |
| `<instance>_deflated16.txt` | Deflated ADMM, window w = 0.025, gamma = 1.1 (the runs reported in the paper) |
| `synthetic_pm_head.txt` | `test/pm_head_test` (baseline vs deflated, exact eigenvectors) |

All logs are produced by `src/admm_mc.cu` of this repository (`scripts/run.sh bench <instance> both`).

