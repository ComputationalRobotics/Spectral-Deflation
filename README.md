# Spectral Deflation

Code for **spectral deflation**: a gated, sketch-based front-end that removes the few
dominant spectral directions of a matrix before an iterative polynomial filter, lets the
filter resolve only the remaining, better-conditioned bulk, and restores the removed
directions exactly afterwards. The same rule is applied to two iterations:

| application | matrix function | filter it precedes | code |
|---|---|---|---|
| **Deflated Muon** | polar factor of the momentum matrix (LLM pretraining) | Newton–Schulz / Polar Express | [`defmuon/`](defmuon/) |
| **Deflated ADMM** | PSD projection inside ADMM for SDP (Max-Cut / min-bisection) | the FP16 factorization-free composite filter of Kang et al. (arXiv:2507.09165) | [`defadmm/`](defadmm/) |

## Layout

```
defmuon/                 Deflated Muon: GPT-2 model, data pipeline, distributed training loop
defmuon/optim/           Muon optimizer, Polar Express, the deflation front-end (deflation_batched.py)
                         and the optional early-stage cutoff (deflation_cutoff.py)
run_hydra.py             training entry point (hydra); run_hydra_cutoff.py adds the deflated arms
hydra_conf/              configs: main_bench.yaml, gpt_model/, training_data/
experiments/             Slurm job scripts for the benchmark and ablations, FineWeb tokenization
plotting/                figure / CSV generation from the run JSONs
spectra/                 spectral study of Muon's momentum matrices (collection + analysis)
defadmm/                 Deflated ADMM: CUDA solver, data, scripts, results (self-contained)
```

---

## Deflated Muon (`defmuon/`)

Benchmarks deflation as a preconditioner for Muon's polar factorization on GPT-2
pretraining (FineWeb). Four polar-method arms share identical optimizer settings:

| arm | polar method |
|---|---|
| `ns5` | Muon with the standard Newton–Schulz quintic (baseline) |
| `polarexpress` | Muon with the Polar Express schedule (baseline) |
| `deflated_ns_cut` | deflation front-end + Newton–Schulz (ours) |
| `deflated_pe_cut` | deflation front-end + Polar Express (ours) |

The deflated arms run deflation for the whole run. Setting `DEFMUON_DEFL_CUTOFF=N`
(used only by the cutoff ablation) switches to the plain path after N optimizer steps;
the `_cut` suffix marks the arms that accept this switch. The front-end lives in
`defmuon/optim/deflation_batched.py`, the optimizer in `defmuon/optim/muon.py`.

### Setup

```bash
./setup_env.sh          # creates .venv and pip-installs the package
source .venv/bin/activate
```

### Data

Shards are 100M GPT-2 tokens in a simple binary format under `$DEFMUON_DATA_DIR`
(default `./data`). To build the 1B-token FineWeb dataset:

```bash
python experiments/prepare_fineweb.py --shards 11   # 1 val + 10 train shards
```

The 10B-token split is built by `experiments/prepare_fineweb10B.sbatch`.

### Running

Single run (4 GPUs, one arm):

```bash
torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=gpt-large \
    optimizer_params.args.polar_method=deflated_ns_cut
```

`-cn` selects a config from `hydra_conf/`; `gpt_model` is `gpt-large` (the only model config).

Job scripts in `experiments/` run the full benchmark and the ablations; each documents
its arguments and submit loop in its header:

| script | experiment |
|---|---|
| `main_bench.sbatch` | 1B-token main benchmark, four arms |
| `ns_iters_ablation.sbatch` | number of Newton–Schulz iterations |
| `lr_ablation.sbatch` | learning rate |
| `momentum_ablation.sbatch` | momentum coefficient |
| `data10B_ablation.sbatch` | 10B-token training set |
| `cutoff_ablation.sbatch` | early-stage deflation cutoff |
| `spectra.sbatch` | spectral collection (see `spectra/README.md`) |

Adjust the `#SBATCH` headers and environment activation to your cluster.

### Outputs and figures

Each run writes `logs_jobid_<id>.json` into its hydra run directory (loss curves, step
times, deflation gate statistics). `plotting/make_figures.sh` regenerates every figure
and CSV from these JSONs into `figures/`; the individual `plotting/plot_*.py` scripts
take the run root and output directory as arguments. Deflation knobs are documented at
the top of `defmuon/optim/deflation_batched.py`.



## Deflated ADMM (`defadmm/`)

Spectral deflation as a preconditioner for the PSD-cone projection inside a
factorization-free SDP solver: the FP16 composite filter of Kang et al. in a three-block
ADMM, run entirely in FP16 on Mittelmann's G55mc, G59mc, G60mc and G60_mb instances.
CUDA (sm_90) only; nothing here imports the Python package above.

The folder is self-contained: `src/` holds the ADMM driver (modes `baseline32`,
`baseline16`, `deflated16`) together with Kang et al.'s reference filter, vendored
unchanged; `data/` the four instances in a compact format; `scripts/` the build/run
script, the Slurm wrapper, the SDPA converter and the report/plot scripts; `results/`
the benchmark logs (documented in `defadmm/results/README.md`); `test/` a synthetic
single-projection test.

In `deflated16` mode each projection tracks the dominant eigenspace of the ADMM iterate
with a warm-started block subspace step, gates on the Ritz magnitudes, removes the
clipped head, pads it at ±1/γ, runs the unchanged FP16 composite on the remainder and
adds the positive head back exactly. The output is then symmetrized,
`P <- (P + P^T)/2`, since the reconstruction is symmetric only up to rounding.

### Run

```bash
defadmm/scripts/run.sh build                     # nvcc (CUDA 12, sm_90) -> defadmm/bin/{admm,pm_head_test}
defadmm/scripts/run.sh bench G55mc both          # baseline16 + deflated16, 10000 iterations
defadmm/scripts/run.sh bench G60_mb deflated16   # one arm
(cd defadmm && sbatch scripts/h200.sbatch bench G55mc both)   # Slurm
defadmm/scripts/report.py; defadmm/scripts/plot_convergence.py
```

Driver arguments: `admm_mc <problem.sdp> <mode> <maxiter> <sigma> [eta_tol=1e-4] [log_every=25]
[window=0.025] [gamma=1.1] [symmetrize=1]`. Runs are reproducible: the deflation block seed is
fixed in the source and `run.sh` sets `DEFADMM_LANCZOS_SEED` for the Lanczos scale estimate.
