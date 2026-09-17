# Spectral Deflation
Code for **spectral deflation**: remove the few dominant spectral directions of a matrix before an
iterative polynomial filter, let the filter resolve only the better-conditioned bulk, and add the removed
directions back exactly afterwards. The same idea is applied in two settings, and this codebase runs the
experiments in our paper, *Spectral Deflation for Factorization-Free Matrix Filtering in Muon and
Semidefinite Programming*:

- **Deflated Muon** ([`defmuon/`](defmuon/)): deflation in front of Newton–Schulz / Polar Express in the
  Muon optimizer, benchmarked on GPT-2 pretraining (FineWeb). Built on
  [GPT-opt](https://github.com/modichirag/GPT-opt).
- **Deflated ADMM** ([`defadmm/`](defadmm/)): deflation in front of the FP16 PSD-projection filter of
  Kang et al. (arXiv:2507.09165) inside an ADMM solver for Max-Cut / min-bisection SDPs. A self-contained
  CUDA program.

To start, set up the virtual environment and install dependencies by running
```bash
./setup_env.sh
```

which does
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
```
Of course, you can name the virtual environment anything you want.

### Data
The dataloader reads shards of 100M GPT-2 tokens from `$DEFMUON_DATA_DIR` (default `./data`).
To download and tokenize the 1B-token FineWeb set used by the main benchmark:
```bash
python3 experiments/prepare_fineweb.py --shards 11   # 1 val + 10 train shards
```
The 10B-token set for the data ablation is built by `experiments/prepare_fineweb10B.sbatch`.

### Run Example:
```bash
torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=gpt-large \
    optimizer_params.args.polar_method=deflated_ns_cut
```
In the above example, `-cn` specifies the configuration file inside `hydra_conf`, and
`optimizer_params.args.polar_method` picks the arm. All four arms share the same Muon settings
(`hydra_conf/main_bench.yaml`) and differ only in how the polar factor is computed:

| `polar_method` | polar method |
|---|---|
| `ns5` | Newton–Schulz quintic (standard Muon) |
| `polarexpress` | Polar Express |
| `deflated_ns_cut` | deflation + Newton–Schulz (ours) |
| `deflated_pe_cut` | deflation + Polar Express (ours) |

`run_hydra_cutoff.py` is `run_hydra.py` plus the two deflated arms; the baseline arms run through it
unchanged. The deflation front-end is `defmuon/optim/deflation_batched.py` (its knobs are documented at the
top of the file), the optimizer `defmuon/optim/muon.py`.

Each run writes `logs_jobid_<id>.json` into its hydra run directory (loss curves, step times, deflation
gate statistics).

### Plot Results:
```bash
python3 plotting/plot_main_bench.py gpt-large
```
`plotting/plot_*.py` take the run root and output directory as arguments; `plotting/make_figures.sh`
regenerates every figure and CSV under `figures/`.

# On the cluster

### Using Slurm:
The job scripts in `experiments/` run the full benchmark and the ablations. Make sure the `#SBATCH`
partition/GPU lines and the `source .venv/bin/activate` line match your cluster!
```bash
sbatch experiments/main_bench.sbatch gpt-large
```

| script | experiment |
|---|---|
| `main_bench.sbatch` | 1B-token main benchmark, four arms sequentially in one job |
| `ns_iters_ablation.sbatch` | number of Newton–Schulz iterations |
| `lr_ablation.sbatch` | learning rate |
| `momentum_ablation.sbatch` | momentum coefficient |
| `data10B_ablation.sbatch` | 10B-token training set |
| `cutoff_ablation.sbatch` | deflation only for the first `DEFMUON_DEFL_CUTOFF` steps |
| `spectra.sbatch` | spectral study of the momentum matrices (see [`spectra/README.md`](spectra/README.md)) |

Each script documents its arguments in its header. An arm whose output directory already holds a JSON is
skipped, so a requeued job resumes from the first incomplete arm.

### See current jobs
```bash
squeue --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" --me
```

# Paper Plots
The main benchmark is GPT-2 large on 1B FineWeb tokens with 4 GPUs (`experiments/main_bench.sbatch`).
These runs take long, so the deflated arms and the baselines can be split into separate jobs by
trimming the `for pm in ...` loop in the script. Once the JSONs are in `outputs/`, run
```bash
bash plotting/make_figures.sh
```
to reproduce every figure and CSV of the paper under `figures/`.

# Deflated ADMM
[`defadmm/`](defadmm/) does not use the Python package above; it only needs `nvcc` (CUDA 12, sm_90:
H100/H200). It runs the Max-Cut / min-bisection SDP benchmark of Kang et al. on Mittelmann's G55mc,
G59mc, G60mc and G60_mb instances (`defadmm/data/`), comparing the FP16 baseline projection
(`baseline16`) against the deflated one (`deflated16`).
```bash
defadmm/scripts/run.sh build                     # -> defadmm/bin/{admm,pm_head_test}
defadmm/scripts/run.sh bench G55mc both          # baseline16 + deflated16, 10000 iterations
defadmm/scripts/run.sh bench G60_mb deflated16   # one arm
(cd defadmm && sbatch scripts/h200.sbatch bench G55mc both)   # Slurm
```
Results go to `defadmm/results/<instance>_<mode>.txt`; `defadmm/scripts/report.py` and
`defadmm/scripts/plot_convergence.py` produce the table and figures. The full argument list is in
the header of `defadmm/scripts/run.sh`.
