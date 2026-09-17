#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -o outputs/slurm_logs/spectra_%j.out --open-mode=append
# Section 2.1 spectral-collection run.  args: $1 = gpt_model
# Plain ns5 Muon over the FULL 1905-step benchmark budget (max_steps would move
# the lr-warmup end and change the trajectory), with spectra/collect.py dumping
# the mid-layer momentum matrices at steps {1, 500, 1000, 1500} and tracking
# singular-value quantiles every 10 steps.  Step timing of these runs is NOT
# comparable to main_bench (the quantile SVDs bill into step_times).
# A run whose output dir already holds a JSON is SKIPPED (requeue-resumable).
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_SPECTRA_TAG=$1
mkdir -p outputs/slurm_logs
dir="outputs/spectra/$1-ns5"
if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
  echo "SPECTRA_SKIP $1 (already complete)"
else
  echo "################### SPECTRA $1 ns5 ###################"
  torchrun --standalone --nproc_per_node=4 spectra/run_hydra_spectra.py -cn main_bench \
    gpt_model=$1 training_data=fineweb \
    optimizer_params.args.polar_method=ns5 optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "SPECTRA_DONE $1" || echo "SPECTRA_FAILED $1"
fi
echo "SPECTRA_JOB_DONE $1"
