#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -o outputs/slurm_logs/momabl_%j.out --open-mode=append
# ABLATION: sensitivity of the four polar methods to the momentum coefficient.
# Same protocol as the main experiment (gpt-large, lr 0.01, 1B FineWeb tokens =
# 1905 steps, wd 0, ns_steps 5, Nesterov, constant-linear 0.4-warmup decayed to
# zero) except momentum, which sweeps {0.90, 0.95, 0.98}.  0.95 is the main-
# experiment value; its tier deliberately RE-RUNS here rather than reusing
# main_bench data, so all three points share protocol and same-node pairing.
#
# args: $1 = momentum (0.90 | 0.95 | 0.98), $2 = gpt_model (default gpt-large)
# JOB PACKING: one job = one momentum tier; the four muon arms {ns5,
# polarexpress, deflated_ns_cut, deflated_pe_cut} run SEQUENTIALLY on the SAME node, so
# same-momentum method comparisons are same-hardware.  Submit the sweep with:
#     for m in 0.90 0.95 0.98; do sbatch experiments/momentum_ablation.sbatch $m; done
# An arm whose output dir already holds a JSON is SKIPPED, so a
# preemption-requeue resumes from the first incomplete arm.
# Figures/CSVs afterwards: python plotting/plot_momentum_ablation.py gpt-large
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
mom=$1
model=${2:-gpt-large}
if [ -z "$mom" ]; then
  echo "usage: sbatch experiments/momentum_ablation.sbatch <momentum> [gpt_model]"; exit 1
fi
for pm in ns5 polarexpress deflated_ns_cut deflated_pe_cut; do
  dir="outputs/momentum_ablation/$model-std0.01/$pm-ns5-mom$mom"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP momabl $model $pm mom=$mom (already complete)"
    continue
  fi
  echo "################### MOMABL $model $pm momentum=$mom ###################"
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb \
    optimizer_params.args.lr=0.01 optimizer_params.args.momentum=$mom \
    optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE momabl $model $pm mom=$mom" || echo "ARM_FAILED momabl $model $pm mom=$mom"
done
echo "MOMABL_JOB_DONE $model mom=$mom"
