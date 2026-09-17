#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -o outputs/slurm_logs/nsiters_%j.out --open-mode=append
# ABLATION: stability of the four polar methods in the number of NS iterations.
# Same protocol as the main experiment (gpt-large, lr 0.01, 1B FineWeb tokens =
# 1905 steps, wd 0, constant-linear 0.4-warmup decayed to zero) except ns_steps,
# which sweeps {0, 1, 2, 3, 4, 5}.
# ns_steps=0 is the pure normalized entry (X0, no polynomial iteration); for the
# deflated arms that is the deflation front-end alone.  Deflation runs for the
# whole run (no cutoff), so each run is depth-consistent end to end.
#
# args: $1 = ns_steps (0..5), $2 = gpt_model (default gpt-large)
# JOB PACKING: one job = one ns_steps tier; the four muon arms {ns5,
# polarexpress, deflated_ns_cut, deflated_pe_cut} run SEQUENTIALLY on the SAME node, so
# same-depth method comparisons are same-hardware.  Submit the full sweep with:
#     for k in 0 1 2 3 4 5; do sbatch experiments/ns_iters_ablation.sbatch $k; done
# An arm whose output dir already holds a JSON is SKIPPED, so a
# preemption-requeue resumes from the first incomplete arm.
# Figures/CSVs afterwards: python plotting/plot_ns_iters.py gpt-large
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
k=$1
model=${2:-gpt-large}
if [ -z "$k" ]; then
  echo "usage: sbatch experiments/ns_iters_ablation.sbatch <ns_steps 0..5> [gpt_model]"; exit 1
fi
for pm in ns5 polarexpress deflated_ns_cut deflated_pe_cut; do
  dir="outputs/ns_iters_ablation/$model-std0.01/$pm-ns$k"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP nsiters $model $pm ns=$k (already complete)"
    continue
  fi
  echo "################### NSITERS $model $pm ns_steps=$k ###################"
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb \
    optimizer_params.args.lr=0.01 \
    optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=$k \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE nsiters $model $pm ns=$k" || echo "ARM_FAILED nsiters $model $pm ns=$k"
done
echo "NSITERS_JOB_DONE $model ns=$k"
