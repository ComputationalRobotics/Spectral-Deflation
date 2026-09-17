#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -o outputs/slurm_logs/lrabl_%j.out --open-mode=append
# ABLATION: sensitivity of the four polar methods to the learning rate.
# Same protocol as the main experiment (gpt-large, 1B FineWeb tokens = 1905
# steps, wd 0, ns_steps 5, constant-linear 0.4-warmup decayed to zero) except
# the lr, which sweeps {0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002}.  0.01 is the main-
# experiment lr; its tier deliberately RE-RUNS here rather than reusing
# main_bench data, so all nine tiers share protocol and same-node pairing.
#
#  args: $1 = lr (one of 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002), $2 = gpt_model (default gpt-large)
# JOB PACKING: one job = one lr tier; the four muon arms {ns5, polarexpress,
# deflated_ns_cut, deflated_pe_cut} run SEQUENTIALLY on the SAME node, so same-lr method
# comparisons are same-hardware.  Submit the full sweep with:
#     for lr in 0.01 0.009 0.008 0.007 0.006 0.005 0.004 0.003 0.002; do sbatch experiments/lr_ablation.sbatch $lr; done
# An arm whose output dir already holds a JSON is SKIPPED, so a
# preemption-requeue resumes from the first incomplete arm.
# Figures/CSVs afterwards: python plotting/plot_lr_ablation.py gpt-large
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
lr=$1
model=${2:-gpt-large}
if [ -z "$lr" ]; then
  echo "usage: sbatch experiments/lr_ablation.sbatch <lr> [gpt_model]"; exit 1
fi
for pm in ns5 polarexpress deflated_ns_cut deflated_pe_cut; do
  dir="outputs/lr_ablation/$model-std0.01/$pm-ns5-lr$lr"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP lrabl $model $pm lr=$lr (already complete)"
    continue
  fi
  echo "################### LRABL $model $pm lr=$lr ###################"
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb \
    optimizer_params.args.lr=$lr \
    optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE lrabl $model $pm lr=$lr" || echo "ARM_FAILED lrabl $model $pm lr=$lr"
done
echo "LRABL_JOB_DONE $model lr=$lr"
