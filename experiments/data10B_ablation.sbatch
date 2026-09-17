#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -t 48:00:00
#SBATCH -o outputs/slurm_logs/data10B_%j.out --open-mode=append
# ABLATION: the main-experiment comparison at 10x the training data.
# 10B FineWeb tokens (one epoch over the fineweb10B shard dir -- prepare it
# with experiments/prepare_fineweb10B.sbatch; ~19k optimizer steps), gpt-large,
# lr 0.01, wd 0, ns_steps 5.
# One deliberate departure from the 1B protocol, scale-motivated:
#   - lr schedule: constant for the FIRST 5% of steps, then linear decay to
#     zero over the remaining 95% (the 1B protocol holds 40% constant -- at
#     19k steps that would waste most of the budget at peak lr).
# Deflation runs for the whole run (no DEFMUON_DEFL_CUTOFF), as in the 1B
# main experiment.
# Validation every 100 optimizer steps (val_step=400 micro at grad_accum=4);
# the 1B cadence would spend hours on eval at this scale.
# NO checkpoint resume: a requeue restarts an arm from scratch; completed
# arms are skipped.
#
# args: $1 = gpt_model (default gpt-large = the protocol)
#       $2 = muon lr (default 0.01 = the protocol). Another lr is a side study
#            and writes to outputs/data10B_lr<lr>/ instead.
#       $3 = fraction of steps at constant lr before the linear decay (default
#            0.05 = the protocol). Another value is a side study and writes to
#            outputs/data10B_wu<frac>/.  Suffixes combine.
# The four muon arms run SEQUENTIALLY on the SAME node (~13h/arm for
# gpt-large -- the job needs the full 48h).
# Figures/CSVs afterwards (reuses the main-benchmark plot script):
#   python plotting/plot_main_bench.py gpt-large \
#       --root outputs/data10B_ablation --out figures/data10B_ablation
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
model=${1:-gpt-large}
lr=${2:-0.01}
wu=${3:-0.05}   # fraction of steps at constant lr before the linear decay
# lr 0.01 + 5% constant is the protocol; any other lr /
# constant fraction is a side study and gets its own tree (suffixes combine:
# data10B_lr0.005_wu0.1)
root=outputs/data10B_ablation
[ "$lr" != "0.01" -o "$wu" != "0.05" ] && root=outputs/data10B
[ "$lr" != "0.01" ] && root=${root}_lr$lr
[ "$wu" != "0.05" ] && root=${root}_wu$wu

for pm in ns5 polarexpress deflated_ns_cut deflated_pe_cut; do
  dir="$root/$model-std0.01/$pm-ns5"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP data10B $model $pm (already complete)"
    continue
  fi
  echo "################### DATA10B $model lr=$lr wu=$wu $pm ###################"
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb10B \
    optimizer_params.args.lr=$lr \
    lr_schedule.warm_up_fraction=$wu \
    logging_params.val_step=400 \
    optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE data10B $model $pm" || echo "ARM_FAILED data10B $model $pm"
done
echo "DATA10B_JOB_DONE $model lr=$lr wu=$wu"
