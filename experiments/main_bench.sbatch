#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -o outputs/slurm_logs/mainbench_%j.out --open-mode=append
# MAIN EXPERIMENT: GPT-2 large, 1B FineWeb tokens (1905 optimizer steps), four
# arms run SEQUENTIALLY inside one job (same node, so step-time comparisons are
# same-hardware):
#     ns5              Muon, standard Newton-Schulz quintic, lr 0.01
#     polarexpress     Muon, Polar Express schedule, lr 0.01
#     deflated_ns_cut  Deflated Muon (NS), deflation for the whole run, lr 0.01
#     deflated_pe_cut  Deflated Muon (PE), deflation for the whole run, lr 0.01
# All arms share wd 0 and the constant-linear schedule (0.4 warmup, decayed to
# ZERO -- min_lr_frac 0.0).
# args: $1 = gpt_model (default gpt-large)
# The deflated arms run deflation for the WHOLE run (DEFMUON_DEFL_CUTOFF is
# deliberately NOT set -- an early-stage cutoff is only an ablation, experiments/cutoff_ablation.sbatch).  The
# run_hydra_cutoff.py entry point only ADDS the deflated_*_cut methods; the
# baseline arms run through it unchanged.
# DEFMUON_TIME_POLAR=1 records the per-step deflation/iteration split into the
# JSON alongside step_times.  An arm whose output dir already holds a JSON is
# SKIPPED, so a preemption-requeue resumes from the first incomplete arm.
# Figures/CSVs afterwards: python plotting/plot_main_bench.py gpt-large
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
model=${1:-gpt-large}
for pm in ns5 polarexpress deflated_ns_cut deflated_pe_cut; do
  dir="outputs/main_bench/$model-std0.01/$pm-ns5"
  cfg="main_bench"
  arm_overrides="optimizer_params.args.lr=0.01 optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=5"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP $model $pm (already complete)"
    continue
  fi
  echo "################### MAIN $model $pm ###################"
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn $cfg \
    gpt_model=$model training_data=fineweb \
    $arm_overrides \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE $model $pm" || echo "ARM_FAILED $model $pm"
done
echo "MAINBENCH_JOB_DONE $model"
