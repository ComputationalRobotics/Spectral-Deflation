#!/bin/bash
#SBATCH -p seas_gpu,gpu_h200 --gres=gpu:nvidia_h200:4 -c 48 --mem=400G --requeue
#SBATCH -t 24:00:00
#SBATCH -o outputs/slurm_logs/gate_%j.out --open-mode=append
# ABLATION: robustness of Deflated Muon in the two gate constants of
# defmuon/optim/deflation_batched.py,
#     w   = GATE_WINDOW  (DEFMUON_DEFL_WINDOW; SKETCH_FRAC follows it, so the sketch is 2wm)
#     tau = GATE_RATIO   (DEFMUON_DEFL_RATIO;  DEFLATE_THRESH follows it, so clip == gate)
# One-dimensional sweeps around the main-experiment operating point (w, tau) = (0.025, 0.1):
#     w   in {0.0025, 0.005, 0.01, 0.025}   at tau = 0.1
#     tau in {0.05, 0.1, 0.2, 0.3}          at w   = 0.025
# Same protocol as the main experiment otherwise (gpt-large, lr 0.01, 1B FineWeb
# tokens = 1905 steps, wd 0, constant-linear 0.4-warmup decayed to zero, 5
# iterations, deflation for the whole run, no DEFMUON_DEFL_CUTOFF).
#
# args: $1 = deflated arm, deflated_ns_cut (Newton-Schulz) or deflated_pe_cut (Polar
#            Express); $2 = gpt_model (default gpt-large)
# JOB PACKING: one job = one mapping; the seven (w, tau) settings, INCLUDING the
# default point as a same-hardware reference, run SEQUENTIALLY on the same node, so
# every comparison within a mapping is same-hardware.  Submit both mappings with:
#     sbatch experiments/gate_ablation.sbatch deflated_ns_cut
#     sbatch experiments/gate_ablation.sbatch deflated_pe_cut
# To continue a mapping on the node its earlier arms ran on, pin the node on the
# command line (both H200 nodes used so far are in seas_gpu only):
#     sbatch -p seas_gpu -w <node> [--dependency=afterany:<jobid>] experiments/gate_ablation.sbatch <arm>
# A setting whose output dir already holds a JSON is SKIPPED, so a
# preemption-requeue (or a pinned follow-up job) resumes from the first incomplete
# setting.  The gate statistics (fire rate, mean deflated width) land in
# gate_stats / gate_series of the JSON.
source /n/netscratch/hankyang_lab/Everyone/haoransun/prismenv/bin/activate
export DEFMUON_DATA_DIR=${DEFMUON_DATA_DIR:-/n/netscratch/hankyang_lab/Everyone/haoransun/gptopt_data}
command -v torchrun >/dev/null || { echo "torchrun not found after activating prismenv"; exit 1; }
[ -d "$DEFMUON_DATA_DIR/fineweb1B-gpt2" ] || { echo "missing $DEFMUON_DATA_DIR/fineweb1B-gpt2"; exit 1; }
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
pm=$1
model=${2:-gpt-large}
case "$pm" in
  deflated_ns_cut|deflated_pe_cut) ;;
  *) echo "usage: sbatch experiments/gate_ablation.sbatch <deflated_ns_cut|deflated_pe_cut> [gpt_model]"; exit 1 ;;
esac
echo "GATE_JOB_START $model $pm on $(hostname)"
# "w tau" pairs: default first, then the w sweep, then the tau sweep
settings=(
  "0.025 0.1"
  "0.01 0.1" "0.005 0.1" "0.0025 0.1"
  "0.025 0.05" "0.025 0.2" "0.025 0.3"
)
for st in "${settings[@]}"; do
  read -r w tau <<< "$st"
  dir="outputs/gate_ablation/$model-std0.01/$pm-w$w-tau$tau"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "ARM_SKIP gate $model $pm w=$w tau=$tau (already complete)"
    continue
  fi
  echo "################### GATE $model $pm w=$w tau=$tau ###################"
  DEFMUON_DEFL_WINDOW=$w DEFMUON_DEFL_RATIO=$tau \
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb \
    optimizer_params.args.lr=0.01 \
    optimizer_params.args.polar_method=$pm optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "ARM_DONE gate $model $pm w=$w tau=$tau" || echo "ARM_FAILED gate $model $pm w=$w tau=$tau"
done
echo "GATE_JOB_DONE $model $pm"
