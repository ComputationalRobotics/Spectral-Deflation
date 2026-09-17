#!/bin/bash
#SBATCH -p gpu --gres=gpu:4 -c 48 --mem=400G --requeue
#SBATCH -t 10:30:00
#SBATCH -o outputs/slurm_logs/cutabl_%j.out --open-mode=append
# ABLATION: effect of the early-stage deflation cutoff on the deflated arm.
# Same protocol as the main experiment (gpt-large, lr 0.01, 1B FineWeb tokens =
# 1905 steps, wd 0, ns_steps 5, constant-linear 0.4-warmup decayed to zero);
# only the deflated_ns_cut arm runs -- the cutoff is a deflation-only knob, the
# ns5 / polarexpress arms have no analogue (their same-protocol
# baselines are the main-experiment runs).
# Cutoff sweeps {10, 30, 50, 100}% of the step budget: deflation runs for the
# first DEFMUON_DEFL_CUTOFF steps, then the factorizer hard-switches to the
# plain path.  100% never switches = full-run deflation = the main-experiment
# protocol.  This
# is the ONLY experiment that sets DEFMUON_DEFL_CUTOFF.
#
# args: $1 = gpt_model (default gpt-large)
# ALL FOUR cutoffs run SEQUENTIALLY inside this ONE job on the SAME node --
# the cutoff comparison itself is the deliverable, so it must be same-hardware.
# A run whose output dir already holds a JSON is SKIPPED, so a
# preemption-requeue resumes from the first incomplete run.
# Figures/CSVs afterwards: python plotting/plot_cutoff_ablation.py gpt-large
#
# Adjust the partition/gres lines and the environment activation to your cluster.
source .venv/bin/activate
export OMP_NUM_THREADS=8
export DEFMUON_TIME_POLAR=1
mkdir -p outputs/slurm_logs
model=${1:-gpt-large}

# The true step budget, from the train-shard headers (token counts at int32
# offset 2; tokens per optimizer step = 2^19) -- 1905 for the 1B set.
TOTAL_STEPS=$(python - <<'PYEOF'
import glob, os, torch
root = os.environ.get("DEFMUON_DATA_DIR", "./data")
tok = sum(int(torch.from_file(f, False, 3, dtype=torch.int32)[2])
          for f in glob.glob(os.path.join(root, "fineweb1B-gpt2", "*train_*.bin")))
print(tok // 524288)
PYEOF
)
if [ -z "$TOTAL_STEPS" ] || [ "$TOTAL_STEPS" -eq 0 ]; then
  echo "fineweb1B shards missing or empty -- run experiments/prepare_fineweb.py first"; exit 1
fi

for arm in deflated_ns_cut deflated_pe_cut; do
for pct in 10 30 50 100; do
  dir="outputs/cutoff_ablation/$model-std0.01/$arm-ns5-cut$pct"
  if ls "$dir"/logs_jobid_*.json >/dev/null 2>&1; then
    echo "RUN_SKIP cutabl $model $arm cut=$pct% (already complete)"
    continue
  fi
  cutoff=$((TOTAL_STEPS * pct / 100))
  echo "################### CUTABL $model $arm cutoff=$pct% ($cutoff/$TOTAL_STEPS steps) ###################"
  DEFMUON_DEFL_CUTOFF=$cutoff \
  torchrun --standalone --nproc_per_node=4 run_hydra_cutoff.py -cn main_bench \
    gpt_model=$model training_data=fineweb \
    optimizer_params.args.lr=0.01 \
    optimizer_params.args.polar_method=$arm optimizer_params.args.ns_steps=5 \
    hydra.run.dir="$dir" \
  && echo "RUN_DONE cutabl $model $arm cut=$pct%" || echo "RUN_FAILED cutabl $model $arm cut=$pct%"
done
done
echo "CUTABL_JOB_DONE $model"
