#!/bin/bash
#SBATCH -p shared -c 48 --mem=200G -t 12:00:00
#SBATCH -o outputs/slurm_logs/prep10B_%j.out --open-mode=append
# Tokenize the 10B-token FineWeb set for the 10B-scale ablation (data10B_ablation):
# 101 shards of 100M GPT-2 tokens (shard 0 = val, 100 train shards = 10B
# training tokens) into $DEFMUON_DATA_DIR/fineweb10B-gpt2/ (~20GB on disk).
# CPU-only job.
#
# Adjust the partition line and the environment activation to your cluster.
source .venv/bin/activate
mkdir -p outputs/slurm_logs
python experiments/prepare_fineweb.py --shards 101 --name fineweb10B
echo "PREP10B_DONE"
