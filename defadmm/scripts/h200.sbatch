#!/bin/bash
#SBATCH -J defadmm
#SBATCH -p gpu_requeue
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 0:20:00
#SBATCH --mem=64G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
# usage: sbatch scripts/h200.sbatch <phase> [args]   (see scripts/run.sh)
# The binaries are built for sm_90; --constraint=h200 keeps the job off A100 nodes.
# baseline32 alone needs ~15 min of the 20; run it in its own job.
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
bash "$SLURM_SUBMIT_DIR/scripts/run.sh" "$@"
