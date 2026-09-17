#!/bin/bash
#SBATCH -J track
#SBATCH -p gpu_requeue
#SBATCH --constraint=h200
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 1:30:00
#SBATCH --mem=64G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
# usage: (cd tracking_accuracy && sbatch h200.sbatch <instance> [maxiter])   -- see run.sh
nvidia-smi --query-gpu=name --format=csv,noheader | head -1
bash "$SLURM_SUBMIT_DIR/run.sh" run "$@"
