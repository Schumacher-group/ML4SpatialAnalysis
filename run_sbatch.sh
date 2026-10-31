#!/bin/bash
#SBATCH --job-name=SpatialCellTNBC
#SBATCH --account=sander  # Replace with your Slurm account (if applicable)
#SBATCH --partition=gpu_quad        # GPU-Quad partition
#SBATCH --gres=gpu:teslaV100s:1        # one Quadro RTX6000
#SBATCH --cpus-per-task=4           # 4 CPU cores
#SBATCH --mem=16G                   # 16 GB RAM
#SBATCH --time=48:00:00             # 48 hours

# launch interactive shell
srun --pty bash -i
