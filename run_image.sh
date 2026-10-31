#!/bin/bash

#SBATCH --job-name=SpatialCellTNBCImages  # Replace with your desired job name
#SBATCH --account=mok439@o2.hms.harvard.edu  # Replace with your Slurm account (if applicable)
#SBATCH --partition=short  # Replace with appropriate partition for GPUs
#SBATCH --gres=gpu:1  # Request 1 GPU
#SBATCH --cpus-per-task=4  # Request 8 CPU cores
#SBATCH --mem=4GB  # Request 64GB of memory
#SBATCH --time=8:00:00  # Request 5 days of walltime
#SBATCH -o hostname_%j.out
#SBATCH --mail-type=ALL  # Email notification for job events (optional)
#SBATCH --mail-user=mohammad_khan@o2.hms.harvard.com  # Email address for notifications (optional)

module load gcc/9.2.0
module load cuda/12.1
module load miniconda3/23.1.0

# Activate your conda environment
source activate SpatialCellTNBC  # Replace with your conda environment name

/n/cluster/bin/job_gpu_monitor.sh &

# Run your Python script with GPU support
python run_image.py --configs/config_images.yaml # Replace with your script name and arguments (if applicable)

# Deactivate conda environment (optional)
source deactivate
