#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=zero
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:10:10
#SBATCH --output=model_edges_zeros_%A.out


module purge
module load 2022
module load Anaconda3/2022.05

# Your job starts in the directory where you call sbatch
# Activate your environment
source activate dl2023
cd ..
srun python nbody_main.py --zero_edges --d_model 128 --num_heads 4 --num_layers 4 --lr 0.0006 --batch_size 50 --weight_decay 7.04e-06 --num_edges 10
