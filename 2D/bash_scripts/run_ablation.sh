#!/bin/bash

#SBATCH -A  <project-id> 
#SBATCH -t 168:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=intro_prior_2D

source /path/to/your/conda/bin/activate intro_prior

nvidia-smi -l 60 & 

python ./demo_ablation.py --dataset $1 --with_wandb $2


