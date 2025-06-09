#!/bin/bash

#SBATCH -A  <project-id> 
#SBATCH -t 168:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=intro_prior_2D

source /path/to/your/conda/bin/activate intro_prior

nvidia-smi -l 60 & 

python ./demo_grid_search.py --dataset $1 --model $2 --prior_mode $3 --learnable_contributions $4 --intro_prior $5 --with_wandb $6 --seed $7


