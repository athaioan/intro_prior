#!/bin/bash
#SBATCH -A  <project-id> 
#SBATCH -t 168:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=intro_prior

source /path/to/your/conda/bin/activate intro_prior

nvidia-smi -l 60 & 
python ./ablation_train_vae.py --dataset $1 --train_mode $2 --ablate_param $3 --with_wandb $4 --seed $5


