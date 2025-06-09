#!/bin/bash
#SBATCH -A  <project-id> 
#SBATCH -t 24:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --job-name=eval_intro_prior

source /path/to/your/conda/bin/activate intro_prior

nvidia-smi -l 60 & 
python ../ablation_eval_vae.py --dataset $1 --eval_mode $2 --ablate_param $3 --seed $4


