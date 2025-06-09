 #!/bin/bash

## dataset | train_mode | ablate_param | with_wandb | seed

sbatch ./bash_scripts/parameter_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 0
sbatch ./bash_scripts/parameter_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 0
sbatch ./bash_scripts/parameter_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 0

sbatch ./bash_scripts/parameter_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 1
sbatch ./bash_scripts/parameter_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 1
sbatch ./bash_scripts/parameter_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 1

sbatch ./bash_scripts/parameter_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 2
sbatch ./bash_scripts/parameter_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 2
sbatch ./bash_scripts/parameter_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment' 'assignment_enc_entropy_reg' 'True' 2

