 #!/bin/bash

## dataset | train_mode | ablate_param | seed

sbatch ./bash_scripts/eval_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 0
sbatch ./bash_scripts/eval_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 0
sbatch ./bash_scripts/eval_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 0

sbatch ./bash_scripts/eval_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 1
sbatch ./bash_scripts/eval_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 1
sbatch ./bash_scripts/eval_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 1

sbatch ./bash_scripts/eval_ablation.sh 'cifar10' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 2
sbatch ./bash_scripts/eval_ablation.sh 'fmnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 2
sbatch ./bash_scripts/eval_ablation.sh 'mnist' 'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment' 'assignment_enc_entropy_reg' 2
