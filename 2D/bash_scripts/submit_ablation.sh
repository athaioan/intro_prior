 #!/bin/bash

## dataset | with_wandb

DATASETS=('8Gaussian' 'spiral' 'checkerboard' 'rings')

for dataset in ${DATASETS[@]}; do

    echo "Running grid search for dataset $dataset"

    sbatch ./run_ablation.sh "$dataset" 'False'

done






