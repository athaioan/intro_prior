 #!/bin/bash



DATASETS=('8Gaussian' 'spiral' 'checkerboard' 'rings')

for dataset in ${DATASETS[@]}; do

    echo "Running grid search for dataset $dataset"

    # VAE
    ## dataset | model | prior_mode | learnable_contributions | intro_prior | with_wandb | seed
    sbatch ./run_grid_search.sh "$dataset" 'VAE' 'imposed' 'False' 'False' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'VAE' 'vamp' 'False' 'False' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'VAE' 'vamp' 'True' 'False' 'False' 0 

    # sIntroVAE
    ## dataset | model | prior_mode | learnable_contributions | intro_prior | with_wandb | seed
    sbatch ./run_grid_search.sh "$dataset" 'sIntroVAE' 'imposed' 'False' 'False' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'sIntroVAE' 'vamp' 'False' 'False' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'sIntroVAE' 'vamp' 'False' 'True' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'sIntroVAE' 'vamp' 'True' 'False' 'False' 0 
    sbatch ./run_grid_search.sh "$dataset" 'sIntroVAE' 'vamp' 'True' 'True' 'False' 0 

done






