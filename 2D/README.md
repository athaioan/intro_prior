# Replicating the ablation study (i.e., Tab. 1)

This guide outlines the steps to replicate the ablation study results presented in Table 1 of the paper.

## Steps to perform the ablation

1. **Run the parameter grid search:**

    ```
    bash ./bash_scripts/submit_grid_search.sh
    ```

2. **Run the parameter ablation**

    ```
    bash ./bash_scripts/submit_ablation.sh
    ```

3. **Compile and summarize the results:**

    ```
    python compile_results.py
    ```

## Code layout
- `bash_scripts` - The bash scripts used to carry out the ablations
- `models` - Defines the models.
- `dataset` - Dataloading.
- `train_scripts` - The python scripts used to train the VAEs.
- `eval_scripts` - The python scripts used to evaluate the VAEs.
- `utils` - The utilities used in the codebase.
- `metrics` - The function used for evaluating the metrics.
