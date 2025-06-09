# Example Usage

Training S-IntroVAE under a trainable MoG(100)

```
python ./train_vae.py --dataset cifar10 --num_components 100 --intro_prior True --sampling_with_grad True
```

Training S-IntroVAE under a fixed MoG(100)

```
python ./train_vae.py --dataset cifar10 --num_components 100 --intro_prior False --sampling_with_grad False
```

# Replicating the ablation study (i.e., Tab. 2 and Tab. 5)

This guide outlines the steps to replicate the ablation study results presented in Table 2 and Table 5 of the paper.

## Steps to perform the ablation

1. **Run the parameter ablation:**

    ```
    bash ./bash_scripts/submit_parameter_ablation.sh
    ```

2. **Run the evaluation for the ablation:**

    ```
    bash ./bash_scripts/submit_eval_ablation.sh
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
