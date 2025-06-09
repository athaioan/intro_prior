import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["WANDB__SERVICE_WAIT"] = "300"


from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import argparse
from utils.configs import  augment_configs_wtih_args, get_optimal_configs, augment_config_ablation
from train_scripts.prior_decoder_cooperation_target_prior_variance_clipping_entropy_encoder_assignment import train_introspective_vae

       

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAE')

    parser.add_argument('--dataset', default='8Gaussian', type=str, required=False)
    parser.add_argument('--clip_logvar', default=True, type=bool, required=False)
    parser.add_argument('--gamma', default=1, type=float, required=False) 
    parser.add_argument('--assignment_enc_entropy_reg', default=0, type=float, required=False)

    parser.add_argument('--result_root', default="path/to/results/results_intro_prior_2D_ablation", type=str, required=False)
    parser.add_argument('--result_iter', default=5_000, type=int, required=False)
    parser.add_argument('--plot_qualitative', default='True', type=str, required=False)
    parser.add_argument('--with_wandb', default='False', type=str, required=False)

    args = parser.parse_args()

    args.plot_qualitative = True if args.plot_qualitative == 'True' else False
    args.with_wandb = True if args.with_wandb == 'True' else False

   
    configs = get_optimal_configs(args.result_root, dataset=args.dataset, models = ['VAE', 'sIntroVAE'], prior_modes=['imposed', 'vamp'], 
                                  num_components=64, verbose=True)

    args.result_root = args.result_root + "_ablation"
    configs = augment_configs_wtih_args(configs, vars(args))
    configs = augment_config_ablation(configs, multi_seeds=range(5), num_components=[64])

    pool = Pool(processes=16)  
    for key_index in range(len(configs)):   
        
        pool.apply_async(train_introspective_vae, kwds=configs[key_index])    

    pool.close()
    pool.join()




