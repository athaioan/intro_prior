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
from utils.configs import get_dataset_configs, augment_configs_wtih_args
from train_scripts.prior_decoder_cooperation_target_prior_variance_clipping_entropy_encoder_assignment import train_introspective_vae

       

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAE')

    parser.add_argument('--dataset', default='8Gaussian', type=str, required=False)

    parser.add_argument('--model', default='sIntroVAE', type=str, required=False)

    parser.add_argument('--prior_mode', default='vamp', type=str, required=False)
    parser.add_argument('--learnable_contributions', default='True', type=str, required=False)
    parser.add_argument('--intro_prior', default='True', type=str, required=False)
    parser.add_argument('--clip_logvar', default=True, type=bool, required=False)
    parser.add_argument('--num_components', default=64, type=int, required=False)

    parser.add_argument('--gamma', default=1, type=float, required=False) 
    parser.add_argument('--assignment_enc_entropy_reg', default=0, type=float, required=False)

    parser.add_argument('--result_root', default="path/to/results/results_intro_prior_2D_ablation", type=str, required=False)
    parser.add_argument('--result_iter', default=5_000, type=int, required=False)
    parser.add_argument('--plot_qualitative', default='True', type=str, required=False)
    parser.add_argument('--with_wandb', default='True', type=str, required=False)

    parser.add_argument('--seed', default=0, type=int, required=False)

    args = parser.parse_args()

    args.plot_qualitative = True if args.plot_qualitative == 'True' else False
    args.with_wandb = True if args.with_wandb == 'True' else False

    args.learnable_contributions = True if args.learnable_contributions == 'True' else False
    args.intro_prior = True if args.intro_prior == 'True' else False

   
    configs = get_dataset_configs(dataset=args.dataset, model=args.model, prior_mode=args.prior_mode,
                                                       learnable_contributions= args.learnable_contributions, 
                                                       intro_prior=args.intro_prior, 
                                                       num_components=args.num_components)
    
    configs = augment_configs_wtih_args(configs, vars(args))

    pool = Pool(processes=32)  
    for key_index in range(len(configs)):
        
        pool.apply_async(train_introspective_vae, kwds=configs[key_index])    

    pool.close()
    pool.join()






