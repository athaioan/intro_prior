import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["WANDB__SERVICE_WAIT"] = "300"

if '57' in os.path.dirname(os.path.dirname(os.path.realpath(__file__))):
    ## Hacky/ugly/anonymous way to tell if on shared cluster
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from datetime import datetime
datetime.today().strftime('%d-%m-%y')

from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import argparse

from utils import get_configs, augment_intro_configs, augment_configs, get_training_mode

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Train base S-IntroVAE ablation')

    parser.add_argument('--data_root', default='path/to/datasets', type=str, required=False)
    parser.add_argument('--result_root', default='path/to/results/results_intro_prior_images_low', 
                        type=str, required=False)
    parser.add_argument('--train_mode', default='pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment', 
                        choices=get_training_mode(), type=str, required=False)
    
    parser.add_argument('--dataset', default='cifar10', type=str, required=False)
    parser.add_argument('--batch_size', default=128, type=int, required=False)
    parser.add_argument('--num_workers', default=0, type=int, required=False)

    parser.add_argument('--z_dim', default=2, type=int, required=False)
    parser.add_argument('--lr', default=2e-4, type=float, required=False)
    parser.add_argument('--prior_lr', default=2e-4, type=float, required=False)
    parser.add_argument('--num_epochs', default=220, type=int, required=False)
    parser.add_argument('--num_vae', default=20, type=int, required=False)
    parser.add_argument('--recon_loss_type', default='mse', type=str, required=False)
    parser.add_argument('--kl_loss_type', default='stochastic', type=str, required=False)
    parser.add_argument('--beta_kl', default=1.0, type=float, required=False)
    parser.add_argument('--beta_rec', default=1.0, type=float, required=False)
    parser.add_argument('--beta_neg', default=256, type=float, required=False)
    parser.add_argument('--alpha', default=2.0, type=float, required=False)
    parser.add_argument('--gamma_r', default=1e-8, type=float, required=False)
    parser.add_argument('--gamma', default=1, type=float, required=False)
    parser.add_argument('--MC', default=100, type=int, required=False)


    parser.add_argument('--prior_mode', default='vamp', type=str, required=False)

    parser.add_argument('--num_components', default=100, type=int, required=False)
    parser.add_argument('--init_mode', default='data', choices=['random', 'data'], type=str, required=False)
    parser.add_argument('--learnable_contributions', default=True, type=bool, required=False)
    parser.add_argument('--sampling_with_grad', default=True, type=str2bool, required=False)
    parser.add_argument('--intro_prior', default=True, type=str2bool, required=False)
    parser.add_argument('--mog_warmup', default=5, type=int, required=False)
    

    parser.add_argument('--result_iter', default=20, type=int, required=False)
    parser.add_argument('--fid_iter', default=40, type=int, required=False)
    parser.add_argument('--fid_start', default=60, type=int, required=False)

    parser.add_argument('--with_metrics', default='True', type=str, required=False)
    parser.add_argument('--plot_qualitative', default='True', type=str, required=False)
    parser.add_argument('--with_wandb', default='True', type=str, required=False)
    parser.add_argument('--date', default=None, type=str, required=False)

    parser.add_argument('--seed', default=0, type=int, required=False)
    parser.add_argument('--device', default='cuda', type=str, required=False)
    parser.add_argument('--pretrained', default=None, type=str, required=False)
    

    parser.add_argument('--clip_logvar', default=True, type=str, required=False)
    parser.add_argument('--assignment_enc_entropy_reg', default=100, type=float, required=False)


    args = parser.parse_args()
    date = datetime.today().strftime('_%d_%m_%y') if args.date is None else '_{}_{}_{}'.format(*args.date)
    args.result_root = os.path.join(args.result_root, args.train_mode + date)
    args.group_wandb = args.result_root.split('/')[-1]

    args.with_metrics = True if args.with_metrics == 'True' else False
    args.plot_qualitative = True if args.plot_qualitative == 'True' else False
    args.with_wandb = True if args.with_wandb == 'True' else False
    
    train_introspective_vae = get_training_mode(args.train_mode)


    ## Ablation on temperatureKL for KL annealing
    hyperparams = get_configs(datasets=[args.dataset], models=['sIntroVAE'], prior_modes=[args.prior_mode], 
                              num_components=[args.num_components], init_mode=[args.init_mode], learnable_contributions=[args.learnable_contributions], 
                              sampling_with_grad=[args.sampling_with_grad], intro_prior=[args.intro_prior], 
                              intro_with_grad=True)
    hyperparams = augment_configs(hyperparams, vars(args))

    train_introspective_vae(**hyperparams[0])