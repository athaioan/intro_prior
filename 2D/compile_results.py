import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ["WANDB__SERVICE_WAIT"] = "300"


from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

import numpy as np
import scipy.stats
import argparse
from utils.configs import  augment_configs_wtih_args, get_optimal_configs, augment_config_ablation


def get_stats(experiment_dir, confidence=0.95, interval_se=False, se=False):

    gnelbos = []
    kls = []
    jsds = []
    entropys = []


    multiple_seeds_experiments = os.listdir(experiment_dir)
    multiple_seeds_experiments = [os.path.join(experiment_dir, experiment) for experiment in multiple_seeds_experiments]

    for seed_experiment in multiple_seeds_experiments:

        with open(os.path.join(seed_experiment, 'results_log_soft_intro_vae.txt'), 'r') as f:
            rows = f.readlines()
        
        gnelbo = float(rows[-1].split("gnelbo_")[-1].split("_")[0])
        kl = float(rows[-1].split("kl_")[-1].split("_")[0])
        jsd = float(rows[-1].split("jsd_")[-1].split("_")[0])
        entropy = float(rows[-1].split("entropy_")[-1].split("_")[0])
    
        gnelbos.append(gnelbo)
        kls.append(kl)
        jsds.append(jsd)
        entropys.append(entropy)
    
    gnelbos = np.array(gnelbos) * 1e7
    kls = np.array(kls)
    jsds = np.array(jsds)

    n = len(multiple_seeds_experiments)

    gnelbo_mean, gnelbo_std, gnelbo_se = np.mean(gnelbos), np.std(gnelbos), scipy.stats.sem(gnelbos)
    gnelbo_se = gnelbo_se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    kl_mean, kl_std, kl_se = np.mean(kls), np.std(kls), scipy.stats.sem(kls)
    kl_se = kl_se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    jsd_mean, jsd_std, jsd_se = np.mean(jsds), np.std(jsds), scipy.stats.sem(jsds)
    jsd_se = jsd_se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)

    entropy_mean, entropy_std, entropy_se = np.mean(entropys), np.std(entropys), scipy.stats.sem(entropys)
    entropy_se = entropy_se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    
    average_result = {'gnelbo': {'mean': gnelbo_mean, 'std': gnelbo_std, 'se': gnelbo_se},
                      'kl': {'mean': kl_mean, 'std': kl_std, 'se': kl_se},
                      'jsd': {'mean': jsd_mean, 'std': jsd_std, 'se': jsd_se},
                      'entropy': {'mean': entropy_mean, 'std': entropy_std, 'se': entropy_se},
                      }
    
    result = ''
    for key in average_result.keys():


        if interval_se:
            result +=  ' {}  {:.2f} \u00B1 {:.2f}'.format(key, average_result[key]['mean'], average_result[key]['se'])
        elif se:
            result +=  ' {}  {:.2f} \u00B1 {:.2f}'.format(key, average_result[key]['mean'], average_result[key]['std']/np.sqrt(n))
        else:
            result +=  ' {}  {:.2f} \u00B1 {:.2f}'.format(key, average_result[key]['mean'], average_result[key]['std'])

    print(result,'\n')

    return average_result, result


def get_session_name(**kwargs):

    model_type='VAE' if kwargs.get('num_vae')==kwargs.get('num_iter') else 'sIntroVAE'

    folder_name = '{}/{}/'.format(kwargs.get('dataset'), model_type)

    if model_type=='VAE':

        
        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'),
                                                                                     kwargs.get('learnable_contributions'))

    elif model_type=='sIntroVAE':

        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}_sampleGrad:{}_introPrior:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'), 
                                                                                                                 kwargs.get('learnable_contributions'), 
                                                                                                                 kwargs.get('sampling_with_grad'), kwargs.get('intro_prior'))
        

    session_name = '{}{}'.format(folder_name, prior_name)

    return session_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train VAE')

    parser.add_argument('--dataset', default='spiral', type=str, required=False)
    parser.add_argument('--clip_logvar', default=True, type=bool, required=False)
    parser.add_argument('--gamma', default=1, type=float, required=False) 
    parser.add_argument('--assignment_enc_entropy_reg', default=0, type=float, required=False)

    parser.add_argument('--result_root', default="path/to/results/results_intro_prior_2D_ablation", type=str, required=False)
    parser.add_argument('--result_iter', default=5_000, type=int, required=False)
    parser.add_argument('--plot_qualitative', default='True', type=str, required=False)
    parser.add_argument('--with_wandb', default='True', type=str, required=False)

    args = parser.parse_args()

    args.plot_qualitative = True if args.plot_qualitative == 'True' else False
    args.with_wandb = True if args.with_wandb == 'True' else False


    for args.dataset in ['8Gaussian', 'spiral', 'checkerboard', 'rings']:

        configs = get_optimal_configs(args.result_root, dataset=args.dataset, models = ['VAE','sIntroVAE'], prior_modes=['imposed', 'vamp'], 
                                    num_components=64, verbose=True)


        configs = augment_configs_wtih_args(configs, vars(args))
        configs = augment_config_ablation(configs, multi_seeds=range(1), num_components=[64])
        
        for key_index in range(len(configs)):
            print(get_session_name(**configs[key_index]))

            experiment_seeds = os.path.join(args.result_root, get_session_name(**configs[key_index]))
            average_result, print_output = get_stats(experiment_seeds, interval_se=False, se=True)




