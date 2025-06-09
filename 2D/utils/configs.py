import copy
import os 
import numpy as np



## Generic Configs Utils ##

def augment_configs_wtih_args(hyperparams, args):

    for index in hyperparams:  
        for key in args.keys():
            if key not in hyperparams[index].keys():
                hyperparams[index][key] = args[key]

    return hyperparams

def get_config(model='VAE', dataset='8Gaussian', prior_mode='imposed', num_components=None, init_mode='data', 
               learnable_contributions=False, sampling_with_grad=False, intro_prior=False):

    # Defining network configs
    # optimal configurations provided by https://arxiv.org/abs/2012.13253
    configs = {
                          "8Gaussian": 
                                        { "VAE" : {'dataset': '8Gaussian', 'num_iter': 30_000, 'num_vae': 30_000, 'beta_rec': 0.8, 'beta_kl': 0.05, 'beta_neg': None, 'lr': 2e-4, 
                                                   'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                   'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior
                                                   },

                                          "sIntroVAE":  {'dataset': '8Gaussian', 'num_iter': 30_000, 'num_vae': 2_000, 'beta_rec': 0.2, 'beta_kl': 0.3, 'beta_neg': 0.9, 'lr': 2e-4,
                                                        'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                        'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior}
                                        },

                            "spiral": 
                                { "VAE" : {'dataset': 'spiral', 'num_iter': 30_000, 'num_vae': 30_000, 'beta_rec': 1, 'beta_kl': 0.05, 'beta_neg': None, 'lr': 2e-4, 
                                           'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                            'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior,
                                            },

                                  "sIntroVAE":  {'dataset': 'spiral', 'num_iter': 30_000, 'num_vae': 2_000, 'beta_rec': 0.2, 'beta_kl': 0.5, 'beta_neg': 1.0, 'lr': 2e-4,
                                                 'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                 'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior}
                                },
                                    
                                    
                            "checkerboard": 
                                { "VAE" : {'dataset': 'checkerboard', 'num_iter': 30_000, 'num_vae': 30_000, 'beta_rec': 0.8, 'beta_kl': 0.1, 'beta_neg': None, 'lr': 2e-4,
                                           'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                           'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior,
                                             },

                                  "sIntroVAE":  {'dataset': 'checkerboard', 'num_iter': 30_000, 'num_vae':  2_000, 'beta_rec': 0.2, 'beta_kl': 0.1, 'beta_neg': 0.2, 'lr': 2e-4, 
                                                'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior}
                                },


                            "rings": 
                                { "VAE" : {'dataset': 'rings', 'num_iter': 30_000, 'num_vae': 30_000, 'beta_rec': 0.8, 'beta_kl': 0.05, 'beta_neg': None, 'lr': 2e-4, 
                                           'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                            'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior,
                                            },

                                  "sIntroVAE":  {'dataset': 'rings', 'num_iter': 30_000, 'num_vae': 2_000, 'beta_rec': 0.2, 'beta_kl': 0.2, 'beta_neg': 1.0, 'lr': 2e-4, 
                                                 'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior}
                                }

                        }
                        


    return configs[dataset][model]

def get_base_configs(dataset='8Gaussian', models =['VAE', 'sIntroVAE'], prior_modes=['imposed', 'vamp'], num_components=64):

    dataset_configs = {}

    for model in models:

        for prior in prior_modes:
            
            config = get_config(dataset= dataset, 
                                model= model,
                                prior_mode= prior,
                                num_components= num_components)

            if prior == 'imposed':
                config['num_components'] = None
                dataset_configs[len(dataset_configs)] = copy.deepcopy(config)

            else:

                for learnable_contributions in [False, True]:

                    config["learnable_contributions"]= learnable_contributions 

                    if model == 'VAE':

                        dataset_configs[len(dataset_configs)] = copy.deepcopy(config)
                    else:

                        for intro_prior in [False, True]:

                            config["intro_prior"] = intro_prior 
                            config["sampling_with_grad"]  = intro_prior

                            dataset_configs[len(dataset_configs)] = copy.deepcopy(config)
             
    return dataset_configs


## Grid-search Configs Utils ##

def get_grid_search(dataset=None,  prior_mode=None, num_components=None, learnable_contributions=None, intro_prior=None, init_mode=None, 
                                   num_vae_list=[], beta_rec_list=[], beta_kl_list=[], beta_neg_list=[], alpha_list=[], chosen_hyperparams={}):

    for alpha in alpha_list:   
        for num_vae in num_vae_list:
            for beta_rec in beta_rec_list:
                for beta_kl in beta_kl_list:
                    
                    for beta_neg_index in beta_neg_list:
                        beta_neg = float("%0.3f" % (beta_neg_index*beta_kl)) 

                        chosen_hyperparams[len(chosen_hyperparams)]={
                                                                        'dataset': dataset,
                                                                        'num_vae':num_vae,
                                                                        'beta_rec':beta_rec,
                                                                        'beta_kl':beta_kl,
                                                                        'beta_neg':beta_neg,
                                                                        'alpha': alpha,
                                                                        'prior_mode':prior_mode, 
                                                                        'num_components':num_components, 
                                                                        'learnable_contributions':learnable_contributions, 
                                                                        'intro_prior':intro_prior, 
                                                                        'sampling_with_grad':intro_prior,
                                                                        'init_mode':init_mode
                                                                        }

    return chosen_hyperparams

def get_hyperparams(dataset='8Gaussian', model='VAE', prior_mode=None, num_components=None, learnable_contributions=None, intro_prior=None, init_mode=None, chosen_hyperparams={}):

    

    chosen_hyperparams = get_grid_search(dataset=dataset,
                                        num_vae_list= [30_000] if model == 'VAE' else [2_000], 
                                        beta_rec_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,  0.7, 0.8, 1.0]
                                                        if model == 'VAE' else [0.05, 0.2, 0.3, 0.4, 0.8, 1.0],
                                        beta_kl_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0]
                                                        if model == 'VAE' else [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0],                                                                                                                                                              
                                        beta_neg_list= [1] if model == 'VAE' else [1,2,3,4,5], 
                                        alpha_list=[2],

                                        prior_mode=prior_mode, 
                                        num_components=num_components,
                                        init_mode=init_mode,
                                        learnable_contributions = learnable_contributions,
                                        intro_prior = intro_prior,
                                        chosen_hyperparams = chosen_hyperparams,
                                        )

    return chosen_hyperparams

def get_dataset_configs(dataset='8Gaussian', model='VAE', prior_mode='imposed', learnable_contributions=False, intro_prior=False, num_components=64,):

    chosen_hyperparams={}

    if prior_mode == 'imposed':
        chosen_hyperparams = get_hyperparams(dataset=dataset, model=model, 
                                                              prior_mode='imposed', learnable_contributions=False, intro_prior=False, 
                                                              chosen_hyperparams=chosen_hyperparams)
        
    else:
        chosen_hyperparams = get_hyperparams(dataset=dataset, model=model,
                                                              prior_mode=prior_mode, init_mode= "data", num_components=num_components,
                                                              learnable_contributions=learnable_contributions, intro_prior=intro_prior, 
                                                              chosen_hyperparams=chosen_hyperparams)

    return chosen_hyperparams



## Ablation Configs Utils ##

def get_optimal_params(model, current_dir): 
     
    kl_best = np.inf
    experiments = os.listdir(current_dir)

    if model == 'VAE':

        beta_neg_best = None

        ## reading experiments
        for experiment in experiments:
            
            try:
                with open(os.path.join(current_dir, experiment, 'results_log_soft_intro_vae.txt'), 'r') as f:
                    rows = f.readlines()
                
                beta_kl = float(rows[-1].split("beta_kl_")[-1].split("_")[0])
                beta_rec = float(rows[-1].split("beta_rec_")[-1].split("_")[0])
                gnelbo = float(rows[-1].split("gnelbo_")[-1].split("_")[0])
                kl = float(rows[-1].split("kl_")[-1].split("_")[0])
                jsd = float(rows[-1].split("jsd_")[-1].split("_")[0])

                if kl < kl_best:
                    kl_best = kl
                    gnelbo_best = gnelbo
                    jsd_best = jsd

                    beta_rec_best = beta_rec
                    beta_kl_best = beta_kl

            except:
                continue
    
            
    else:

        ## reading experiments
        for experiment in experiments:
            
            try:
                with open(os.path.join(current_dir, experiment, 'results_log_soft_intro_vae.txt'), 'r') as f:
                    rows = f.readlines()
                
                beta_kl = float(rows[-1].split("beta_kl_")[-1].split("_")[0])
                beta_rec = float(rows[-1].split("beta_rec_")[-1].split("_")[0])
                beta_neg = float(rows[-1].split("beta_neg_")[-1].split("_")[0])

                gnelbo = float(rows[-1].split("gnelbo_")[-1].split("_")[0])
                kl = float(rows[-1].split("kl_")[-1].split("_")[0])
                jsd = float(rows[-1].split("jsd_")[-1].split("_")[0])

                if kl < kl_best:
                    kl_best = kl
                    gnelbo_best = gnelbo
                    jsd_best = jsd

                    beta_rec_best = beta_rec
                    beta_kl_best = beta_kl
                    beta_neg_best = beta_neg
                
            except:
                continue

    return beta_rec_best, beta_kl_best, beta_neg_best, kl_best, jsd_best, gnelbo_best

def get_mode_dir(results_dir, model_, config):
     
    dataset_ = config['dataset']
    prior_ = config['prior_mode']
    num_components_ = config['num_components']
    init_mode_ = config['init_mode']
    learnable_contributions_ = config['learnable_contributions']

    if model_ == 'VAE':
        current_dir = f'{dataset_}/{model_}/prior:{prior_}_numComponents:{num_components_}_init:{init_mode_}_learnableComponent:{learnable_contributions_}/'

    else:
        sample_with_grad_ = config['sampling_with_grad']
        intro_prior_ = config['intro_prior']

        current_dir = f'{dataset_}/{model_}/prior:{prior_}_numComponents:{num_components_}_init:{init_mode_}_learnableComponent:{learnable_contributions_}_sampleGrad:{sample_with_grad_}_introPrior:{intro_prior_}/'

    mode_dir = os.path.join(results_dir, current_dir)

    return mode_dir

def get_optimal_configs(results_dir, dataset='8Gaussian', models =['VAE', 'sIntroVAE'], prior_modes=['imposed', 'vamp'], num_components=64, verbose=False):

    dataset_configs = {}

    for model in models:

        for prior in prior_modes:
            
            config = get_config(dataset= dataset, 
                                model= model,
                                prior_mode= prior,
                                num_components= num_components)

            if prior == 'imposed':
                config['num_components'] = None
                config['init_mode'] = None
                dataset_configs[len(dataset_configs)] = copy.deepcopy(config)

                current_dir = get_mode_dir(results_dir, model, dataset_configs[len(dataset_configs)-1])
                beta_rec, beta_kl, beta_neg, best_kl, best_jsd, best_gnelbo = get_optimal_params(model, current_dir)

                dataset_configs[len(dataset_configs)-1]['beta_rec'] = beta_rec
                dataset_configs[len(dataset_configs)-1]['beta_kl'] = beta_kl
                dataset_configs[len(dataset_configs)-1]['beta_neg'] = beta_neg

                if verbose:
                    print('\n', current_dir)
                    print('beta_rec: ', beta_rec, 'beta_kl: ', beta_kl, 'beta_neg: ', beta_neg, '\n', 'best_kl: ',  best_kl, 'best_jsd: ', best_jsd, 'best_gnelbo: ', best_gnelbo, '\n')
                    print(len(os.listdir(current_dir)))


            else:

                for learnable_contributions in [False, True]:

                    config["learnable_contributions"]= learnable_contributions 

                    if model == 'VAE':

                        dataset_configs[len(dataset_configs)] = copy.deepcopy(config)
                        current_dir = get_mode_dir(results_dir, model, dataset_configs[len(dataset_configs)-1])
        
                        beta_rec, beta_kl, beta_neg, best_kl, best_jsd, best_gnelbo = get_optimal_params(model, current_dir)

                        dataset_configs[len(dataset_configs)-1]['beta_rec'] = beta_rec
                        dataset_configs[len(dataset_configs)-1]['beta_kl'] = beta_kl
                        dataset_configs[len(dataset_configs)-1]['beta_neg'] = beta_neg

                        if verbose:
                            print('\n', current_dir)
                            print('beta_rec: ', beta_rec, 'beta_kl: ', beta_kl, 'beta_neg: ', beta_neg, '\n', 'best_kl: ',  best_kl, 'best_jsd: ', best_jsd, 'best_gnelbo: ', best_gnelbo, '\n')
                            print(len(os.listdir(current_dir)))

                    else:

                        for intro_prior in [False, True]:

                            config["intro_prior"] = intro_prior 
                            config["sampling_with_grad"]  = intro_prior

                            dataset_configs[len(dataset_configs)] = copy.deepcopy(config)
                            current_dir = get_mode_dir(results_dir, model, dataset_configs[len(dataset_configs)-1])
                            beta_rec, beta_kl, beta_neg, best_kl, best_jsd, best_gnelbo = get_optimal_params(model, current_dir)

                            dataset_configs[len(dataset_configs)-1]['beta_rec'] = beta_rec
                            dataset_configs[len(dataset_configs)-1]['beta_kl'] = beta_kl
                            dataset_configs[len(dataset_configs)-1]['beta_neg'] = beta_neg

                            if verbose:
                                print('\n', current_dir)
                                print('beta_rec: ', beta_rec, 'beta_kl: ', beta_kl, 'beta_neg: ', beta_neg, '\n', 'best_kl: ',  best_kl, 'best_jsd: ', best_jsd, 'best_gnelbo: ', best_gnelbo, '\n')
                                print(len(os.listdir(current_dir)))


    return dataset_configs

def augment_config_ablation(chosen_hyperparams, multi_seeds=[0], num_components=[1]):

    """
    Function augmenting the optimal configs across multi_seeds and num_components
    """

    chosen_hyperparams_augmented = {}
    for key_index in chosen_hyperparams:
                for seed in multi_seeds:

                        if chosen_hyperparams[key_index]['prior_mode'] == 'imposed':
                                chosen_hyperparams_augmented[len(chosen_hyperparams_augmented)] =  copy.deepcopy(chosen_hyperparams[key_index])
                                chosen_hyperparams_augmented[len(chosen_hyperparams_augmented)-1].update({'seed':seed}) 
                        else:
                                ## learnable prior (evaluate different number of components)
                                for num_C in reversed(num_components):

                                        chosen_hyperparams_augmented[len(chosen_hyperparams_augmented)] =  copy.deepcopy(chosen_hyperparams[key_index])
                                        chosen_hyperparams_augmented[len(chosen_hyperparams_augmented)-1].update({'seed':seed}) 
                                        chosen_hyperparams_augmented[len(chosen_hyperparams_augmented)-1].update({'num_components':num_C}) 

    return chosen_hyperparams_augmented
