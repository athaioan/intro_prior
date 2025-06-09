def recommended_configs(model='VAE', dataset='cifar10', prior_mode='imposed', num_components=None, init_mode=None, 
                        learnable_contributions=False, sampling_with_grad=False, intro_prior=False):


    """
    Recommended hyper-parameters:
    - CIFAR10: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - SVHN: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 128, batch_size: 32
    - MNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    - FashionMNIST: beta_kl: 1.0, beta_rec: 1.0, beta_neg: 256, z_dim: 32, batch_size: 128
    """

    if 'gray' or 'expanded' in dataset:
        dataset_config = dataset.split('_')[0]
    else:
        dataset_config = dataset

    
    configs = {"cifar10": 
                        {"VAE" : {'dataset': dataset, 'z_dim':128, 'batch_size':32, 'num_epochs':220, 'num_vae':220, 'beta_rec':1.0, 'beta_kl': 1.0, 'beta_neg': None, 'lr':2e-4, 
                                                   'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                   'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},
                                        
                        "sIntroVAE": {'dataset': dataset, 'z_dim':128, 'batch_size':32, 'num_epochs':220, 'num_vae':20, 'beta_rec':1.0, 'beta_kl':1.0, 'beta_neg':256, 'lr':2e-4, 
                                                      'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                                      'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},
                        },
                                        

               "svhn": 
                        {"VAE" : {'dataset': dataset, 'z_dim':128, 'batch_size':32, 'num_epochs':220, 'num_vae':220, 'beta_rec':1.0, 'beta_kl': 1.0, 'beta_neg': None, 'lr':2e-4, 
                                  'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                  'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},

                         "sIntroVAE": {'dataset': dataset, 'z_dim':128, 'batch_size':32, 'num_epochs':220, 'num_vae':20, 'beta_rec':1.0, 'beta_kl':1.0, 'beta_neg':256, 'lr':2e-4, 
                                       'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                       'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},
                        },

               "mnist": 
                       {"VAE" : {'dataset': dataset, 'z_dim':32, 'batch_size':128, 'num_epochs':220, 'num_vae':220, 'beta_rec':1.0, 'beta_kl': 1.0, 'beta_neg': None, 'lr':2e-4, 
                                 'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                 'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},

                        "sIntroVAE": {'dataset': 'mnist', 'z_dim':32, 'batch_size':128, 'num_epochs':220, 'num_vae':20, 'beta_rec':1.0, 'beta_kl':1.0, 'beta_neg':128, 'lr':2e-4, 
                                      'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                      'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},
                       },
                       
                            
               "fmnist": 
                       {"VAE" : {'dataset': dataset, 'z_dim':32, 'batch_size':128, 'num_epochs':220, 'num_vae':220, 'beta_rec':1.0, 'beta_kl': 1.0, 'beta_neg': None, 'lr':2e-4, 
                                 'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                 'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},

                        "sIntroVAE": {'dataset': dataset, 'z_dim':32, 'batch_size':128, 'num_epochs':220, 'num_vae':20, 'beta_rec':1.0, 'beta_kl':1.0, 'beta_neg':256, 'lr':2e-4, 
                                      'prior_mode': prior_mode, 'num_components': num_components, 'init_mode': init_mode, 'learnable_contributions': learnable_contributions, 
                                      'sampling_with_grad':sampling_with_grad, 'intro_prior':intro_prior},
                       },                                                                    
                }

    return configs[dataset_config][model]


def get_configs(datasets=['cifar10'], models=['VAE', 'sIntroVAE'], prior_modes=['imposed', 'MoG', 'vamp'], num_components=[128], init_mode=['random','data'], learnable_contributions=[True, False], 
                sampling_with_grad=[False, True], intro_prior=[False, True], intro_with_grad = True):
    
    configs = {}

    for current_dataset in datasets:
        for current_model in models:
            # VAE configs
            if current_model == 'VAE':
                for current_prior_mode in prior_modes:
                    if current_prior_mode == 'imposed':

                        configs[len(configs)] = recommended_configs(dataset=current_dataset, model=current_model, prior_mode='imposed')

                    else:

                        for current_num_components in num_components:
                            for current_init_mode in init_mode:
                                for current_learnable_contrubtion in learnable_contributions:

                                    if current_learnable_contrubtion and current_num_components==1:
                                        ## skip learnable contributions when num_components = 1
                                        continue


                                    configs[len(configs)] = recommended_configs(dataset=current_dataset, model=current_model, 
                                                                                prior_mode=current_prior_mode, num_components=current_num_components, init_mode=current_init_mode,
                                                                                learnable_contributions=current_learnable_contrubtion)
            # Introspective configs
            elif current_model == 'sIntroVAE':
                
                for current_prior_mode in prior_modes:
                    if current_prior_mode == 'imposed':
                        configs[len(configs)] = recommended_configs(dataset=current_dataset, model=current_model, prior_mode='imposed')

                    else:

                        for current_num_components in num_components:
                            for current_init_mode in init_mode:
                                for current_learnable_contrubtion in learnable_contributions:
                                    
                                    if current_learnable_contrubtion and current_num_components==1:
                                        ## skip learnable contributions when num_components = 1
                                        continue


                                    for current_sampling_with_grad in sampling_with_grad:
                                        for current_intro_prior in intro_prior:
                                            
                                            if not(current_intro_prior):
                                                if current_sampling_with_grad:
                                                    ## skip sampling with grad when in non intro-prior mode
                                                    continue
                                            
                                            if intro_with_grad and current_intro_prior:
                                                if not(current_sampling_with_grad):
                                                    ## skip NOT sampling with grad when in intro-prior mode
                                                    continue
                                            

                                            configs[len(configs)] = recommended_configs(dataset=current_dataset, model=current_model, 
                                                                                        prior_mode=current_prior_mode, num_components=current_num_components, init_mode=current_init_mode,
                                                                                        learnable_contributions=current_learnable_contrubtion, sampling_with_grad=current_sampling_with_grad,
                                                                                        intro_prior=current_intro_prior)
            else:
                raise NotImplementedError("Model type not recognized")

    return configs



def augment_configs(hyperparams, args):

    for index in hyperparams:  
        for key in args.keys():
            if key not in hyperparams[index].keys():
                hyperparams[index][key] = args[key]

    return hyperparams


def augment_intro_configs(hyperparams, param_ablate, only_intro_mode=True):

    param_values = { 'beta_pos': [0, 1], #'beta_pos': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1],
                     'prior_reg': [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                     'logvar_lr_ratio': [1], #[0, 1e-2, 1e-1, 1],
                     'mode_consistent_reg': [1, 0, 1e-1, 1e-2],
                     'entropy_reg': [0, 1e-3, 1e-2, 1e-1],
                     'assignment_enc_entropy_reg': [0, 1, 10, 100],
                     'clip_logvar': [False, True],
                   }
    
    values = param_values[param_ablate]

    intro_hyperparams = {}
    non_intro_hyperparams = {}

    count = 0
    for index in hyperparams:  

        if hyperparams[index]['intro_prior']:
            ## create a copy of that hyperparam with the ablated value
            for value in values:
                intro_hyperparams[count] = hyperparams[index].copy()
                intro_hyperparams[count][param_ablate] = value
                count += 1

        else:

            if only_intro_mode or hyperparams[index]['prior_mode'] == 'imposed':
                non_intro_hyperparams[count] = hyperparams[index].copy()
                count += 1
            else:
                for value in values:
                    non_intro_hyperparams[count] = hyperparams[index].copy()
                    non_intro_hyperparams[count][param_ablate] = value
                    count += 1


    
    augmented_hyperparams = {**intro_hyperparams, **non_intro_hyperparams}
    augmented_hyperparams = dict(sorted(augmented_hyperparams.items()))

    return augmented_hyperparams
    

def augment_encoder_ovefit_configs(hyperparams, bneg=[0, 1, 128, 256, 1024, 2056, 4112], num_components=[1,10], deform_mode=['uniform_fake', 'uniform_real', 'uniform_both'], num_vae=0, num_epochs=2_000, logvar_lr_ratio=1):

    augmented_hyperparams = {}

    count = 0
    for index in hyperparams:  
        for bneg_val in bneg:
            for num_components_val in num_components:
                for deform_mode_val in deform_mode:
                    augmented_hyperparams[count] = hyperparams[index].copy()
                    augmented_hyperparams[count]['beta_neg'] = bneg_val
                    augmented_hyperparams[count]['num_components'] = num_components_val
                    augmented_hyperparams[count]['deform_mode'] = deform_mode_val

                    augmented_hyperparams[count]['batch_size'] = 10
                    augmented_hyperparams[count]['num_vae'] = num_vae
                    augmented_hyperparams[count]['num_epochs'] = num_epochs
                    augmented_hyperparams[count]['logvar_lr_ratio'] = logvar_lr_ratio
                    
                    if augmented_hyperparams[count]['dataset'] == 'fmnist':
                        if augmented_hyperparams[count]['num_components'] == 1:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_fmnist_MoG_1_fixed.pth'
                        elif augmented_hyperparams[count]['num_components'] == 10:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_fmnist_MoG_10_fixed.pth'
                    
                    elif augmented_hyperparams[count]['dataset'] == 'mnist':
                        if augmented_hyperparams[count]['num_components'] == 1:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_mnist_MoG_1_fixed'
                        elif augmented_hyperparams[count]['num_components'] == 10:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_mnist_MoG_10_fixed'
                    
                    elif augmented_hyperparams[count]['dataset'] == 'cifar10':
                        if augmented_hyperparams[count]['num_components'] == 1:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_cifar_MoG_1_fixed'
                        elif augmented_hyperparams[count]['num_components'] == 10:
                            augmented_hyperparams[count]['pretrained'] = 'path_to_cifar_MoG_10_fixed'
                    
                    count += 1

    augmented_hyperparams = dict(sorted(augmented_hyperparams.items()))
    return augmented_hyperparams
    
