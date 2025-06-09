import random, os
import numpy as np
import torch
import datetime
import matplotlib.pyplot as plt
import cv2

           
def plot_loss_history(log_hist, result_dir):
    num_plots = len(log_hist)
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    fig.subplots_adjust(hspace=0.5)

    colors = plt.cm.rainbow(np.linspace(0, 1, num_plots))  # Generate a range of colors
    for i, (key, values) in enumerate(log_hist.items()):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        ax.plot(values, color=colors[i])  # Set the color for each line
        ax.set_title(key)
        ax.set_xlabel('Iteration')

    plt.savefig(os.path.join(result_dir, 'loss_plots.png'))
    plt.close(fig)


def store_history_dict(result_dict, result_dir):
    with open(os.path.join(result_dir, 'history.txt'), 'w') as f:
        for key, value in result_dict.items():
            f.write(f'{key}: {value}\n')
    return 

def load_history_dict(result_dir):
    result_dict = {}
    with open(os.path.join(result_dir, 'history.txt'), 'r') as f:
        for line in f:
            key, value = line.split(': ')
            result_dict[key] = value
    return result_dict

def save_checkpoint(model, name_session, epoch, fid):
    model_out_path = os.path.join(name_session,"model_epoch_{}_fid_{}.pth".format(epoch, fid))
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, model_out_path)
    print("model checkpoint saved @ {}".format(model_out_path))
    return


def seed_everything(seed: int):

    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\n --------- {} seed specified ---------\n".format(seed))

    return


def construct_gif(summary_dir):

    summary_imgs = os.listdir(summary_dir)
    summary_imgs = [img for img in summary_imgs if 'summary' in img]

    summary_imgs.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    import imageio

    # Create a list of image paths
    image_paths = [os.path.join(summary_dir, img) for img in summary_imgs]

    # Create the output GIF file path
    output_gif_path = os.path.join(summary_dir, 'summary.gif')

    # Read images and save as GIF
    img_size = imageio.imread(image_paths[0]).shape
    images = [cv2.resize(imageio.imread(img_path), (img_size[1], img_size[0])) for img_path in image_paths]
    imageio.mimsave(output_gif_path, images)

    print("GIF created successfully!")

def get_training_mode(mode=None):
        
    choices=[   
              ## prior--decoder cooperation
             'pd_cooperation_target_overfit_encoder', ## train script to realize the Proposition 1.
             'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment_supp_explode', ## training script used to demonstrating the exploding logvariance behavior clipping vs no clipping
             'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment_supp_explode_assignment', ## training script used to demonstrating the behavior under unregularized responsibilities
             'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment', ##  clipping the logvariances to a certain range + regularizing the entropy of the soft-assignments (i.e. perform the encoder update s.t. all modes contributing to the posterior)
             'pd_cooperation_target_prior_variance_clipping_entropy_encoder_assignment_decoder_constistent', ##  clipping the logvariances to a certain range + regularizing the entropy of the soft-assignments (i.e. perform the encoder update s.t. all modes contributing to the posterior) + decoder self-consistency 
            
              ## prior--encoder cooperation
             'pe_cooperation_target_prior_variance_clipping_entropy_encoder_assignment', ## training prior and encoder in cooperation
            ]    
    
    if mode is None:
        return choices
    
    else:
           
        import importlib

        module_mapping = {
            'pd': 'prior_decoder',
            'pe': 'prior_encoder'
        }

        module = mode.split("_")

        if module[0] in module_mapping:
            module[0] = module_mapping[module[0]]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        module = "_".join(module)
        module = '.'.join(['train_scripts', module])
        
        
        return importlib.import_module(module).train_introspective_vae


def get_eval_mode(mode=None):
        
    choices=[
             
             'pd_cooperation_target_prior_variance_clipping_enctropy_encoder_assignment', ##  script for running evaluation
            
            ]    
    
    if mode is None:
        return choices
    
    else:
           
        import importlib

        module_mapping = {
            'pd': 'prior_decoder',
            'pe': 'prior_encoder'
        }

        module = mode.split("_")

        if module[0] in module_mapping:
            module[0] = module_mapping[module[0]]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        module = "_".join(module)
        module = '.'.join(['eval_scripts', module])
        
        
        return importlib.import_module(module).eval_introspective_vae


class LogHist():
    
    def __init__(self):
        self.log_hist = {  
                         'fid_inception_train': [], 
                         'fid_inception_eval': [],
                         'fid_SwAV_train': [],
                         'fid_SwAV_eval': [],


                         'precision_train': [],
                         'recall_train': [],
                         'precision_eval': [],
                         'recall_eval': [],
                         'test_acc': [],

                         'ELBO_train': [],
                         'KL_train': [],
                         'RE_mse_train': [],

                         'ELBO_test': [],
                         'KL_test': [],
                         'RE_mse_test': [],

                         'lossE' : [],
                         'lossD' : [],
                         'exp_elbo_rec' : [],
                         'exp_elbo_fake' : [],
                            
                         'real_logvar' : [],
                         'fake_logvar' : [],
                         'rec_logvar' : [],
                         'prior_logvar' : [],
                         'prior_uni_logvar': [],

                         'real_rec' : [],
                         'fake_rec' : [],

                         'real_kl' : [],           
                         'fake_kl' : [],

                         'norm_grad_e' : [],
                         'norm_grad_d' : [],
                         'norm_jacobian_d' : [],

                         'max_real_logvar' : [],
                         'min_real_logvar' : [],

                         'max_prior_unimodal_logvar' : [],
                         'min_prior_unimodal_logvar' : [],
                         'max_prior_logvar' : [],
                         'min_prior_logvar' : [],

                         'squared_mu':[],
                         'num_inactive_ratio':[],

                         'squared_mu_logvarp_corr':[],
                         'logvar_logvarp_corr':[],

                         'mu_grad':[],
                         'logvar_grad':[],

                         'mu_logvar_grad_ratio' : [],
                         'KL_z_grad_max' : [],
                         'KL_z_grad_min' : [],

                         'RE_z_grad_max' : [],
                         'RE_z_grad_min' : [],

                         'entropy_ancoring' : [],
                         'constistent_loss' : [],

                         'KL_condition' : [],
                         'prior_KL_real_mode' : [],

                         'assignment_neg_entropy': [],

                        }

        self.log_hist_categories = {
                                    'metrics': ['fid_inception_train', 'fid_inception_eval', 'fid_SwAV_train', 'fid_SwAV_eval',
                                                'precision_train', 'recall_train', 'precision_eval', 'recall_eval', 'test_acc',
                                                'ELBO_train', 'KL_train', 'RE_mse_train', 'ELBO_test', 'KL_test', 'RE_mse_test'],

                                    'loss': ['lossE', 'lossD','exp_elbo_rec', 'exp_elbo_fake',
                                             'real_rec', 'fake_rec',
                                             'real_kl', 'fake_kl'],

                                    'logvar': ['real_logvar', 'fake_logvar', 'rec_logvar', 'prior_logvar', 'prior_uni_logvar',
                                               'max_real_logvar', 'min_real_logvar',
                                               'max_prior_logvar', 'min_prior_logvar', 
                                               'max_prior_unimodal_logvar', 'min_prior_unimodal_logvar'],

                                    'grad': ['norm_grad_e', 'norm_grad_d', 'norm_jacobian_d',
                                             'mu_grad', 'logvar_grad',
                                             'KL_z_grad_max', 'KL_z_grad_min', 
                                             'RE_z_grad_max', 'RE_z_grad_min'],

                                    'condition': ['entropy_ancoring', 'KL_condition', 'mu_logvar_grad_ratio', 
                                                  'squared_mu_logvarp_corr', 'logvar_logvarp_corr', 'squared_mu', 
                                                  'prior_KL_real_mode', 'num_inactive_ratio', 'assignment_neg_entropy']
                                }

        # Reverse the categries
        self.log_hist_categories_reversed = {}
        for key, values in self.log_hist_categories.items():
            for value in values:
                self.log_hist_categories_reversed[value] = key


    def append_log_hist(self, key, value):
        self.log_hist[key].append(value)
        return


    def log_to_wandb(self, wandb, keys, commit=False):
        dict_to_log = {}
        for key in keys:

            if not np.isnan(self.log_hist[key][-1]):
                
                category = self.log_hist_categories_reversed[key] if key in self.log_hist_categories_reversed else 'miscellaneous' 

                dict_to_log['/'.join([category,key])] = self.log_hist[key][-1]  

        wandb.log(dict_to_log, commit=commit)
        
        return