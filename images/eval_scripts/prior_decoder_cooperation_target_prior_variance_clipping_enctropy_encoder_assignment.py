import os
import sys

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import argparse
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
import datetime

from utils import seed_everything, save_checkpoint, extract_qualitative, plot_loss_history, construct_gif, LogHist
from utils.losses import kl_loss_deterministic, calc_reconstruction_loss, calc_kl_loss, reparameterize, jacobian_norm, \
                         compute_grad_norm, get_mog_logvar
from dataset import load_dataset
from models import SoftIntroVAE
from metrics import calculate_fid_given_dataset, calculate_ELBO, calculate_dift, calculate_entropy_soft_assignment, train_BN, \
                    calculate_precision_recall_given_dataset, classification_performance

def get_session_name(**kwargs):

    model_type='VAE' if kwargs.get('num_vae')==kwargs.get('num_epochs') else 'sIntroVAE'

    folder_name='{}/{}/'.format(kwargs.get('dataset'), model_type)

    if model_type=='VAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'),
                                                                               kwargs.get('seed'))
    elif model_type=='sIntroVAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_betaNeg:{}_logvarianceLRRatio:{}_clipLogvar:{}_assingmentEncEntropyReg:{}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'), 
                                                                                                      kwargs.get('beta_neg'), kwargs.get('logvar_lr_ratio'), 
                                                                                                      kwargs.get('clip_logvar'), kwargs.get('assignment_enc_entropy_reg'), 
                                                                                                                                 kwargs.get('seed'))

    if model_type=='VAE':
        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'),
                                                                                     kwargs.get('learnable_contributions'))
    elif model_type=='sIntroVAE':
        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}_sampleGrad:{}_introPrior:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'), 
                                                                                                                 kwargs.get('learnable_contributions'), 
                                                                                                                 kwargs.get('sampling_with_grad'), kwargs.get('intro_prior'))

    session_name = '{}{}'.format(folder_name, prior_name)
   
    return session_name



def get_pretrained(result_dir, mode='train'):

    result_txt_path = 'results_fid_{}.txt'.format(mode)

    with open(os.path.join(result_dir, result_txt_path)) as f:
        lines = [line.rstrip('\n') for line in f]

    best_fid = lines[0].split(" ")[-1]  
    pretrained = [i for i in os.listdir(result_dir) if 'fid_{}'.format(best_fid) in i][0]

    pretrained = os.path.join(result_dir, pretrained)


    return result_txt_path, best_fid, pretrained




def eval_introspective_vae(dataset='cifar10', batch_size=128, num_workers=0, 
                            z_dim=2, lr=2e-4, prior_lr=2e-4, num_epochs=220, num_vae=0, recon_loss_type='mse', kl_loss_type='stochastic', beta_kl=1.0, beta_rec=1.0, beta_neg=256, logvar_lr_ratio=0.1,
                            clip_logvar=False, assignment_enc_entropy_reg=0,
                            alpha=2.0, gamma_r=1e-8, gamma=1, MC=100, 
                            prior_mode='imposed', num_components=1, init_mode='random', learnable_contributions=False, sampling_with_grad=False, intro_prior=False, mog_warmup=4,
                            result_iter=10, fid_start=100, fid_iter=40, num_row=8, with_metrics=True, plot_qualitative=True, 
                            figures_to_include = ['rec_image', 'MoGs', 'gen_image', 
                                                  'latent', '1D_latents', 'manifold_real', 'manifold_fake', 'manifold_noise'],
                            seed=0, data_root='./data', result_root='./results', result_pretrained_folder=None, with_wandb=False, group_wandb='ablation', entity_wandb='main_intro_prior',
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), pretrained=None,                          
                            optim_betas_enc_dec=(0.9, 0.999), 
                            optim_betas_prior=(0.9, 0.999),
                            best_model_mode = 'train',
                            **kwargs): 
    

    '''
    Introspective Prior Learning via prior-decoder cooperation.
    '''

    if prior_mode == 'vamp' and num_epochs > num_vae and num_vae-mog_warmup < 0:
        raise ValueError('Not suffient VAE epochs to train the Vamp prior')

    if prior_mode != 'imposed' and num_components > 1 and kl_loss_type == 'deterministic':
        raise ValueError('No closed-form KL loss for MoG priors (consider CS divergance)')
    
    session_name = get_session_name(dataset=dataset, num_epochs=num_epochs, num_vae=num_vae, 
                                                     gamma=gamma, beta_rec=beta_rec, beta_kl=beta_kl, beta_neg=beta_neg, logvar_lr_ratio=logvar_lr_ratio,
                                                     clip_logvar=clip_logvar, assignment_enc_entropy_reg=assignment_enc_entropy_reg,
                                                     prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                                                     learnable_contributions=learnable_contributions, sampling_with_grad=sampling_with_grad, intro_prior=intro_prior,
                                                     seed=seed)

  
    result_dir = os.path.join(result_root, session_name)

    ## finding run based on seed
    available_run = os.listdir(result_dir)

    for current_run in available_run:

        current_seed = int(current_run.split('seed:')[-1].split("_")[0])
        current_entropy_reg = float(current_run.split('assingmentEncEntropyReg:')[-1].split("_")[0])

        if seed == current_seed and assignment_enc_entropy_reg == current_entropy_reg:
            result_dir = os.path.join(result_dir, current_run)
            break
    try:
        result_txt_path, best_fid, pretrained = get_pretrained(result_dir, mode=best_model_mode)
    except:
        raise ValueError('Result file not found')

    if int(pretrained.split("/")[-1].split("epoch_")[-1].split("_")[0]) > num_vae and prior_mode == 'vamp':
        ## optimal FID during S-IntroVAE 
        # turning Vamp into MoG
        prior_mode = 'MoG'
    else:
        ## optimal FID during VAE
        intro_prior = False
    
    train_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='train')
    test_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='test')

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

         

    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, 
                         device=device, train_data_loader=train_data_loader, pretrained=pretrained,
                         prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                         learnable_contributions=learnable_contributions, clip_logvar=clip_logvar and intro_prior)

    seed_everything(seed) # to have identical visualization between different modes
    extract_qualitative(model, train_set, save_dir=result_dir, beta_rec=beta_rec, beta_kl=beta_kl, MC=MC, recon_loss_type=recon_loss_type,
                        kl_loss_type=kl_loss_type, batch_size=batch_size, it='final', nrow=num_row, device=device, delete_figures=False,
                        figures_to_include=figures_to_include, normalize=True)

  
    fid_inception_rec_train = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048, device=device, num_images=50_000, fid_backbone='inceptionV3', gen_mode='reconstruct', eval_mode=False)

    precision_train, recall_train = calculate_precision_recall_given_dataset(train_data_loader, model, batch_size, cuda=True,  dims=2048, device=device, num_images=50_000, 
                                                                             fid_backbone="inceptionV3", eval_mode=False)

    
    model = train_BN(model, train_data_loader, device=device, num_train_iters=1)
    test_acc = classification_performance(model, train_data_loader, test_data_loader, device='cuda', 
                                                 num_random_train_samples=1, num_random_test_samples=1, normalize=True)
    

    drift_measure = calculate_dift(model, test_data_loader, device=device)

    entropy_soft_assignment = calculate_entropy_soft_assignment(model, test_data_loader, mc=MC, device=device)

    eval_metrics = { 'fid_inception_rec_train': fid_inception_rec_train, 'precision_train': precision_train, 'recall_train': recall_train,  
                     'test_acc_svm_few': test_acc['svm_few'], 'test_acc_svm_many': test_acc['svm_many'], 
                     'test_acc_knn_few': test_acc['knn_few'], 'test_acc_knn_many': test_acc['knn_many'], 
                     'drift_measure':drift_measure, 'entropy_soft_assignment':entropy_soft_assignment, 
                     'pretrained_pth': pretrained.split("/")[-1]}

    with open(os.path.join(result_dir, result_txt_path), 'a') as f:
                for key in ['fid_inception_rec_train', 'precision_train', 'recall_train',
                            'test_acc_svm_few', 'test_acc_svm_many', 'test_acc_knn_few', 'test_acc_knn_many',
                            'drift_measure', 'entropy_soft_assignment', 'pretrained_pth']:
                    
                    f.write(f"{key}: {eval_metrics[key]}\n")
                    
    return