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

from utils import seed_everything, save_checkpoint, extract_qualitative, plot_loss_history, store_history_dict, construct_gif, LogHist
from utils.losses import kl_loss_deterministic, calc_reconstruction_loss, calc_kl_loss, kl_soft_assignments, reparameterize, jacobian_norm, compute_grad_norm, get_mog_logvar
from dataset import load_dataset
from models import SoftIntroVAE
from metrics import calculate_fid_given_dataset, calculate_ELBO, train_BN
import torchvision.utils as vutils


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
  
    session_name = '{}{}{}'.format(folder_name, prior_name, params)
    id_name = session_name.replace('/', '_')

    session_name = '{}_time:{}'.format(session_name, str(datetime.datetime.now())[5:19].replace(':', '-'))
   
    return session_name, id_name, model_type


def train_introspective_vae(dataset='cifar10', batch_size=128, num_workers=0, 
                            z_dim=2, lr=2e-4, prior_lr=2e-4, num_epochs=220, num_vae=0, recon_loss_type='mse', kl_loss_type='stochastic', beta_kl=1.0, beta_rec=1.0, beta_neg=256,
                            logvar_lr_ratio=1, clip_logvar=False,  assignment_enc_entropy_reg=0.1,
                            alpha=2.0, gamma_r=1e-8, gamma=1, MC=100, 
                            prior_mode='imposed', num_components=1, init_mode='random', learnable_contributions=False, sampling_with_grad=False, intro_prior=False, mog_warmup=4,
                            result_iter=10, fid_start=100, fid_iter=40, num_row=8, with_metrics=True, plot_qualitative=True, 
                            figures_to_include = ['rec_image', 'MoGs', 'gen_image', 
                                                  'explainability_image_with_relu', 'explainability_image', 'KL_jacobian',
                                                  'latent_KL_grads', 
                                                  'posterior_prior',
                                                  'latent', '1D_latents', 'manifold_real', 'manifold_fake', 'manifold_noise'],
                            seed=0, data_root='./data', result_root='./results', with_wandb=False, group_wandb='ablation', entity_wandb='main_intro_prior',
                            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), pretrained=None,                          
                            optim_betas_enc_dec=(0.9, 0.999), 
                            optim_betas_prior=(0.9, 0.999),
                            **kwargs): 
    

    '''
    Introspective Prior Learning via prior-decoder cooperation.
    '''

    if prior_mode == 'vamp' and num_epochs > num_vae and num_vae-mog_warmup < 0:
        raise ValueError('Not suffient VAE epochs to train the Vamp prior')

    if prior_mode != 'imposed' and num_components > 1 and kl_loss_type == 'deterministic':
        raise ValueError('No closed-form KL loss for MoG priors (consider CS divergance)')
    
    session_name, id_session, model_type = get_session_name(dataset=dataset, num_epochs=num_epochs, num_vae=num_vae, 
                                                            gamma=gamma, beta_rec=beta_rec, beta_kl=beta_kl, beta_neg=beta_neg, logvar_lr_ratio=logvar_lr_ratio,
                                                            clip_logvar=clip_logvar, assignment_enc_entropy_reg=assignment_enc_entropy_reg,
                                                            prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                                                            learnable_contributions=learnable_contributions, sampling_with_grad=sampling_with_grad, intro_prior=intro_prior,
                                                            seed=seed)

    result_dir = os.path.join(result_root, session_name)
    os.makedirs(result_dir, exist_ok=True)

    if with_wandb:
        wandb.init(group=group_wandb, reinit=True, entity=entity_wandb,
                   project='prior_learning_sintro',
                   name=id_session,
                   dir=result_root,
                   config = {'model_type':model_type, 'dataset':dataset,
                             'prior':prior_mode, 'intro_prior': intro_prior, 
                             'num_components':num_components if num_components !=None else 1, 
                             'logvar_lr_ratio':logvar_lr_ratio, 'learnable_contributions':learnable_contributions,
                             'sampling_with_grad': sampling_with_grad, 'clip_logvar':clip_logvar, 'assignment_enc_entropy_reg':assignment_enc_entropy_reg,
                             'seed':seed},
                   settings=wandb.Settings(_disable_stats=True)
                )

    seed_everything(seed)

    train_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='train')
    test_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='test')

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, 
                         device=device, train_data_loader=train_data_loader, pretrained=pretrained,
                         prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                         learnable_contributions=learnable_contributions, clip_logvar=False) # clip_logvar=False during vae stage

    scale = 1 / (ch * image_size ** 2) 

    milestones=(350,)

    optimizer_e = optim.Adam(list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()), lr=lr, betas=optim_betas_enc_dec)
    optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr, betas=optim_betas_enc_dec)
   
    e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1)
    d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1)
    if prior_mode != 'imposed':
        optimizer_p = optim.Adam(model.encoder.prior.parameters(), lr=prior_lr, betas=optim_betas_prior)
        p_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=milestones, gamma=0.1)


    history_logger = LogHist()
    model.train()

    fid_inception_train_log_list = []
    max_logvar_log_list = []
        

    for epoch in range(0, num_epochs):

        if prior_mode != 'imposed' and intro_prior and epoch == num_vae and clip_logvar and not(model.clip_logvar):
            _, logvar_MoG, _ = model.get_prior_params()

            model.init_clip_logvar(clip_logvar_min=logvar_MoG.detach().clone().min(axis=0)[0],
                                   clip_logvar_max=logvar_MoG.detach().clone().max(axis=0)[0])
            
        

        if prior_mode == 'vamp' and num_epochs > num_vae and epoch == num_vae-mog_warmup:
            # turning vamp into MoG
            milestones=((milestones[0] - epoch),)
            optimizer_e, optimizer_d, optimizer_p, e_scheduler, d_scheduler, p_scheduler = model.vamp_to_mog(num_components, learnable_contributions,
                                                                                                             optimizer_e, optimizer_d, optimizer_p,
                                                                                                             e_scheduler, d_scheduler, p_scheduler, 
                                                                                                             milestones, lr, prior_lr, optim_betas_enc_dec, optim_betas_prior)
                                                                                                             
        ### Change learning rate of variance
        if epoch == np.maximum(num_vae-mog_warmup, 0)  and prior_mode != 'imposed' and intro_prior:
            if learnable_contributions:
                optimizer_p = optim.Adam(
                                        [{"params": model.encoder.prior.mu, 'lr': prior_lr},
                                         {"params": model.encoder.prior.w, 'lr': prior_lr},
                                         {"params": model.encoder.prior.logvar, 'lr': logvar_lr_ratio * prior_lr},
                                         ])
            else:
                optimizer_p = optim.Adam(
                                        [{"params": model.encoder.prior.mu, 'lr': prior_lr},
                                         {"params": model.encoder.prior.logvar, 'lr': logvar_lr_ratio * prior_lr},
                                         ])

            p_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=milestones , gamma=0.1, verbose=False)


        if plot_qualitative and epoch % result_iter == 0 or epoch == num_epochs - 1:
            seed_everything(epoch) # to have identical visualization between different modes
            extract_qualitative(model, train_set, save_dir=result_dir, beta_rec=beta_rec, beta_kl=beta_kl, MC=MC, recon_loss_type=recon_loss_type,
                                kl_loss_type=kl_loss_type, batch_size=batch_size, it=epoch, nrow=num_row, device=device, figures_to_include=figures_to_include)


        ## computing FID
        if with_metrics and ((epoch == 0) or (epoch >= fid_start and epoch % fid_iter == 0) or epoch == num_epochs - 1):

            with torch.no_grad():
                
                model.train()
                print('Calculating fids...')

                fid_inception_train = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048, device=device, 
                                                                  num_images=50_000, fid_backbone='inceptionV3', eval_mode=False)

                history_logger.append_log_hist('fid_inception_train', fid_inception_train)


                if with_wandb:
                    history_logger.log_to_wandb(wandb, ['fid_inception_train'], commit=False)


                save_checkpoint(model, result_dir, epoch, fid_inception_train)



        pbar = tqdm(iterable=train_data_loader)

        for iter_index, (batch, batch_labels) in enumerate(pbar):
                    
                    # --------------train----------------                  
                    if len(batch.size()) == 3:
                        batch = batch.unsqueeze(0)


                    if epoch < num_vae:
                        
                        # =========== Update E, D, P ================
                        for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                                param.requires_grad = True 

                        if model.encoder.prior.type != 'imposed':
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = True 

                        for param in model.decoder.parameters():  
                                param.requires_grad = True

                        real_batch = batch.to(device)

                        # # reconstruct real data
                        real_mu, real_logvar, z_rec_hook, rec = model(real_batch)
                        z_rec_hook.retain_grad()  

                        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='mean') 
                        
                        ## retrieving prior pararms
                        mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()
                        
                        mu_unimodal_MoG, logvar_unimodal_MoG = get_mog_logvar(mu_MoG, logvar_MoG, wc_MoG)

                        z_kl_hook = reparameterize(real_mu.repeat(MC,1), real_logvar.repeat(MC,1)) 
                        z_kl_hook.retain_grad()  
                        loss_kl = calc_kl_loss(model, real_mu, real_logvar, z_s=z_kl_hook,
                                               target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                               mc=MC, kl_loss_type=kl_loss_type, reduction='mean')
                        
                        loss = beta_rec * loss_rec + beta_kl * loss_kl

                        optimizer_d.zero_grad()
                        optimizer_e.zero_grad()

                        if prior_mode  !='imposed':
                            optimizer_p.zero_grad()
                            mu_MoG.retain_grad()

                        real_mu.retain_grad()

                        loss.backward()
                        optimizer_e.step()
                        optimizer_d.step()


                        norm_grad_e = compute_grad_norm(optimizer_e)
                        norm_grad_d = compute_grad_norm(optimizer_d)
                        norm_jacobian_d = jacobian_norm(model, batch_size=batch_size, sigma=1e-4, with_grad=False)
                        KL_z_grad = z_kl_hook.grad.reshape(MC, real_batch.shape[0], -1).sum(axis=(0,1)).abs()
                        RE_z_grad = z_rec_hook.grad.reshape(real_batch.shape[0], -1).sum(axis=(0)).abs()

                        if prior_mode  !='imposed':
                            optimizer_p.step()

                        history_logger.append_log_hist('real_rec', loss_rec.item()) 
                        history_logger.append_log_hist('real_kl', loss_kl.item()) 
                        history_logger.append_log_hist('real_logvar', real_logvar.mean().item()) 
                        history_logger.append_log_hist('prior_logvar', logvar_MoG.mean().item()) 
                        history_logger.append_log_hist('prior_uni_logvar', logvar_unimodal_MoG.mean().item()) 

                        history_logger.append_log_hist('norm_grad_e', norm_grad_e.item()) 
                        history_logger.append_log_hist('norm_grad_d', norm_grad_d.item()) 
                        history_logger.append_log_hist('norm_jacobian_d', norm_jacobian_d.item()) 

                        history_logger.append_log_hist('KL_z_grad_max', KL_z_grad.max().item()) 
                        history_logger.append_log_hist('KL_z_grad_min', KL_z_grad.min().item()) 
                        history_logger.append_log_hist('RE_z_grad_max', RE_z_grad.max().item()) 
                        history_logger.append_log_hist('RE_z_grad_min', RE_z_grad.min().item())

                        history_logger.append_log_hist('squared_mu', (real_mu.unsqueeze(0) - mu_MoG.unsqueeze(1)).pow(2).mean().item())
                        history_logger.append_log_hist('squared_mu_logvarp_corr', 
                                                       np.corrcoef(np.array(history_logger.log_hist['squared_mu'])[-2_000:],
                                                                   np.array(history_logger.log_hist['prior_logvar'][-2_000:]))[0,1]) 
                        history_logger.append_log_hist('logvar_logvarp_corr', 
                                                       np.corrcoef(np.array(history_logger.log_hist['real_logvar'])[-2_000:],
                                                                   np.array(history_logger.log_hist['prior_logvar'][-2_000:]))[0,1]) 
                        
                        
                        # if iter_index == 0 and ((epoch == 0) or (epoch >= fid_start and epoch % fid_iter == 0) or epoch == num_epochs - 1):
                        #     ## store generated image
                        #     noise_batch, noise_indices = model.sample_noise(real_mu.shape[0], ret_ind=True, with_grad=sampling_with_grad)
                        #     fake_img = model.sample(noise_batch)
                        #     vutils.save_image(fake_img[:8].data.cpu(),
                        #                       os.path.join(result_dir,'fake_img_{}.png'.format(len(max_logvar_log_list))),
                        #                       nrow=4)

                        ## store generated image
                        noise_batch, noise_indices = model.sample_noise(real_mu.shape[0], ret_ind=True, with_grad=sampling_with_grad)
                        fake_img = model.sample(noise_batch)
                        vutils.save_image(fake_img[:8].data.cpu(),
                                            os.path.join(result_dir,'fake_img_{}.png'.format(epoch)),
                                            nrow=4)



                        max_logvar_log_list.append(logvar_MoG.max().item())
                        fid_inception_train_log_list.append(fid_inception_train)
                        print(epoch)

                        if with_wandb:
                             history_logger.log_to_wandb(wandb, ['real_rec', 'real_kl', 
                                                                 'prior_logvar', 'prior_uni_logvar',
                                                                 'norm_grad_e', 'norm_grad_d', 'norm_jacobian_d',
                                                                 'KL_z_grad_max', 'KL_z_grad_min', 'RE_z_grad_max', 'RE_z_grad_min',
                                                                 'squared_mu', 'squared_mu_logvarp_corr', 'logvar_logvarp_corr'],
                                                                 commit=True)
   
                    else:

                       
                        #sIntroVAE training
                        current_batch_size = batch.size(0)

                        real_batch = batch.to(device)

                        # # # # # ========= Update E ==================
                        for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                                param.requires_grad = True 

                        if model.encoder.prior.type != 'imposed':
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = False 

                        for param in model.decoder.parameters():  
                                param.requires_grad = False


                        noise_batch, noise_indices = model.sample_noise(current_batch_size, ret_ind=True, with_grad=sampling_with_grad)
                        fake = model.sample(noise_batch)

                        real_mu, real_logvar, z_rec_hook, rec = model(real_batch)
                        z_rec_hook.retain_grad()  

                        fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())  ## see IntroVAE Algorithm 1 ln.8
                        rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())  ## see IntroVAE Algorithm 1 ln.8

                        mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()

                        real_kl_z_hook = reparameterize(real_mu.repeat(MC,1), real_logvar.repeat(MC,1)) 
                        real_kl_z_hook.retain_grad()  
                        lossE_real_kl = calc_kl_loss(model, real_mu, real_logvar, z_s=real_kl_z_hook,
                                                    target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                                    mc=MC, kl_loss_type=kl_loss_type, reduction='mean')

                        fake_kl_e = calc_kl_loss(model, fake_mu, fake_logvar,
                                                target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                                mc=MC, kl_loss_type=kl_loss_type, reduction='none')
                        
                        rec_kl_e = calc_kl_loss(model, rec_mu, rec_logvar,
                                                target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                                mc=MC, kl_loss_type=kl_loss_type, reduction='none')

                        # reconstruction loss for the reconstructed data
                        loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
                        # reconstruction loss for the generated data
                        loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
                        # reconstruction loss for the real data
                        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='mean')


                        # assignment entropy regularization
                        soft_assignment_real = kl_soft_assignments(model, z_s=real_kl_z_hook,
                                                                        target_mu=mu_MoG, 
                                                                        target_logvar=logvar_MoG,
                                                                        target_w_c=wc_MoG)
                        
                        # using an epsilon (1e-8) to avoid nan when taking the log of assignemtns approaching 0
                        assignment_neg_entropy = soft_assignment_real.mean(axis=1) * (soft_assignment_real.mean(axis=1)+1e-8).log()

                        if torch.log(torch.tensor(mu_MoG.shape[0])) > 1:
                            assignment_neg_entropy = assignment_neg_entropy.sum() / torch.log(torch.tensor(mu_MoG.shape[0]))
                        else:
                            # uni-modal prior
                            assignment_neg_entropy = assignment_neg_entropy.sum()


                        while len(loss_rec_rec.shape) > 1:
                            loss_rec_rec = loss_rec_rec.sum(-1)
                        while len(loss_fake_rec.shape) > 1:
                            loss_fake_rec = loss_fake_rec.sum(-1)

                        exp_elbo_rec = (-alpha * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
                        exp_elbo_fake = (-alpha * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()

                        lossE_fake = (1/alpha) * ( exp_elbo_fake + exp_elbo_rec) / 2
                        lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)


                        lossE = lossE_real + lossE_fake + scale * assignment_enc_entropy_reg * assignment_neg_entropy

                        # updating encoder
                        optimizer_e.zero_grad()              

                        # retain grads             
                        real_mu.retain_grad()       
                        real_logvar.retain_grad()       
                        
                        lossE.backward()
                        optimizer_e.step()

                        norm_grad_e = compute_grad_norm(optimizer_e)

                        ## Note that we account for the scale multiplying the scake*beta_kl multiplying the lossE_real_kl
                        KL_z_grad = 1/(scale*beta_kl) * real_kl_z_hook.grad.reshape(MC, real_batch.shape[0], -1).sum(axis=(0,1)).abs()
                        RE_z_grad = 1/(scale*beta_kl) * z_rec_hook.grad.reshape(real_batch.shape[0], -1).sum(axis=(0)).abs()
                        real_mu_grad = real_mu.grad.sum(axis=0).abs()
                        real_logvar_grad = real_logvar.grad.sum(axis=0).abs()


                        ## Spectral radius of the aggr. posterior
                        max_real_logvar, min_real_logvar = real_logvar.mean(axis=0).max(), real_logvar.mean(axis=0).min()
                        max_prior_logvar, min_prior_logvar = logvar_MoG.mean(axis=0).max(), logvar_MoG.mean(axis=0).min()

                        # compute var of mog prior 
                        mu_unimodal_MoG, logvar_unimodal_MoG = get_mog_logvar(mu_MoG, logvar_MoG, wc_MoG)
                        max_unimodal_logvar_MoG, min_unimodal_logvar_MoG = logvar_unimodal_MoG.max(), logvar_unimodal_MoG.min()
                        
                        history_logger.append_log_hist('lossE', lossE.item()) 
                        history_logger.append_log_hist('exp_elbo_rec', exp_elbo_rec.item()) 
                        history_logger.append_log_hist('exp_elbo_fake', exp_elbo_fake.item())
                        
                        history_logger.append_log_hist('real_logvar', real_logvar.mean().item()) 
                        history_logger.append_log_hist('fake_logvar', fake_logvar.mean().item()) 
                        history_logger.append_log_hist('rec_logvar', rec_logvar.mean().item()) 
                        history_logger.append_log_hist('prior_logvar', logvar_MoG.mean().item()) 
                        history_logger.append_log_hist('prior_uni_logvar', logvar_unimodal_MoG.mean().item()) 
                        
                        history_logger.append_log_hist('max_real_logvar', max_real_logvar.item()) 
                        history_logger.append_log_hist('min_real_logvar', min_real_logvar.item()) 
                        
                        history_logger.append_log_hist('max_prior_unimodal_logvar', max_unimodal_logvar_MoG.item()) 
                        history_logger.append_log_hist('min_prior_unimodal_logvar', min_unimodal_logvar_MoG.item()) 
                        history_logger.append_log_hist('max_prior_logvar', max_prior_logvar.item()) 
                        history_logger.append_log_hist('min_prior_logvar', min_prior_logvar.item()) 


                        history_logger.append_log_hist('norm_grad_e', norm_grad_e.item()) 
                        history_logger.append_log_hist('mu_grad', real_mu_grad.mean().item()) 
                        history_logger.append_log_hist('logvar_grad', real_logvar_grad.mean().item()) 
                        
                        history_logger.append_log_hist('KL_z_grad_max', KL_z_grad.max().item()) 
                        history_logger.append_log_hist('KL_z_grad_min',KL_z_grad.min().item()) 
                        
                        history_logger.append_log_hist('RE_z_grad_max',RE_z_grad.max().item()) 

                        history_logger.append_log_hist('RE_z_grad_min',RE_z_grad.min().item())
                       
                        history_logger.append_log_hist('mu_logvar_grad_ratio', real_mu_grad.mean().item()/(real_logvar_grad.mean().item()+1e-8)) 
                        
                        history_logger.append_log_hist('KL_condition', KL_z_grad.max().item()/(KL_z_grad.min().item()+1e-8)) 
                        history_logger.append_log_hist('assignment_neg_entropy', assignment_neg_entropy.item())

                        history_logger.append_log_hist('squared_mu', (real_mu.unsqueeze(0) - mu_MoG.unsqueeze(1)).pow(2).mean().item()) 
                        history_logger.append_log_hist('squared_mu_logvarp_corr', 
                                                       np.corrcoef(np.array(history_logger.log_hist['squared_mu'])[-2_000:],
                                                                   np.array(history_logger.log_hist['prior_logvar'][-2_000:]))[0,1]) 
                        history_logger.append_log_hist('logvar_logvarp_corr', 
                                                       np.corrcoef(np.array(history_logger.log_hist['real_logvar'])[-2_000:],
                                                                   np.array(history_logger.log_hist['prior_logvar'][-2_000:]))[0,1]) 

                        # if iter_index == 0 and ((epoch == 0) or (epoch >= fid_start and epoch % fid_iter == 0) or epoch == num_epochs - 1):
                        #     ## store generated image
                        #     noise_batch, noise_indices = model.sample_noise(real_mu.shape[0], ret_ind=True, with_grad=sampling_with_grad)
                        #     fake_img = model.sample(noise_batch)
                        #     vutils.save_image(fake_img[:8].data.cpu(),
                        #                       os.path.join(result_dir,'fake_img_{}.png'.format(len(max_logvar_log_list))),
                        #                       nrow=4)

                        ## store generated image
                        noise_batch, noise_indices = model.sample_noise(real_mu.shape[0], ret_ind=True, with_grad=sampling_with_grad)
                        fake_img = model.sample(noise_batch)
                        vutils.save_image(fake_img[:8].data.cpu(),
                                            os.path.join(result_dir,'fake_img_{}.png'.format(epoch)),
                                            nrow=4)

                        max_logvar_log_list.append(logvar_MoG.max().item())
                        fid_inception_train_log_list.append(fid_inception_train)


                        if with_wandb:
                             history_logger.log_to_wandb(wandb, ['lossE', 'exp_elbo_rec', 'exp_elbo_fake', 
                                                                 'real_logvar', 'fake_logvar', 'rec_logvar', 'prior_logvar', 'prior_uni_logvar',
                                                                 'max_real_logvar', 'min_real_logvar', 
                                                                 'max_prior_unimodal_logvar', 'min_prior_unimodal_logvar', 
                                                                 'max_prior_logvar', 'min_prior_logvar', 
                                                                 'norm_grad_e', 'mu_grad', 'logvar_grad',
                                                                 'KL_z_grad_max', 'KL_z_grad_min', 'RE_z_grad_max', 'RE_z_grad_min',
                                                                 'mu_logvar_grad_ratio', 'KL_condition', 'assignment_neg_entropy', 'squared_mu', 'squared_mu_logvarp_corr', 'logvar_logvarp_corr'],
                                                                 commit=False)
                             
                             wandb.log({'histogram_real_logvar': wandb.Histogram(real_logvar.mean(axis=0).data.cpu().numpy()),
                                        'histogram_prior_logvar': wandb.Histogram(logvar_MoG.mean(axis=0).data.cpu().numpy()),
                                        'histogram_prior_unimodal_logvar': wandb.Histogram(logvar_unimodal_MoG.data.cpu().numpy())},
                                        commit=False)


                        # # # # # ========= Update D & P ==================
                        for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                                param.requires_grad = False 

                        if prior_mode != 'imposed' and intro_prior:
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = True 
                                
                        for param in model.decoder.parameters():  
                                param.requires_grad = True 


                        noise_batch, noise_indices = model.sample_noise(current_batch_size, ret_ind=True, with_grad=sampling_with_grad)
                        fake = model.sample(noise_batch)
                        rec = model.decoder(z_rec_hook.detach())  ## see IntoVAE fig7 (b) 


                        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='mean')

                        # prepare fake data for ELBO
                        rec_mu, rec_logvar = model.encode(rec)
                        z_rec = reparameterize(rec_mu, rec_logvar)

                        fake_mu, fake_logvar = model.encode(fake)
                        z_fake = reparameterize(fake_mu, fake_logvar)

                        rec_rec = model.decode(z_rec.detach()) 
                        rec_fake = model.decode(z_fake.detach())

                        loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type, reduction='mean')
                        loss_rec_fake = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction='mean')


                        mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()

                        # prior non-trainable for fake/rec push forward distribution
                        rec_kl = calc_kl_loss(model, rec_mu, rec_logvar,
                                              target_mu=mu_MoG.detach(), target_logvar=logvar_MoG.detach(), target_w_c=wc_MoG.detach(),
                                              mc=MC, kl_loss_type=kl_loss_type, reduction='mean')
                        fake_kl = calc_kl_loss(model, fake_mu, fake_logvar, 
                                               target_mu=mu_MoG.detach(), target_logvar=logvar_MoG.detach(), target_w_c=wc_MoG.detach(),
                                               mc=MC, kl_loss_type=kl_loss_type, reduction='mean')
                        # prior trainable for real push forward distribution
                        real_kl = calc_kl_loss(model, real_mu.detach(), real_logvar.detach(), 
                                               target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                               mc=MC, kl_loss_type=kl_loss_type, reduction='mean')

                        lossD = scale * (beta_rec * loss_rec +  beta_kl * real_kl + \
                                         gamma * beta_kl * (rec_kl + fake_kl) / 2 + \
                                         gamma * gamma_r * beta_rec * (loss_rec_rec + loss_rec_fake) / 2)
                        
                        if prior_mode != 'imposed' and intro_prior:
                            optimizer_p.zero_grad()

                        optimizer_d.zero_grad()

                        lossD.backward()
                        optimizer_d.step()

                        if prior_mode != 'imposed' and intro_prior:
                            optimizer_p.step()

                        norm_grad_d = compute_grad_norm(optimizer_d)
                        norm_jacobian_d = jacobian_norm(model, batch_size=batch_size, sigma=1e-4, with_grad=False)


                        history_logger.append_log_hist('lossD', lossD.item()) 
                        history_logger.append_log_hist('real_rec', loss_rec.item())  
                        history_logger.append_log_hist('fake_rec', loss_rec_fake.item())  
                        history_logger.append_log_hist('real_kl', real_kl.item())  
                        history_logger.append_log_hist('fake_kl', fake_kl.item())  
                        history_logger.append_log_hist('norm_grad_d', norm_grad_d.item())  
                        history_logger.append_log_hist('norm_jacobian_d',  norm_jacobian_d.item())  

                        if with_wandb:
                            history_logger.log_to_wandb(wandb, ['lossD',
                                                                'real_rec', 'fake_rec', 'real_kl', 'fake_kl', 
                                                                'norm_grad_d'], commit=True)

                        if torch.isnan(lossD) or torch.isnan(lossE):
                            # plot history before exiting
                            plot_loss_history(history_logger.log_hist, result_dir)
                            store_history_dict(history_logger.log_hist, result_dir)


                            with open(os.path.join(result_dir,'max_logvar_log_list.txt'), 'w') as f:
                                for item in max_logvar_log_list:
                                    f.write("%s\n" % item)
                            with open(os.path.join(result_dir,'fid_inception_train_log_list.txt'), 'w') as f:
                                for item in fid_inception_train_log_list:
                                    f.write("%s\n" % item)



                            raise SystemError('NaN loss')

        e_scheduler.step()
        if prior_mode !='imposed':
            p_scheduler.step() 
        d_scheduler.step()

    plot_loss_history(history_logger.log_hist, result_dir)
    store_history_dict(history_logger.log_hist, result_dir)

    with open(os.path.join(result_dir,'max_logvar_log_list.txt'), 'w') as f:
        for item in max_logvar_log_list:
            f.write("%s\n" % item)
    with open(os.path.join(result_dir,'fid_inception_train_log_list.txt'), 'w') as f:
        for item in fid_inception_train_log_list:
            f.write("%s\n" % item)
    
    if len(figures_to_include) > 0:
        construct_gif(result_dir)

    return