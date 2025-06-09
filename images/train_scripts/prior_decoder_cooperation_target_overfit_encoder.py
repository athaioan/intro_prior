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
from utils.losses import kl_loss_deterministic, calc_reconstruction_loss, calc_kl_loss, reparameterize, jacobian_norm, compute_grad_norm, get_mog_logvar
from dataset import load_dataset
from models import SoftIntroVAE
from metrics import calculate_fid_given_dataset, calculate_ELBO, train_BN
import torchvision.utils as vutils

def get_summary(figures_dir, it, figures_to_include, delete_figures=True, margin_offset=5, target_width = 1050):
    
    from PIL import Image

    img_loss = [os.path.join(figures_dir,"{}_{}.png".format(it,fig_name)) for fig_name in figures_to_include if 'samples' not in fig_name]
    img_sample = [os.path.join(figures_dir,"{}_{}.png".format(it,fig_name)) for fig_name in figures_to_include if 'samples' in fig_name]


    # taken from https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python 
    images = [Image.open(fig_name) for fig_name in img_loss]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    loss_im = Image.new('RGB', (total_width+ margin_offset*len(images), max_height), (255, 255, 255))

    x_offset = 0
    for im in images:
        loss_im.paste(im, (x_offset,0))
        x_offset += margin_offset + im.size[0]


    # Resize new_im
    w_new, h_new = loss_im.size
    new_height = int(h_new * (target_width / w_new))
    loss_im_resized = loss_im.resize((target_width, new_height), Image.BILINEAR)

    # Resize sample_im
    sample_im = plt.imread(img_sample[0])
    sample_im_pil = Image.fromarray((sample_im * 255).astype(np.uint8)) if sample_im.dtype != np.uint8 else Image.fromarray(sample_im)
    w_sample, h_sample = sample_im_pil.size
    sample_new_height = int(h_sample * (target_width / w_sample))
    sample_im_resized = sample_im_pil.resize((target_width, sample_new_height), Image.BILINEAR)


    # Concatenate new_im and sample_im_resized vertically
    # Place sample_im_resized first, then new_im below it
    total_height = sample_im_resized.height + loss_im_resized.height
    concat_im = Image.new('RGB', (loss_im_resized.width, total_height), (255, 255, 255))
    concat_im.paste(sample_im_resized, (0, 0))
    concat_im.paste(loss_im_resized, (0, sample_im_resized.height))

    concat_im.save(os.path.join(figures_dir,"summary_{}.png".format(it)))


    if delete_figures:
        ## delete figures
        img_files = [os.path.join(figures_dir, fig_name) for fig_name in os.listdir(figures_dir) if fig_name.startswith(str(it)) and fig_name.endswith('.png') and fig_name != "summary_{}.png".format(it)]
        for img in img_files:
            if os.path.exists(img):
                    os.remove(img)
    return


def get_session_name(**kwargs):

    model_type='VAE' if kwargs.get('num_vae')==kwargs.get('num_epochs') else 'sIntroVAE'

    folder_name='{}/{}/'.format(kwargs.get('dataset'), model_type)

    if model_type=='VAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'),
                                                                               kwargs.get('seed'))
    elif model_type=='sIntroVAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_betaNeg:{}_logvarianceLRRatio:{}_deformMode:{}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'), 
                                                                                                      kwargs.get('beta_neg'), kwargs.get('logvar_lr_ratio'), kwargs.get('deform_mode'), kwargs.get('seed'))

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

def get_uniform_batch(train_data_loader, device='cpu', break_iter=50):

    ## get uniform batch_size
    pbar = tqdm(iterable=train_data_loader, disable=True)
    for iter_index, (batch, batch_labels) in enumerate(pbar):


        if len(batch.size()) == 3:
            batch = batch.unsqueeze(0)  

        if iter_index == 0:
            real_batch = batch.to(device)
            labels = batch_labels.to(device)
        else:
            real_batch = torch.cat((real_batch, batch.to(device)), dim=0)
            labels = torch.cat((labels, batch_labels.to(device)), dim=0)

        if iter_index == break_iter:
            break

    sort_indices = torch.argsort(labels)
    labels = labels[sort_indices]
    real_batch = real_batch[sort_indices]
    
    unique_indices = np.unique(labels.data.cpu().numpy(), return_index=True)[1]

    batch_labels = labels[unique_indices[:10]]
    real_batch_uniform = real_batch[unique_indices[:10]]
    current_batch_size = real_batch_uniform.size(0)

    return real_batch_uniform, batch_labels, current_batch_size

def deform_batch(batch_uniform, sample='real', deform_mode='uniform_real', repeat_index=0):

    batch_size = batch_uniform.shape[0]

    if deform_mode == 'uniform_real':
        if sample == 'real':
            batch = batch_uniform
        else:
            batch =  (batch_uniform[repeat_index].unsqueeze(0)).repeat(batch_size,1,1,1)

    elif  deform_mode == 'uniform_fake':
        if sample == 'real':
            batch =  (batch_uniform[repeat_index].unsqueeze(0)).repeat(batch_size,1,1,1)
        else:
            batch = batch_uniform

    elif deform_mode == 'uniform_both':
            batch = batch_uniform

    return batch

def forward_single_batch(model, batch):

    mu, logvar, z, rec = model(batch)

    mu = torch.zeros_like(mu)
    logvar = torch.zeros_like(logvar)
    z = torch.zeros_like(z)
    rec = torch.zeros_like(rec)

    for index in range(batch.shape[0]):
        mu_temp, logvar_temp, z_temp, rec_temp  = model(batch[index].unsqueeze(0))
        mu[index, :], logvar[index, :], z[index, :], rec[index, :]  = mu_temp[0], logvar_temp[0], z_temp[0], rec_temp[0]

    return mu, logvar, z, rec



def train_introspective_vae(dataset='cifar10', batch_size=128, num_workers=0, 
                            z_dim=2, lr=2e-4, prior_lr=2e-4, num_epochs=220, num_vae=0, recon_loss_type='mse', kl_loss_type='stochastic', beta_kl=1.0, beta_rec=1.0, beta_neg=256, logvar_lr_ratio=1, deform_mode='uniform_real',
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
                                                          gamma=gamma, beta_rec=beta_rec, beta_kl=beta_kl, beta_neg=beta_neg, logvar_lr_ratio=logvar_lr_ratio, deform_mode=deform_mode,
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
                             'beta_neg':beta_neg, 'deform_mode': deform_mode,
                             'num_components':num_components if num_components !=None else 1, 
                             'logvar_lr_ratio':logvar_lr_ratio, 'seed':seed},
                   settings=wandb.Settings(_disable_stats=True)
                )

    seed_everything(seed)

    train_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='train')
    test_set, image_size, ch, channels, border = load_dataset(dataset, data_root=data_root, split='test')

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size, 
                         device=device, train_data_loader=train_data_loader, pretrained=pretrained,
                         prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                         learnable_contributions=learnable_contributions)

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

    for epoch in range(0, num_epochs):
        

        if num_epochs > num_vae and epoch == num_vae-mog_warmup:
            # turning vamp into MoG
            milestones=((milestones[0] - mog_warmup - num_vae),)
            optimizer_e, optimizer_d, optimizer_p, e_scheduler, d_scheduler, p_scheduler = model.vamp_to_mog(num_components, learnable_contributions,
                                                                                                             optimizer_e, optimizer_d, optimizer_p,
                                                                                                             e_scheduler, d_scheduler, p_scheduler, 
                                                                                                             milestones, lr, prior_lr, optim_betas_enc_dec, optim_betas_prior)
        ### Change learning rate of variance
        if epoch == num_vae-mog_warmup and prior_mode != 'imposed' and intro_prior:
            if learnable_contributions:
                              optim.Adam(
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


        if epoch == 0:
            ## get one image per class
            real_batch_uniform, batch_labels, current_batch_size = get_uniform_batch(train_data_loader, device=device, break_iter=50)


        if epoch < num_vae:
            
            # VAE training
            for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                    param.requires_grad = True 

            if model.encoder.prior.type != 'imposed':
                    for param in model.encoder.prior.parameters():
                            param.requires_grad = True 

            for param in model.decoder.parameters():  
                    param.requires_grad = True

            ## deforming real_batch
            # real_batch = deform_batch(real_batch_uniform, sample='real', deform_mode=deform_mode)
            real_batch = real_batch_uniform

            # # reconstruct real data
            real_mu, real_logvar, z_rec_hook, rec = model(real_batch)
            # real_mu, real_logvar, z_rec_hook, rec = forward_single_batch(model, real_batch) 
            z_rec_hook.retain_grad()  

            # =========== Update E, D ================
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='mean') 
            
            ## retrieving prior pararms
            mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()
            mu_unimodal_MoG, logvar_unimodal_MoG = get_mog_logvar(mu_MoG, logvar_MoG, wc_MoG)

            z_kl_hook = reparameterize(real_mu.repeat(MC,1), real_logvar.repeat(MC,1)) 
            z_kl_hook.retain_grad()  
            loss_kl = calc_kl_loss(model, real_mu, real_logvar, z_s=z_kl_hook,
                                    target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                    mc=MC, kl_loss_type=kl_loss_type, reduction='mean')
            
            vutils.save_image(torch.cat((real_batch[:current_batch_size].data.cpu(),
                                        rec[:current_batch_size].data.cpu()), dim=0),
                             os.path.join(result_dir,'samples.png'), nrow=current_batch_size)


            loss = loss_rec + loss_kl

            print(loss_kl)

            optimizer_d.zero_grad()
            optimizer_e.zero_grad()

            if prior_mode  !='imposed':
                optimizer_p.zero_grad()
                mu_MoG.retain_grad()

            real_mu.retain_grad()

            loss.backward()
            optimizer_e.step()
            optimizer_d.step()




        else:




            # # # # # ========= Update E ==================
            for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                    param.requires_grad = True 

            if model.encoder.prior.type != 'imposed':
                    for param in model.encoder.prior.parameters():
                            param.requires_grad = False 

            for param in model.decoder.parameters():  
                    param.requires_grad = False

            ## deforming real_batch
            real_batch = deform_batch(real_batch_uniform, sample='real', deform_mode=deform_mode)

            ## deforming fake_batch
            fake = deform_batch(real_batch_uniform, sample='fake', deform_mode=deform_mode, repeat_index=0)

            # real_mu, real_logvar, z_rec_hook, rec = model(real_batch)
            real_mu, real_logvar, z_rec_hook, rec = forward_single_batch(model, real_batch) 

            z_rec_hook.retain_grad()  

            # fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())  ## see IntroVAE Algorithm 1 ln.8
            fake_mu, fake_logvar, z_fake, rec_fake = forward_single_batch(model, fake.detach()) 



            mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()


            lossE_real_kl = calc_kl_loss(model, real_mu, real_logvar,
                                            target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                            mc=MC, kl_loss_type=kl_loss_type, reduction='mean')

            fake_kl_e = calc_kl_loss(model, fake_mu, fake_logvar,
                                        target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                        mc=MC, kl_loss_type=kl_loss_type, reduction='none')
            
            # rec_kl_e = calc_kl_loss(model, rec_mu, rec_logvar,
            #                         target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
            #                         mc=MC, kl_loss_type=kl_loss_type, reduction='none')

            # reconstruction loss for the reconstructed data
            loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
            # reconstruction loss for the generated data
            # loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
            # reconstruction loss for the real data
            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='mean')

            # while len(loss_rec_rec.shape) > 1:
            #     loss_rec_rec = loss_rec_rec.sum(-1)
            while len(loss_fake_rec.shape) > 1:
                loss_fake_rec = loss_fake_rec.sum(-1)

            # exp_elbo_rec = (-alpha * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
            exp_elbo_fake = (-alpha * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()

            # lossE_fake = (1/alpha) * ( exp_elbo_fake + exp_elbo_rec) / 2
            lossE_fake = (1/alpha) * ( exp_elbo_fake)
            lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)

            lossE = lossE_real + lossE_fake

            print(lossE_real_kl)

            # updating encoder
            optimizer_e.zero_grad()              

            # retain grads             
            real_mu.retain_grad()       
            real_logvar.retain_grad()       
            
            lossE.backward()
            if epoch > 0 :
                optimizer_e.step()                  


            # # # # # ========= Update D & P ==================
            for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                    param.requires_grad = False 

            if prior_mode != 'imposed' and intro_prior:
                    for param in model.encoder.prior.parameters():
                            param.requires_grad = True 
                    
            for param in model.decoder.parameters():  
                    param.requires_grad = True 

            # deforming fake_batch
            fake = deform_batch(real_batch_uniform, sample='fake', deform_mode=deform_mode, repeat_index=0)


            # rec = model.decoder(z_rec_hook.detach())  ## see IntoVAE fig7 (b) 
            real_mu, real_logvar, _, rec = forward_single_batch(model, real_batch) 


            loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction='none')

            fake_mu, fake_logvar, z_fake, rec_fake = forward_single_batch(model, fake) 


            loss_rec_fake = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type, reduction='none')

            mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()


            fake_kl = calc_kl_loss(model, fake_mu, fake_logvar, 
                                    target_mu=mu_MoG.detach(), target_logvar=logvar_MoG.detach(), target_w_c=wc_MoG.detach(),
                                    mc=MC, kl_loss_type=kl_loss_type, reduction='none')
            # prior trainable for real push forward distribution
            real_kl = calc_kl_loss(model, real_mu.detach(), real_logvar.detach(), 
                                    target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                    mc=MC, kl_loss_type=kl_loss_type, reduction='none')

            ## bounding prior to uni-modal

            lossD = scale * (beta_rec * loss_rec.mean() +  beta_kl * real_kl.mean() + \
                                gamma * beta_kl * (fake_kl.mean()) + \
                                gamma * gamma_r * beta_rec * (loss_rec_fake.mean()))
            
            if prior_mode != 'imposed' and intro_prior:
                optimizer_p.zero_grad()

            # optimizer_d.zero_grad()
            lossD.backward()
            # optimizer_d.step()

            
            if prior_mode != 'imposed' and intro_prior:
                optimizer_p.step()
            


            vutils.save_image(torch.cat((real_batch[:current_batch_size].data.cpu(),
                                         rec[:current_batch_size].data.cpu(),
                                         fake[:current_batch_size].data.cpu(),
                                         rec_fake[:current_batch_size].data.cpu(),), dim=0),
                                         os.path.join(result_dir,'{}_samples.png'.format(epoch)), nrow=current_batch_size)
            

            elbo_img = np.zeros((2, current_batch_size))
            elbo_img[0,:] = (beta_rec * loss_rec + beta_kl * real_kl).data.cpu().numpy()
            elbo_img[1,:] = (beta_rec * loss_rec_fake +  beta_kl * fake_kl).data.cpu().numpy()
            plt.figure(figsize=(4, 2))
            plt.imshow(elbo_img, cmap='inferno', vmin=0, vmax=elbo_img.max())
            # plt.colorbar(location='bottom')
            cbar = plt.colorbar(location='bottom')
            cbar.ax.tick_params(labelsize=13) 
            plt.axis('off')
            plt.title('NELBO', fontweight='bold', fontsize=20)
            plt.savefig(os.path.join(result_dir,'{}_elbo_img.png'.format(epoch)),bbox_inches='tight')
            plt.close()


            kl_img = np.zeros((2, current_batch_size))
            kl_img[0,:] = (beta_kl * real_kl).data.cpu().numpy()
            kl_img[1,:] = (beta_kl * fake_kl).data.cpu().numpy()
            plt.figure(figsize=(4, 2))
            plt.imshow(kl_img, cmap='inferno', vmin=0, vmax=elbo_img.max())
            # plt.colorbar(location='bottom')
            cbar = plt.colorbar(location='bottom')
            cbar.ax.tick_params(labelsize=13)  
            plt.axis('off')
            plt.title('KL', fontweight='bold', fontsize=20)
            plt.savefig(os.path.join(result_dir,'{}_kl_img.png'.format(epoch)),bbox_inches='tight')
            plt.close()


            rec_img = np.zeros((2, current_batch_size))
            rec_img[0,:] = (beta_rec * loss_rec).data.cpu().numpy()
            rec_img[1,:] = (beta_rec * loss_rec_fake).data.cpu().numpy()


            plt.figure(figsize=(4, 2))
            plt.imshow(rec_img, cmap='inferno', vmin=0, vmax=elbo_img.max())
            # plt.colorbar(location='bottom')
            cbar = plt.colorbar(location='bottom')
            cbar.ax.tick_params(labelsize=13) 
            plt.axis('off')
            plt.title('REC', fontweight='bold', fontsize=20)

            plt.savefig(os.path.join(result_dir,'{}_rec_img.png'.format(epoch)),bbox_inches='tight')
            plt.close()




            history_logger.append_log_hist('lossD', lossD.item()) 
            history_logger.append_log_hist('real_rec', loss_rec.mean().item())  
            history_logger.append_log_hist('fake_rec', loss_rec_fake.mean().item())  
            history_logger.append_log_hist('real_kl', real_kl.mean().item())  
            history_logger.append_log_hist('fake_kl', fake_kl.mean().item())  


            if with_wandb:
                history_logger.log_to_wandb(wandb, ['real_rec', 'fake_rec', 'real_kl', 'fake_kl', 
                                                    ], commit=False)
                



                wandb.log({"ELBO": wandb.Image(plt.imread(os.path.join(result_dir,'{}_elbo_img.png'.format(epoch))), 
                                               caption=deform_mode)}, commit=False)
                wandb.log({"KL": wandb.Image(plt.imread(os.path.join(result_dir,'{}_kl_img.png'.format(epoch))), 
                                               caption=deform_mode)}, commit=False)
                wandb.log({"REC": wandb.Image(plt.imread(os.path.join(result_dir,'{}_rec_img.png'.format(epoch))), 
                                               caption=deform_mode)}, commit=False)

                wandb.log({"Reconstructions": wandb.Image(plt.imread(os.path.join(result_dir,'{}_samples.png'.format(epoch))), 
                                                          caption=deform_mode)},commit=True)
                

            get_summary(result_dir, epoch, figures_to_include = ['rec_img', 'kl_img', 'elbo_img', 'samples'], delete_figures=True, margin_offset=5)

            if torch.isnan(lossD) or torch.isnan(lossE):
                # plot history before exiting
                plot_loss_history(history_logger.log_hist, result_dir)
                raise SystemError('NaN loss')

        e_scheduler.step()
        if prior_mode !='imposed':
            p_scheduler.step() 
        d_scheduler.step()

    construct_gif(result_dir)

    return