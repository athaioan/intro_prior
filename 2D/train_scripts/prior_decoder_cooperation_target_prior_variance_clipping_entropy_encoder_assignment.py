import os
import datetime

from tqdm import tqdm
import torch
import torch.optim as optim
import wandb


from dataset import ToyDataset
from utils.session import seed_everything, plot_losses, save_checkpoint
from utils.losses import reparameterize, calc_reconstruction_loss, calc_kl_loss, kl_soft_assignments
from utils.results import get_qualitatives, get_quantitatives
from models import SoftIntroVAESimple



def get_session_name(**kwargs):

    model_type='VAE' if kwargs.get('num_vae')==kwargs.get('num_iter') else 'sIntroVAE'

    folder_name = '{}/{}/'.format(kwargs.get('dataset'), model_type)

    if model_type=='VAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'),
                                                                               kwargs.get('seed'))
        
        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'),
                                                                                     kwargs.get('learnable_contributions'))

    elif model_type=='sIntroVAE':
        params = 'gamma:{:2.2f}_betaRec:{:2.2f}_betaKl:{:2.2f}_betaNeg:{}_clipLogvar:{}_assingmentEncEntropyReg:{}_seed:{}'.format(kwargs.get('gamma'), kwargs.get('beta_rec'), kwargs.get('beta_kl'), 
                                                                                                      kwargs.get('beta_neg'),
                                                                                                      kwargs.get('clip_logvar'), kwargs.get('assignment_enc_entropy_reg'), 
                                                                                                      kwargs.get('seed'))
        
        prior_name='prior:{}_numComponents:{}_init:{}_learnableComponent:{}_sampleGrad:{}_introPrior:{}/'.format(kwargs.get('prior_mode'), kwargs.get('num_components'), kwargs.get('init_mode'), 
                                                                                                                 kwargs.get('learnable_contributions'), 
                                                                                                                 kwargs.get('sampling_with_grad'), kwargs.get('intro_prior'))
        

    session_name = '{}{}{}'.format(folder_name, prior_name, params)
    id_name = session_name.replace('/', '_')

    session_name = '{}_time:{}'.format(session_name, str(datetime.datetime.now())[5:19].replace(':', '-'))
   
    return session_name, id_name, model_type



def train_introspective_vae(dataset="8Gaussian", batch_size=512, z_dim=2, lr=2e-4, prior_lr=2e-4, num_iter=30_000, num_vae=2_000,
                            recon_loss_type='mse', kl_loss_type='stochastic', beta_kl=1, beta_rec=1, beta_neg=1, clip_logvar=False, assignment_enc_entropy_reg=0,
                            alpha=2.0, gamma_r=1e-8, gamma=1, MC=100,
                            prior_mode='imposed', num_components=1, init_mode='random', learnable_contributions=False, sampling_with_grad=False, intro_prior=False, mog_warmup=100,
                            result_iter=5_000, plot_qualitative=True, 
                            seed=0, result_root='./results', with_wandb=False, group_wandb='ablation', entity_wandb='main_intro_prior',
                            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), pretrained=None,
                            optim_betas_enc_dec=(0.9, 0.999), 
                            optim_betas_prior=(0.9, 0.999),
                            **kwargs):


        if prior_mode == 'vamp' and num_iter > num_vae and num_vae-mog_warmup < 0:
                raise ValueError('Not suffient VAE epochs to train the Vamp prior')




        session_name, id_session, model_type = get_session_name(dataset=dataset, num_iter=num_iter, num_vae=num_vae, gamma=gamma, beta_rec=beta_rec, beta_kl=beta_kl, beta_neg=beta_neg,
                                                                clip_logvar=clip_logvar, assignment_enc_entropy_reg=assignment_enc_entropy_reg,
                                                                prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                                                                learnable_contributions=learnable_contributions, sampling_with_grad=sampling_with_grad, intro_prior=intro_prior,
                                                                seed=seed)

        
        result_dir = os.path.join(result_root, session_name)
        os.makedirs(result_dir, exist_ok=True)


        if with_wandb:
                wandb.init(group=group_wandb, reinit=True, entity=entity_wandb,
                           project='prior_learning_sintro_2D',
                           name=id_session,
                           dir=result_root,
                           config = {'model_type':model_type, 'dataset':dataset,
                                        'prior':prior_mode, 'intro_prior': intro_prior, 
                                        'beta_rec':beta_rec, 'beta_kl':beta_kl, 'beta_neg':beta_neg,
                                        'num_components':num_components if num_components !=None else 1, 
                                        'learnable_contributions':learnable_contributions,
                                        'sampling_with_grad': sampling_with_grad, 'clip_logvar':clip_logvar, 'assignment_enc_entropy_reg':assignment_enc_entropy_reg,
                                        'MC': MC,
                                        'seed':seed},
                           settings=wandb.Settings(_disable_stats=True)
                           )


        seed_everything(seed)

        train_set = ToyDataset(distr=dataset)
        grid_scale = train_set.scale * train_set.range  

        model = SoftIntroVAESimple(xdim=2, zdim=z_dim, n_layers=3, num_hidden=256, 
                                   device=device, pretrained=pretrained, train_data_loader=train_set,
                                   prior_mode=prior_mode, num_components=num_components, init_mode=init_mode, 
                                   learnable_contributions=learnable_contributions, clip_logvar=False)


        # defining optimizers
        milestones = (10000, 15000)  ## https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html  

        optimizer_e = optim.Adam(model.encoder.main.parameters(), lr=lr, betas=optim_betas_enc_dec)
        e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1, verbose=False) 

        if model.encoder.prior.type != 'imposed':
                # trainable prior     
                optimizer_p = optim.Adam(model.encoder.prior.parameters(), lr=prior_lr, betas=optim_betas_prior)
                p_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=milestones , gamma=0.1, verbose=False) ## TODO: Fix

        optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr, betas=optim_betas_enc_dec)
        d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1, verbose=False)
        scale = 0.5  # 's' (1/n_dims)=1/2


        if plot_qualitative:
                print("plotting qualitative results ...")
                seed_everything(seed)
                get_qualitatives(model, result_dir, train_set, grid_scale, 
                                 kl_loss_type, MC,
                                 it=0, set_title=True)

        for it in tqdm(range(num_iter)):


                ## turning vamp into MoG
                if prior_mode == 'vamp' and num_iter>num_vae and it==num_vae-mog_warmup:

                        milestones = (milestones[0] - it, milestones[1] - it)

                        optimizer_e, optimizer_d, optimizer_p, \
                        e_scheduler, d_scheduler, p_scheduler = model.vamp_to_mog(num_components, learnable_contributions,
                                                                                  optimizer_e, optimizer_d, optimizer_p,
                                                                                  e_scheduler, d_scheduler, p_scheduler, 
                                                                                  milestones, lr, prior_lr, optim_betas_enc_dec, 
                                                                                  optim_betas_prior)

                ## initializing logvar clipping                                                                                        
                if prior_mode != 'imposed' and intro_prior and it==num_vae and clip_logvar and not(model.clip_logvar):
                        _, logvar_MoG, _ = model.get_prior_params()

                        model.init_clip_logvar(clip_logvar_min=logvar_MoG.detach().clone().min(axis=0)[0],
                                               clip_logvar_max=logvar_MoG.detach().clone().max(axis=0)[0])
                       

                batch, _ = train_set.next_batch(batch_size=batch_size, device=device)
              
                model.train()
                # --------------train----------------
                if it < num_vae:


                        # =========== Update E, D, P ================
                        for param in model.encoder.main.parameters():
                                param.requires_grad = True 

                        if model.encoder.prior.type != 'imposed':
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = True 

                        for param in model.decoder.parameters():  
                                param.requires_grad = True

                        # vanilla VAE training, optimizeing the ELBO for both encoder and decoder
                        real_batch = batch.to(device)

                        real_mu, real_logvar, _, rec = model(real_batch)

                        loss_rec = calc_reconstruction_loss(real_batch, rec, 
                                                            loss_type=recon_loss_type, reduction="mean")

                        loss_kl = calc_kl_loss(model, real_mu, real_logvar,
                                               mc=MC, kl_loss_type=kl_loss_type, reduction='mean')
                        
                        loss = beta_rec * loss_rec + beta_kl * loss_kl

                        optimizer_e.zero_grad()
                        optimizer_d.zero_grad()
                        if model.encoder.prior.type != 'imposed':
                                optimizer_p.zero_grad()

                        loss.backward()

                        optimizer_e.step()
                        optimizer_d.step()
                        if model.encoder.prior.type != 'imposed':
                                optimizer_p.step()
                        
                        if with_wandb:
                                wandb.log({"real_rec": loss_rec.item(),
                                           "real_kl": loss_kl.item()}) 
                                                                       
                else:
                        
                        b_size = batch.size(0) 
                        real_batch = batch.to(device)

                        # =========== Update E ================
                        for param in model.encoder.main.parameters():
                                param.requires_grad = True

                        if model.encoder.prior.type != 'imposed':
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = False                    
           
                        for param in model.decoder.parameters():  
                                param.requires_grad = False


                        noise_batch, noise_indices = model.sample_noise(b_size, ret_ind=True, with_grad=sampling_with_grad)
                        fake = model.sample(noise_batch)


                        real_mu, real_logvar, z, rec = model(real_batch)
                        
                        fake_mu, fake_logvar, _, rec_fake = model(fake.detach())  ## see IntroVAE Algorithm 1 ln.8
                        rec_mu, rec_logvar, _, rec_rec = model(rec.detach())  ## see IntroVAE Algorithm 1 ln.8

                        lossE_real_kl = calc_kl_loss(model, real_mu, real_logvar,
                                                     mc=MC, kl_loss_type=kl_loss_type,
                                                     reduction='mean')

                        fake_kl_e = calc_kl_loss(model, fake_mu, fake_logvar,
                                                 mc=MC, kl_loss_type=kl_loss_type,
                                                 reduction='none')
                        
                        rec_kl_e = calc_kl_loss(model, rec_mu, rec_logvar,
                                                mc=MC, kl_loss_type=kl_loss_type,
                                                reduction='none')


                        # reconstruction loss for the reconstructed data
                        loss_fake_rec = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction="none")
                        # reconstruction loss for the generated data
                        loss_rec_rec = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction="none")
                        # reconstruction loss for the real data
                        loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")

                        # get prior params
                        mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()

                        # assignment entropy regularization
                        real_kl_z_hook = reparameterize(real_mu.repeat(MC,1), real_logvar.repeat(MC,1)) 


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

                        # expELBO
                        exp_elbo_fake = (-alpha * scale * (beta_rec * loss_fake_rec + beta_neg * fake_kl_e)).exp().mean()
                        exp_elbo_rec = (-alpha * scale * (beta_rec * loss_rec_rec + beta_neg * rec_kl_e)).exp().mean()
                        
                        
                        lossE_fake = (1/alpha) * ( exp_elbo_fake + exp_elbo_rec) / 2

                        lossE_real = scale * (beta_kl * lossE_real_kl + beta_rec * loss_rec) 


                        lossE = lossE_real + lossE_fake + \
                                scale * assignment_enc_entropy_reg * assignment_neg_entropy
            

                        # updating encoder
                        optimizer_e.zero_grad()                       
                        lossE.backward()
                        optimizer_e.step()

                        if with_wandb:
                                wandb.log({"lossE": lossE.item(),
                                           
                                           "exp_elbo_rec": exp_elbo_rec.item(),
                                           "exp_elbo_fake": exp_elbo_fake.item(),
                                           
                                           "max_prior_logvar": logvar_MoG.max().item(),

                                           "assignment_neg_entropy": assignment_neg_entropy.item(),
                                          }, commit=False) 

                        # =========== D & P ================
                        for param in model.encoder.parameters():
                                param.requires_grad = False

                        if model.encoder.prior.type != 'imposed' and intro_prior:
                                for param in model.encoder.prior.parameters():
                                        param.requires_grad = True

                        for param in model.decoder.parameters():  
                                param.requires_grad = True

                        noise_batch, noise_indices = model.sample_noise(b_size, ret_ind=True, with_grad=sampling_with_grad)
                        fake = model.sample(noise_batch)
                        rec = model.decoder(z.detach())  ## see IntoVAE fig7 (b)

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
                       
                        # get prior params
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
                       
                        lossD = scale * (beta_rec * loss_rec +  beta_kl * real_kl +         
                                         gamma *  beta_kl * (rec_kl + fake_kl) / 2 + \
                                         gamma * gamma_r * beta_rec * (loss_rec_rec + loss_rec_fake) / 2)

                        optimizer_d.zero_grad()
                        if model.encoder.prior.type != 'imposed' and intro_prior:
                                optimizer_p.zero_grad() 

                        lossD.backward()

                        optimizer_d.step()
                        if model.encoder.prior.type != 'imposed' and intro_prior:
                                optimizer_p.step() 


                        if with_wandb:
                                wandb.log({"lossD": lossD.item(),

                                           "real_rec": loss_rec.item(),
                                           "fake_rec": loss_rec_fake.item(),

                                           "real_kl": real_kl.item(),
                                           "fake_kl": fake_kl.item(),
                                          }, commit=True) 

                        if torch.isnan(lossE) or torch.isnan(lossD):
                                print("loss is NaN.")
                                return
                        
                ### optimizer scheduling
                e_scheduler.step()
                d_scheduler.step()
                if model.encoder.prior.type != 'imposed':
                        p_scheduler.step()

                if it>0 and (it+1) % result_iter == 0 or (it+1) == num_iter or (it+1) == num_vae:

                        ## plot losses 
                        if it < num_vae:
                                ## vae stage
                                plot_losses(it, num_iter, loss_rec, loss_kl)
                        else:
                                ## adversarial stage
                                plot_losses(it, num_iter, loss_rec, real_kl, 
                                            fake_kl, rec_kl, exp_elbo_fake, exp_elbo_rec)
                                
                        if plot_qualitative:
                                seed_everything(seed + it)
                                print("plotting qualitative results ...")
                                get_qualitatives(model, result_dir, train_set, grid_scale, 
                                                 kl_loss_type, MC,
                                                 it=it, set_title=True)

                
        save_checkpoint(model, result_dir, it, it)

        seed_everything(seed + it)

        print("computing quantitative results ...")
        get_quantitatives(model, result_dir, train_set, grid_scale, 
                          kl_loss_type, MC,
                          beta_kl, beta_rec, beta_neg, seed)                       
        
        print("plotting qualitative results ...")
        get_qualitatives(model, result_dir, train_set, grid_scale, 
                         kl_loss_type, MC, 
                         it='final', set_title=False, with_ood=False, delete_figures=False)
        return 
       