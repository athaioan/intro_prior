import torch
from tqdm import tqdm

from utils import reparameterize
from utils.losses import calc_reconstruction_loss, calc_kl_loss, kl_soft_assignments
import numpy as np

def train_BN(model, train_data_loader, device='cuda', num_train_iters=5):
       
    with torch.no_grad():

        model.train()

        ## training the BN layers to account for the adversarial training (i.e. model was fed both real and fake samples)
        for epoch in range(num_train_iters):
            pbar = tqdm(iterable=train_data_loader, disable=True)

            for iter_index, (batch, batch_labels) in enumerate(pbar):
                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)
                real_batch = batch.to(device)
                model(real_batch)

    return model


def calculate_ELBO(model, data_loader, kl_loss_type, mc=100, device='cuda',
                   mu_MoG=None, logvar_MoG=None, wc_MoG=None):
    
    
    with torch.no_grad():

        model.eval()

        pbar = tqdm(iterable=data_loader)

        mse_all = 0
        RE_ELBO_all = 0
        KL_all = 0
        num_samples = 0 

        if mu_MoG is None:
            ## prior targets was not provided
            mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()


        for iter_index, (batch, batch_labels) in enumerate(pbar):

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            real_batch = batch.to(device)

            b_size = real_batch.shape[0]
            real_mu, real_logvar = model.encode(real_batch)

            z_rec = reparameterize(real_mu.repeat(mc,1), real_logvar.repeat(mc,1)) 

            num_samples += b_size * mc
            # Split tensor into patches
            z_rec = z_rec.split(b_size)
            for current_z in z_rec:
                rec = model.decode(current_z)
                mse_all += calc_reconstruction_loss(real_batch, rec, loss_type='mse', 
                                                    reduction='sum')
                
                RE_ELBO_all += calc_reconstruction_loss(real_batch, rec, loss_type='elbo', 
                                                    reduction='sum')
                
                KL_all += calc_kl_loss(model, real_mu, real_logvar, z_s=current_z,
                                      target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG,
                                      mc=1, kl_loss_type=kl_loss_type, reduction='sum')
                

    RE = RE_ELBO_all/num_samples
    KL = KL_all/num_samples
    mse = mse_all/(num_samples*np.prod(real_batch.shape[1:]))

    ELBO = RE + KL
    model.train()

    return  ELBO, KL, mse 

def kl_dift(mu_source, logvar_source, mu_target, logvar_target, reduce='mean'):
    
    kl = -0.5 * (1 + logvar_source - logvar_target - (logvar_source.exp() + (mu_source-mu_target).pow(2))/logvar_target.exp()).sum(1)

    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def calculate_dift(model, data_loader, device='cuda'):
    
    
    with torch.no_grad():

        model.eval()

        pbar = tqdm(iterable=data_loader)

        drif_all = 0
        num_samples = 0 


        for iter_index, (batch, batch_labels) in enumerate(pbar):

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            real_batch = batch.to(device)

            b_size = real_batch.shape[0]
            real_mu, real_logvar = model.encode(real_batch)
            
            ## deterministic decoding
            rec = model.decode(real_mu)
            rec_mu, rec_logvar = model.encode(rec)

            ## KL divergence between two Gaussians (closed-form) - KL[Enc(Dec(z))||z]
            drif_all += kl_dift(rec_mu, rec_logvar, real_mu, real_logvar, reduce='sum')

            num_samples += b_size
           


    drift_error = drif_all/num_samples

    model.train()

    return  drift_error


def calculate_entropy_soft_assignment(model, data_loader, mc=100, device='cuda'):
    
    
    with torch.no_grad():

        model.eval()

        pbar = tqdm(iterable=data_loader)

        neg_entropy_all = 0
        num_samples = 0 


        for iter_index, (batch, batch_labels) in enumerate(pbar):

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)
            real_batch = batch.to(device)

            b_size = real_batch.shape[0]
            real_mu, real_logvar = model.encode(real_batch)
            
            z_s = reparameterize(real_mu.repeat(mc,1), real_logvar.repeat(mc,1)) 

            mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()


            # assignment entropy regularization
            soft_assignment_real = kl_soft_assignments(model, z_s=z_s,
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
            
            neg_entropy_all += assignment_neg_entropy * b_size * mc
            num_samples += b_size * mc
           


    soft_assignment_entropy = - neg_entropy_all/num_samples

    model.train()

    return  soft_assignment_entropy
