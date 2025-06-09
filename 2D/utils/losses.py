import torch.nn.functional as F
import torch
import numpy as np

PI = torch.from_numpy(np.asarray(np.pi))

def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std

def softplus(x, beta, threshold=20):
    """
    This function applies the softplus function. 
    Supports vector-beta. The threshold is used for
    numerical stability similar to torch.
    """
    
    # Element-wise threshold condition
    condition = beta * x > threshold
    
    # Vectorized softplus calculation
    output = torch.where(condition, x, 1 / beta * torch.log(1 + torch.exp(beta * x)))
    
    return output



def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)

    elif loss_type == 'elbo':
        # https://github.com/clementchadebec/benchmark_VAE/
        # Gaussian decoder with N(mu, I) - fixed decoder variance
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = 0.5 * recon_error + 0.5*torch.log(2*PI)  # logvar_decoder = 0
        recon_error = recon_error.sum(1)

    else:
        raise NotImplementedError
    
    if reduction == 'sum':
        recon_error = recon_error.sum()
    elif reduction == 'mean':
        recon_error = recon_error.mean()

    return recon_error



# taken from https://github.com/jmtomczak/intro_dgm
def log_Normal_diag(x, mu, log_var, reduction=None, dim=None):

    log_p = - 0.5 * log_var - 0.5 * torch.exp(-log_var) * (x - mu)**2.

    if reduction == 'avg':
        return torch.mean(log_p, dim)
    elif reduction == 'sum':
        return torch.sum(log_p, dim)
    else:
        return log_p
    

def kl_loss_stochastic_with_assignments(model, mean_s, log_var_s, soft_assignments, reduction='sum', mc=1, z_s=None, 
                       target_mu=None, target_logvar=None, target_w_c=None):

    ## MC estimation
    mean_s = mean_s.repeat(mc,1)
    log_var_s = log_var_s.repeat(mc,1)

    if z_s == None:
        ## MC estimationicm
        z_s = reparameterize(mean_s, log_var_s) 
    else:           
        ## z_s given      
        assert mean_s.shape[0] == z_s.shape[0], 'Z_S does not have the correct amount of MC samples'
        
    if target_mu is not None:
        ## different target than source
        mu = target_mu
        logvar = target_logvar
        w_c = target_w_c
    else:
        mu, logvar, w_c = model.get_prior_params()


    ## log_posterior_det
    # log-mixture-of-Gaussians
    w_c = w_c.view(-1,1)
    z_s = z_s.unsqueeze(0) # 1 x (BxMC) x L
    mu = mu.unsqueeze(1) # K x 1 x L
    logvar = logvar.unsqueeze(1) # K x 1 x L

    log_p = log_Normal_diag(z_s, mu, logvar).sum(-1) # K x (BxMC) 
    log_prior_det = torch.logsumexp(log_p, dim=0, keepdim=False) # (BxMC) 

    log_prior_det = (soft_assignments * log_p).sum(axis=0)

    ## log log_prior_det
    log_posterior_det = log_Normal_diag(z_s, mean_s, log_var_s).sum(-1)  # (BxMC)

    kl_mc = log_posterior_det - log_prior_det
    kl_mc = kl_mc.reshape(mc,-1) # MC X B 

    kl = torch.mean(kl_mc, dim=0) # B 

    if reduction == 'sum':
        kl = torch.sum(kl)
    elif reduction == 'mean':
        kl = torch.mean(kl)

    return kl   


def kl_soft_assignments(model, z_s, target_mu=None, target_logvar=None, target_w_c=None):


    if target_mu is not None:
        ## prior provided
        mu = target_mu
        logvar = target_logvar
        w_c = target_w_c
    else:
        mu, logvar, w_c = model.get_prior_params()


    ## log_posterior_det
    # log-mixture-of-Gaussians
    w_c = w_c.view(-1,1)
    z_s = z_s.unsqueeze(0) # 1 x (BxMC) x L
    mu = mu.unsqueeze(1) # K x 1 x L
    logvar = logvar.unsqueeze(1) # K x 1 x L

    log_p = log_Normal_diag(z_s, mu, logvar).sum(-1) + torch.log(w_c) # K x (BxMC) 
    log_prior_det = torch.logsumexp(log_p, dim=0, keepdim=False) # (BxMC) 

    ## computing assignments
    soft_assignments = (log_p - log_prior_det).exp()
    soft_assignments = soft_assignments.unsqueeze(-1)

    return soft_assignments.squeeze(-1)

    

def kl_loss_stochastic(model, mean_s, log_var_s, reduction='sum', mc=1, z_s=None, 
                       target_mu=None, target_logvar=None, target_w_c=None):

    ## MC estimation
    mean_s = mean_s.repeat(mc,1)
    log_var_s = log_var_s.repeat(mc,1)

    if z_s == None:
        ## MC estimationicm
        z_s = reparameterize(mean_s, log_var_s) 
    else:           
        ## z_s given      
        assert mean_s.shape[0] == z_s.shape[0], 'Z_S does not have the correct amount of MC samples'
        
    if target_mu is not None:
        ## different target than source
        mu = target_mu
        logvar = target_logvar
        w_c = target_w_c
    else:
        mu, logvar, w_c = model.get_prior_params()


    ## log_posterior_det
    # log-mixture-of-Gaussians
    w_c = w_c.view(-1,1)
    z_s = z_s.unsqueeze(0) # 1 x (BxMC) x L
    mu = mu.unsqueeze(1) # K x 1 x L
    logvar = logvar.unsqueeze(1) # K x 1 x L

    log_p = log_Normal_diag(z_s, mu, logvar).sum(-1) + torch.log(w_c) # K x (BxMC) 
    log_prior_det = torch.logsumexp(log_p, dim=0, keepdim=False) # (BxMC) 

    ## log log_prior_det
    log_posterior_det = log_Normal_diag(z_s, mean_s, log_var_s).sum(-1)  # (BxMC)

    kl_mc = log_posterior_det - log_prior_det
    kl_mc = kl_mc.reshape(mc,-1) # MC X B 

    kl = torch.mean(kl_mc, dim=0) # B 

    if reduction == 'sum':
        kl = torch.sum(kl)
    elif reduction == 'mean':
        kl = torch.mean(kl)

    return kl   

def kl_loss_stochastic_mi(model, mean_s, log_var_s, noise_indices, reduction='sum', mc=1, z_s=None,
                           target_mu=None, target_logvar=None, target_w_c=None):

    ## MC estimation
    mean_s = mean_s.repeat(mc,1)
    log_var_s = log_var_s.repeat(mc,1)
    noise_indices = noise_indices.repeat(mc)

    if z_s == None:
        ## MC estimation
        z_s = reparameterize(mean_s, log_var_s) 
    else:           
        ## z_s given      
        assert mean_s.shape[0] == z_s.shape[0], 'Z_S does not have the correct amount of MC samples'
        
    if target_mu is not None:
        ## different target than source
        mu = target_mu
        logvar = target_logvar
        w_c = target_w_c
    else:
        mu, logvar, w_c = model.get_prior_params()


    ## log_posterior_det
    # log-mixture-of-Gaussians
    w_c = w_c.view(-1,1)
    z_s = z_s.unsqueeze(0) # 1 x (BxMC) x L
    mu = mu.unsqueeze(1) # K x 1 x L
    logvar = logvar.unsqueeze(1) # K x 1 x L

    

    log_p = log_Normal_diag(z_s, mu, logvar).sum(-1) + torch.log(w_c) # K x (BxMC) 
    # selecting mode according to noise_indices
    log_p = log_p.gather(0, noise_indices.unsqueeze(0)) # 1 x (BxMC)
    log_prior_det = torch.logsumexp(log_p, dim=0, keepdim=False) # (BxMC) 

    ## log log_prior_det
    log_posterior_det = log_Normal_diag(z_s, mean_s, log_var_s).sum(-1)  # (BxMC)

    kl_mc = log_posterior_det - log_prior_det
    kl_mc = kl_mc.reshape(mc,-1) # MC X B 

    kl = torch.mean(kl_mc, dim=0) # B 

    if reduction == 'sum':
        kl = torch.sum(kl)
    elif reduction == 'mean':
        kl = torch.mean(kl)

    return kl   

### Standard Gaussian Prior KL ### 
def kl_loss_deterministic(logvar, mu, reduce='sum', target_mu=None, target_logvar=None):
    ## taken from https://github.com/taldatech/soft-intro-vae-pytorch

    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if target_mu is not None:
        kl = -0.5 * (1 + logvar - target_logvar - (logvar.exp() + (mu-target_mu).pow(2))/target_logvar.exp()).sum(1)

    else:
        # standard Gaussian
        kl = -0.5 * (1 + logvar - logvar.exp() - (mu).pow(2)).sum(1)
    
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl

def calc_kl_loss(model, mean_s, log_var_s, kl_loss_type='stochastic', reduction='sum', mc=1, z_s = None,
                  target_mu=None, target_logvar=None, target_w_c=None, noise_indices=None, soft_assignments=None):

    if kl_loss_type == 'stochastic':
        kl = kl_loss_stochastic(model, mean_s, log_var_s, reduction=reduction, mc=mc, z_s=z_s, target_mu=target_mu, 
                                target_logvar=target_logvar, target_w_c=target_w_c)
    elif kl_loss_type == 'mi':
        kl = kl_loss_stochastic_mi(model, mean_s, log_var_s, noise_indices=noise_indices, reduction=reduction, mc=mc, z_s=z_s, target_mu=target_mu, 
                                target_logvar=target_logvar, target_w_c=target_w_c)
    elif kl_loss_type == 'deterministic':
        kl = kl_loss_deterministic(log_var_s, mean_s, reduce=reduction, target_mu=target_mu, target_logvar=target_logvar)

    elif kl_loss_type == 'assignment_consistent':
        kl = kl_loss_stochastic_with_assignments(model, mean_s, log_var_s, soft_assignments=soft_assignments, 
                                                 reduction=reduction, mc=mc, z_s=z_s, target_mu=target_mu, 
                                                 target_logvar=target_logvar, target_w_c=target_w_c)
    else:
        raise NotImplementedError("kl_loss_type not implemented")

    return kl


