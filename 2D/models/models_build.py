import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_samples_from_dataloader

from utils import reparameterize, softplus


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation=None):
        super(NonLinear, self).__init__()

        self.activation = activation
        self.linear = nn.Linear(int(input_size), int(output_size), bias=bias)


    def forward(self, x):
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)

        return h

class VampPrior(nn.Module):
    def __init__(self, xdim=2, init_mode="data", num_components=1, 
                 train_data_loader=None, learnable_contributions=False, device= 'cpu'):
        super(VampPrior, self).__init__()

        self.C = num_components
        self.type = "vamp"
        self.init_mode = init_mode
        
        self.means = NonLinear(num_components, np.prod(xdim), bias=False,
                                   activation=None)

        self.init_pseudo(train_data_loader, num_components)
        idle_input = Variable(torch.eye(num_components, num_components), requires_grad=False)
        self.register_buffer("idle_input", idle_input)

        if learnable_contributions:
            self.w =  nn.Parameter(torch.zeros(num_components,1,1), requires_grad=True)
        else:
            self.w = torch.zeros(num_components,1,1).to(device)


    def init_pseudo(self, train_loader, num_components):

        batched_data, _ = train_loader.next_batch(batch_size=512)
        batched_data = batched_data.data.cpu().numpy()
        np.random.shuffle(batched_data)
        xdim = np.prod(batched_data.shape[1:])

        
        if self.init_mode == "random":
            ## init from dataset stats
            data_mean, data_std = batched_data.mean(), batched_data.std()
            eps = np.random.randn(num_components, xdim) 
            pseudoinputs = data_mean + data_std * eps

        elif self.init_mode == "data":
            ## init from training data
            pseudoinputs = batched_data[:num_components]

        else:

            raise NotImplementedError("Init mode not implemented")

        pseudoinputs = torch.from_numpy(np.array(pseudoinputs,dtype=np.float32))
        self.means.linear.weight.data = pseudoinputs.T

        return

    def get_pseudoinputs(self):

        pseudoinputs = self.means(self.idle_input)

        return pseudoinputs


    def get_params(self):

        w_c = F.softmax(self.w, dim=0)
        w_c = w_c.squeeze()

        pseudoinputs = self.get_pseudoinputs()

        return pseudoinputs, w_c

class MoGPrior(nn.Module):
    """ Class realizing the MoG prior. """

    # Modified vertion https://github.com/jmtomczak/intro_dgm

    def __init__(self, encoder, zdim=2, num_components=1,  init_mode='data', 
                 learnable_contributions=False, train_data_loader=None, device= 'cpu'):
        super(MoGPrior, self).__init__()

        self.C = num_components
        self.type = "MoG"
        multiplier = 1 / np.sqrt(num_components * zdim)

        if init_mode == 'random':

            self.mu = nn.Parameter(torch.randn(num_components,zdim)*multiplier, requires_grad=True)
            self.logvar = nn.Parameter(torch.randn(num_components,zdim), requires_grad=True)
        
        elif init_mode == 'data':
            batched_data, _ = get_samples_from_dataloader(train_data_loader, num_components, device=device)
            batched_data = batched_data.to(encoder.main.input.weight.device) 
            batch_mu, batch_logvar = encoder(batched_data)
            self.mu = nn.Parameter(batch_mu, requires_grad=True)
            self.logvar = nn.Parameter(batch_logvar, requires_grad=True)

        else:
            raise NotImplementedError("Init mode not implemented")


        if learnable_contributions:
            self.w =  nn.Parameter(torch.zeros(num_components,1,1), requires_grad=True)
        else:
            self.w = torch.zeros(num_components,1,1).to(device)


    def get_params(self):
        """ Returning prior's parameters. """

        w_c = F.softmax(self.w, dim=0)
        w_c = w_c.squeeze()
        return self.mu, self.logvar, w_c

class ImposedPrior(nn.Module):
    """ Class realizing the Standard Gaussian prior. """


    def __init__(self, zdim=2, device= 'cpu', hardcoded=None):
        super(ImposedPrior, self).__init__()

        self.C = 1
        self.type = "imposed"


        if hardcoded:
            ## TODO hardcode prior
            self.mu = torch.zeros(1, zdim).to(device)
            self.logvar =torch.zeros(1, zdim).to(device)
            self.w =torch.zeros(1, 1, 1).to(device)

        else:
            self.mu = torch.zeros(1, zdim).to(device)
            self.logvar =torch.zeros(1, zdim).to(device)
            self.w =torch.zeros(1, 1, 1).to(device)

    def get_params(self):

        w_c = F.softmax(self.w, dim=0)
        w_c = w_c.squeeze()

        return self.mu, self.logvar, w_c

class EncoderSimple(nn.Module):
    def __init__(self, prior_mode, num_components, init_mode, learnable_contributions, xdim=2, zdim=2, n_layers=2, num_hidden=64, train_data_loader=None, device='cpu'):
        super(EncoderSimple, self).__init__()

        self.device = device
        self.xdim = xdim
        self.zdim = zdim
        self.n_layer = n_layers
        self.num_hidden = num_hidden
        self.main = nn.Sequential()
        self.main.add_module('input', nn.Linear(xdim, num_hidden))
        self.main.add_module('act0', nn.ReLU(True))
        for i in range(n_layers):
            self.main.add_module('hidden_%d' % (i + 1), nn.Linear(num_hidden, num_hidden))
            self.main.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.main.add_module('output', nn.Linear(num_hidden, zdim * 2))

        if prior_mode == 'imposed':
            self.prior = ImposedPrior(zdim, device)

        elif prior_mode == 'MoG':
            self.prior = MoGPrior(self, zdim=zdim, num_components=num_components, init_mode=init_mode, learnable_contributions=learnable_contributions, 
                                  train_data_loader=train_data_loader, device=device)

        elif prior_mode == 'vamp':
            assert train_data_loader != None, "No dataloader was given to initialize the vampprior"
            self.prior = VampPrior(xdim=xdim, num_components=num_components, init_mode=init_mode, learnable_contributions=learnable_contributions, 
                                   train_data_loader=train_data_loader, device=device)
                    
        else:       
            NotImplemented("Prior mode not implemented")

    
    
    def forward(self, x):
        y = self.main(x).view(x.size(0), -1)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar

class DecoderSimple(nn.Module):
    def __init__(self, x_dim=2, zdim=2, n_layers=2, num_hidden=64, device='cpu'):
        super(DecoderSimple, self).__init__()

        self.device = device

        self.xdim = x_dim
        self.zdim = zdim
        self.n_layer = n_layers
        self.num_hidden = num_hidden
        self.loggamma = nn.Parameter(torch.tensor(0.0))
        self.main = nn.Sequential()

        self.main.add_module('input', nn.Linear(zdim, num_hidden))
        self.main.add_module('act0', nn.ReLU(True))
        for i in range(n_layers):
            self.main.add_module('hidden_%d' % (i + 1), nn.Linear(num_hidden, num_hidden))
            self.main.add_module('act_%d' % (i + 1), nn.ReLU(True))
        self.main.add_module('output', nn.Linear(num_hidden, x_dim))

    def forward(self, z):
        self.main[8](self.main[7](self.main[6](self.main[5](self.main[4](self.main[3](self.main[2](self.main[1](self.main[0](z)))))))))


        z = z.view(z.size(0), -1)
        return self.main(z)

class SoftIntroVAESimple(nn.Module):
    def __init__(self, prior_mode='imposed', num_components=None, init_mode=None, learnable_contributions=None,
                  xdim=2, zdim=2, n_layers=2, num_hidden=64, device='cpu', pretrained=None, train_data_loader=None, clip_logvar=False):
        super(SoftIntroVAESimple, self).__init__()

        self.xdim = xdim
        self.zdim = zdim
        self.n_layer = n_layers
        self.num_hidden = num_hidden
        self.device = device

        self.encoder = EncoderSimple(prior_mode, num_components, init_mode, learnable_contributions, xdim, zdim, n_layers, num_hidden, device=device, train_data_loader=train_data_loader)
        self.decoder = DecoderSimple(xdim, zdim, n_layers, num_hidden, device)


        if clip_logvar:
           self.init_clip_logvar()
        else:
            self.clip_logvar = False

        self.to(device)

        if pretrained is not None:
            weights = torch.load(pretrained)
            pretrained_dict = weights['model']
            self.load_state_dict(pretrained_dict)


    def forward(self, x, deterministic=False):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = reparameterize(mu, logvar)
        y = self.decode(z)
        return mu, logvar, z, y


    def init_clip_logvar(self, clip_logvar_min=None, clip_logvar_max=None):

        self.clip_logvar = True

        if clip_logvar_min is None:
            ## randomly initialized
            self.clip_logvar_min = nn.Parameter(torch.randn(self.zdim))
            self.clip_logvar_max = nn.Parameter(torch.randn(self.zdim))
        else:
            ## provided values
            self.clip_logvar_min =  nn.Parameter(clip_logvar_min)
            self.clip_logvar_max =  nn.Parameter(clip_logvar_max)

        self.clip_logvar_min.requires_grad = False
        self.clip_logvar_max.requires_grad = False

        return

    def vamp_to_mog(self, num_components, learnable_contributions, 
                    optimizer_e, optimizer_d, optimizer_p,
                    e_scheduler, d_scheduler, p_scheduler, 
                    milestones, lr, prior_lr, optim_betas_enc_dec, optim_betas_prior):

        ## get prior
        pseudoinputs, w_c = self.encoder.prior.get_params()
        mu_MoG, logvar_MoG = self.encoder(pseudoinputs)
        self.encoder.prior = MoGPrior(self.encoder, zdim=mu_MoG.shape[1], num_components=num_components, init_mode='random',
                                      learnable_contributions=learnable_contributions, device=pseudoinputs.device)

        ## initialize MoG with vamp
        prior_weight_dict = self.encoder.prior.state_dict()
        prior_weight_dict['mu'] = mu_MoG
        prior_weight_dict['logvar'] = logvar_MoG

        if learnable_contributions:
            prior_weight_dict['w'] = w_c.view(*w_c.shape,1,1)
        
        self.encoder.prior.load_state_dict(prior_weight_dict)
        self.encoder.prior.to(pseudoinputs.device)

        optimizer_p = optim.Adam(self.encoder.prior.parameters(), lr=prior_lr, betas=optim_betas_prior)
        optimizer_e = optim.Adam(self.encoder.main.parameters(), lr=lr, betas=optim_betas_enc_dec)
        optimizer_d = optim.Adam(self.decoder.parameters(), lr=lr, betas=optim_betas_enc_dec)

        e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=milestones, gamma=0.1, verbose=False) 
        d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=milestones, gamma=0.1, verbose=False)
        p_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=milestones , gamma=0.1, verbose=False)

        return optimizer_e, optimizer_d, optimizer_p, e_scheduler, d_scheduler, p_scheduler


    def get_prior_params(self):
                   
        if self.encoder.prior.type != 'vamp':
            # getting prior parameters (mu, logvar)
            mu_MoG, logvar_MoG, w_c = self.encoder.prior.get_params()
        else:
            # getting vamp prior parameters (mu, logvar)
            pseudoinputs, w_c = self.encoder.prior.get_params()
            mu_MoG, logvar_MoG = self.encoder(pseudoinputs) 


        if self.clip_logvar:
            ## clipping logvar similar to Chang, Bo, et al. "Latent user intent modeling for sequential recommenders."

            delta = (self.clip_logvar_max - self.clip_logvar_min).abs()
            
            beta = 10/(delta + 1e-8) # beta adapting to the range delta, epsilon to avoid division by zero

            softplus_max = softplus(logvar_MoG - self.clip_logvar_max, beta)
            softplus_min = softplus(self.clip_logvar_min - logvar_MoG, beta)

            # Element-wise equality condition i.e. max = min
            condition = (self.clip_logvar_max == self.clip_logvar_min)
            logvar_MoG = torch.where(condition, self.clip_logvar_max, logvar_MoG  - softplus_max + softplus_min)

            
        return mu_MoG, logvar_MoG, w_c

    def sample_noise(self, num_random_samples, with_grad=False, ret_ind=False, deterministic=False, detach_variance=False):
        

        with torch.set_grad_enabled(with_grad):

            mu_MoG, logvar_MoG, w_c = self.get_prior_params()

            if detach_variance:
                logvar_MoG = logvar_MoG.detach()

            # sampling mixture component
            if self.encoder.prior.C > 1:
                indices = torch.multinomial(w_c, num_random_samples, replacement=True)
            else:
                # single component [e.g. imposed N(0,I) prior]
                indices = torch.zeros(num_random_samples, device=w_c.device, dtype=torch.long)
                
            std = torch.exp(0.5 * logvar_MoG[indices,:])
            eps = torch.randn(size=logvar_MoG[indices,:].shape).to(w_c.device)
            noise_batch = mu_MoG[indices,:] + eps * std

            if deterministic:
                noise_batch = mu_MoG[indices,:]
            else:
                noise_batch = mu_MoG[indices,:] + eps * std
            
            if ret_ind:
                return noise_batch, indices
            else:
                return noise_batch


    def sample(self, z):
        y = self.decode(z)
        return y


    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y


    
