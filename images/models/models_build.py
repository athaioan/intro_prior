import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os 
import sys
import math 
import torch.nn.functional as F
import torch.optim as optim

from dataset import get_samples_from_dataloader

from utils import log_Normal_diag, reparameterize, softplus


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
                 learnable_contributions=False, train_data_loader=None,
                 device='cpu', normalized_logvars=False):
        super(VampPrior, self).__init__()

        self.C = num_components
        self.type = "vamp"
        self.init_mode = init_mode
        self.normalized_logvars = normalized_logvars

        
        self.means = NonLinear(num_components, np.prod(xdim), bias=False,
                                   activation=nn.Hardtanh(min_val=0.0, max_val=1.0)) ## image pseudoinputs from [0,1]

        self.init_pseudo(train_data_loader, num_components)
        idle_input = Variable(torch.eye(num_components, num_components), requires_grad=False)
        self.register_buffer("idle_input", idle_input)

        if learnable_contributions:
            self.w =  nn.Parameter(torch.zeros(num_components,1,1), requires_grad=True)
        else:
            self.w = torch.zeros(num_components,1,1).to(device)

    def init_pseudo(self, train_data_loader, num_components):
        
        len_dataset = len(train_data_loader.dataset)
        n_samples = len_dataset if self.init_mode == "random" else num_components

        batched_data, _ = get_samples_from_dataloader(train_data_loader, n_samples)
        batched_data = batched_data.data.cpu().numpy()
        np.random.shuffle(batched_data)
        xdim = np.prod(batched_data.shape[1:])
        
        if self.init_mode == "random":
            ## init from dataset stats
            data_mean, data_std = batched_data.mean(), batched_data.std()
            eps = np.random.randn(num_components, xdim) 
            pseudoinputs = data_mean + data_std * eps
            ## clipping to image range values
            pseudoinputs = pseudoinputs.clip(0,1)

        elif self.init_mode == "data":
            ## init from training data
            pseudoinputs = batched_data[:num_components]
            ## reshape pseudoinputs
            pseudoinputs = pseudoinputs.reshape(num_components, xdim)

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
    def __init__(self, encoder, zdim=2, num_components=1, init_mode='data', 
                 learnable_contributions=False, train_data_loader=None, device='cpu',
                 normalized_logvars=False) :
        super(MoGPrior, self).__init__()

        self.C = num_components
        self.type = "MoG"

        multiplier = 1 / np.sqrt(num_components * zdim)

        if init_mode == 'random':

            self.mu = nn.Parameter(torch.randn(num_components, zdim)*multiplier, requires_grad=True)
            self.logvar =  nn.Parameter(torch.randn(num_components, zdim), requires_grad=True) if not(normalized_logvars) else torch.zeros((num_components, zdim)).to(device)
            
        elif init_mode == 'data':
            batched_data, _ = get_samples_from_dataloader(train_data_loader, num_components)
            batched_data = batched_data.to(encoder.fc.weight.device) 
            batch_mu, batch_logvar = encoder(batched_data)
            self.mu = nn.Parameter(batch_mu, requires_grad=True)
            self.logvar = nn.Parameter(batch_logvar, requires_grad=True) if not(normalized_logvars) else torch.zeros((num_components, zdim)).to(device)

        else:
            raise NotImplementedError("Init mode not implemented")


        if learnable_contributions:
            self.w =  nn.Parameter(torch.zeros(num_components,1,1), requires_grad=True)
        else:
            self.w = torch.zeros(num_components,1,1).to(device)


    def get_params(self):

        w_c = F.softmax(self.w, dim=0)
        w_c = w_c.squeeze()

        return self.mu, self.logvar, w_c

class ImposedPrior(nn.Module):
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
            self.logvar = torch.zeros(1, zdim).to(device)
            self.w =torch.zeros(1, 1, 1).to(device)

    def get_params(self):

        w_c = F.softmax(self.w, dim=0)
        w_c = w_c.squeeze()

        return self.mu, self.logvar, w_c


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    "if inc is not outc" -> "if inc != outc"
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc != outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, prior_mode, num_components, init_mode, learnable_contributions, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, 
                 train_data_loader=None, device='cpu', normalized_logvars=False):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        self.fc = nn.Linear(num_fc_features, 2 * zdim)

        # placeholder for the gradients
        self.gradients = None

        if prior_mode == 'imposed':
            self.prior = ImposedPrior(zdim, device=device)

        elif prior_mode == 'MoG':
            self.prior = MoGPrior(self, zdim=zdim, num_components=num_components, init_mode=init_mode, learnable_contributions=learnable_contributions, 
                                  train_data_loader=train_data_loader, device=device, normalized_logvars=normalized_logvars)
            
        elif prior_mode == 'vamp':
            assert train_data_loader != None, "No dataloader was given to initialize the vampprior"
            xdim = image_size**2 * cdim
            self.prior = VampPrior(xdim=xdim, num_components=num_components, init_mode=init_mode, learnable_contributions=learnable_contributions,
                                   train_data_loader=train_data_loader, device=device, normalized_logvars=normalized_logvars)

        else:
            NotImplemented("Prior mode not implemented")


    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients

    def forward(self, x):

        y = self.main(x)

        if y.requires_grad:
            # register the hook: https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
            h = y.register_hook(self.activations_hook)

        y = self.fc(y.view(x.size(0), -1))
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, 
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        self.fc = nn.Sequential(
            nn.Linear(zdim, num_fc_features),
            nn.ReLU(True),
        )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
              
        y = self.main(y)

        return y
    
    def forward_seq(self, z, id=0):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        
        import matplotlib.pyplot as plt
        for index, layer in enumerate(self.main):
            y = layer(y)
            plt.imshow(y[id].mean(axis=0).data.cpu().numpy())
            plt.savefig('test_{}'.format(index))

        return y


class SoftIntroVAE(nn.Module):
    def __init__(self, prior_mode='imposed', num_components=None, init_mode=None, learnable_contributions=None, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256,
                 device='cpu', train_data_loader=None, pretrained=None, normalized_logvars=False, clip_logvar=False):
        super(SoftIntroVAE, self).__init__()

        self.zdim = zdim
        self.cdim = cdim ## number of channels
        self.image_size = image_size

        self.encoder = Encoder(prior_mode, num_components, init_mode, learnable_contributions, cdim, zdim, channels, image_size, 
                                device=device, train_data_loader=train_data_loader, normalized_logvars=normalized_logvars)

        self.decoder = Decoder(cdim, zdim, channels, image_size,
                               conv_input_size=self.encoder.conv_output_size)
        
        if clip_logvar:
           self.init_clip_logvar()
        else:
            self.clip_logvar = False
        
        self.to(device)

        if pretrained is not None:
            weights = torch.load(pretrained,  map_location=device)
            pretrained_dict = weights['model']
            self.load_state_dict(pretrained_dict)
            print("loaded weights")
            print(pretrained)


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
    
    def reset_prior_params(self, mu, logvar, w, std_coeff=0.1):

        num_components = mu.shape[0]

        contributions = F.softmax(w, dim=0)
        num_inactive = (contributions < (1/num_components)**2).sum()

        inactive_ids = (contributions < (1/num_components)**2).flatten().nonzero().squeeze()
        max_idx = contributions.argmax()
        
        contributions[inactive_ids] = contributions[max_idx] / (num_inactive + 1)
        contributions[max_idx] = contributions[max_idx] / (num_inactive + 1)

        std = std_coeff * torch.exp(0.5 * logvar[max_idx])

        eps = torch.randn_like(mu[inactive_ids]).to(mu.device)

        mu[inactive_ids] = mu[max_idx] + eps * std
        logvar[inactive_ids] = logvar[max_idx].repeat(num_inactive, 1)

        # https://math.stackexchange.com/questions/2786600/invert-the-softmax-function
        C = 10
        w = torch.log(contributions) + C

        return (mu, logvar, w)
    
    def reset_inactive_modes(self, num_components,
                            milestones, prior_lr, logvar_lr_ratio, optim_betas_prior, normalized_logvars=False,):


        ## get prior
        mu_MoG = self.encoder.prior.mu.detach().clone()
        logvar_MoG = self.encoder.prior.logvar.detach().clone()
        w_c = self.encoder.prior.w.detach().clone()

        self.encoder.prior = MoGPrior(self.encoder, zdim=mu_MoG.shape[1], num_components=num_components, init_mode='random',
                                      learnable_contributions=True, train_data_loader=None, device=mu_MoG.device, 
                                      normalized_logvars=normalized_logvars)


        # # ## initialize MoG with vamp
        prior_weight_dict = self.encoder.prior.state_dict()
        mu_MoG, logvar_MoG, w_c = self.reset_prior_params(mu_MoG, logvar_MoG, w_c)

        prior_weight_dict['mu'] = mu_MoG
        prior_weight_dict['w'] = w_c


        if not(normalized_logvars):
            prior_weight_dict['logvar'] = logvar_MoG


        self.encoder.prior.load_state_dict(prior_weight_dict)

        self.encoder.prior.to(mu_MoG.device)

        optimizer_p = optim.Adam(
                                 [{"params": self.encoder.prior.mu, 'lr': prior_lr},
                                  {"params": self.encoder.prior.w, 'lr': prior_lr},
                                  {"params": self.encoder.prior.logvar, 'lr': logvar_lr_ratio * prior_lr},
                                 ], betas=optim_betas_prior)
        
        p_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_p, milestones=milestones, gamma=0.1)


        return optimizer_p, p_scheduler

    def vamp_to_mog(self, num_components, learnable_contributions, 
                    optimizer_e, optimizer_d, optimizer_p,
                    e_scheduler, d_scheduler, p_scheduler, 
                    milestones, lr, prior_lr, optim_betas_enc_dec, optim_betas_prior, normalized_logvars=False,
                    ):

        ## get prior
        num_c = self.encoder.prior.C
        mu_MoG, logvar_MoG, w_c = self.get_prior_params()
        
        self.encoder.prior = MoGPrior(self.encoder, zdim=mu_MoG.shape[1], num_components=num_components, init_mode='random',
                                      learnable_contributions=learnable_contributions, train_data_loader=None, device=mu_MoG.device, 
                                      normalized_logvars=normalized_logvars)

        ## initialize MoG with vamp
        prior_weight_dict = self.encoder.prior.state_dict()
        prior_weight_dict['mu'] = mu_MoG

        if not(normalized_logvars):
            prior_weight_dict['logvar'] = logvar_MoG

        if learnable_contributions:
            if num_c == 1:
                prior_weight_dict['w'] = w_c.view(*w_c.shape,1,1).unsqueeze(0)

            else:

                prior_weight_dict['w'] = w_c.view(*w_c.shape,1,1)
        
        self.encoder.prior.load_state_dict(prior_weight_dict)
        self.encoder.prior.to(mu_MoG.device)


        optimizer_p = optim.Adam(self.encoder.prior.parameters(), lr=prior_lr, betas=optim_betas_prior)
        optimizer_e = optim.Adam(list(self.encoder.main.parameters()) + list(self.encoder.fc.parameters()), lr=lr, betas=optim_betas_enc_dec)
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
            num_c = self.encoder.prior.C
            pseudoinputs = pseudoinputs.view(num_c, self.cdim, self.image_size, self.image_size)
            mu_MoG, logvar_MoG = self.encoder(pseudoinputs) 
            if self.encoder.prior.normalized_logvars:
                logvar_MoG = torch.zeros_like(logvar_MoG)

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
            # sampling with grad or w/o grad wrt to the prior
            
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

            if deterministic:
                noise_batch = mu_MoG[indices,:]
            else:
                noise_batch = mu_MoG[indices,:] + eps * std
            
            if ret_ind:
                return noise_batch, indices
            else:
                return noise_batch


    def generate_samples(self, num_random_samples):

            noise_batch = self.sample_noise(num_random_samples)
            fake_batch = self.sample(noise_batch).data.cpu().numpy()
            fake_batch = np.clip(fake_batch * 255, 0, 255).astype(np.uint8) / 255.0
            fake_batch = torch.from_numpy(fake_batch).type(torch.FloatTensor)
            fake_batch = fake_batch.to(noise_batch.device)

            return fake_batch
    
    def extract_pseudoinputs(self):
        
        ## generating pseudoinputs
        assert  self.encoder.prior.type == "vamp", "generating pseudoinputs is only applicable for VampPrior"

        num_C = self.encoder.prior.C 
        pseudoinputs = self.encoder.prior.get_pseudoinputs()
        pseudoinputs = pseudoinputs.reshape(num_C, self.cdim, 
                                            self.image_size, 
                                            self.image_size)
        
        return pseudoinputs


    def sample(self, z):
        y = self.decode(z)
        return y

    def encode(self, x):
        
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):

        y = self.decoder(z)
        return y
