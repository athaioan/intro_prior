import os 

import numpy as np
from scipy.stats import multivariate_normal

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from utils import calc_reconstruction_loss, calc_kl_loss



# based on https://github.com/taldatech/soft-intro-vae-pytorch

## Quantitative result utilities
def setup_grid(range_lim=4, n_pts=1000, device=torch.device("cpu")):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)


def calculate_elbo_with_grid(model, evalset, test_grid,  kl_loss_type='stochastic', MC=100, beta_kl=1.0, beta_recon=1.0, batch_size=512, num_iter=100,
                             device=torch.device("cpu")):
    _, _, zz = test_grid
    zzk = []
    elbos = []

    ## repeatable eps
    eps = torch.randn(MC, model.zdim).to(device)


    with torch.no_grad():
        for zz_i in zz.split(batch_size, dim=0):
            curr_batch_size = zz_i.shape[0]
            zz_i = zz_i.to(device)
            mu, logvar, _, rec = model(zz_i, deterministic=True)
            recon_error = calc_reconstruction_loss(zz_i, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)

            z_s = mu.repeat(MC,1) + torch.exp(0.5 * logvar.repeat(MC,1)) * eps.repeat_interleave(curr_batch_size, dim=0)
            ## to avoid nan when doing exp(-logvar) which happends for some hyperparams during grid - search
            clipped_logvar = logvar.clamp(-80) 
            kl = calc_kl_loss(model, mu, clipped_logvar, z_s=z_s,
                              mc=MC, kl_loss_type=kl_loss_type, reduction='none')
            
            zzk_i = 1.0 * (beta_kl * kl + beta_recon * recon_error)
            zzk += [zzk_i]
        elbos_grid = torch.cat(zzk, 0)

        for _ in range(num_iter):
            batch, _ = evalset.next_batch(batch_size=batch_size, device=device)
            curr_batch_size = batch.shape[0]
            mu, logvar, _, rec = model(batch, deterministic=True)
            recon_error = calc_reconstruction_loss(batch, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)

            z_s = mu.repeat(MC,1) + torch.exp(0.5 * logvar.repeat(MC,1)) * eps.repeat_interleave(curr_batch_size, dim=0)
            ## to avoid nan when doing exp(-logvar) which happends for some hyperparams during grid - search
            clipped_logvar = logvar.clamp(-80) 
            kl = calc_kl_loss(model, mu, clipped_logvar, z_s=z_s,
                              mc=MC, kl_loss_type=kl_loss_type, reduction='none')


            elbos += [1.0 * (beta_kl * kl + beta_recon * recon_error)]

    elbos = torch.cat(elbos, dim=0)
    normalizing_factor = torch.cat([elbos_grid, elbos], dim=0).sum()
    elbos = elbos / normalizing_factor
    return elbos.mean().data.cpu().item()



def calculate_sample_kl(model, evalset, num_samples=5000, device=torch.device("cpu"), hist_bins=100, use_jsd=False,
                        xy_range=(-2, 2)):
    hist_range = [[xy_range[0], xy_range[1]], [xy_range[0], xy_range[1]]]
    real_samples, _ = evalset.next_batch(batch_size=num_samples, device=device)
    real_samples = real_samples.data.cpu().numpy()
    real_hist, _, _ = np.histogram2d(real_samples[:, 0], real_samples[:, 1], bins=hist_bins, density=True,
                                     range=hist_range)
    real_hist = torch.tensor(real_hist).to(device)
    noise_batch = model.sample_noise(num_samples)         
    fake_samples = model.sample(noise_batch).data.cpu().numpy()

    fake_hist, _, _ = np.histogram2d(fake_samples[:, 0], fake_samples[:, 1], bins=hist_bins, density=True,
                                     range=hist_range)
    fake_hist = torch.tensor(fake_hist).to(device)
 
    if use_jsd:
        # sample symmetric KL
        kl_1 = F.kl_div(torch.log(real_hist + 1e-14), 0.5 * (fake_hist + real_hist), reduction='batchmean')
        kl_2 = F.kl_div(torch.log(fake_hist + 1e-14), 0.5 * (fake_hist + real_hist), reduction='batchmean')
        jsd = 0.5 * (kl_1 + kl_2)
        return jsd.data.cpu().item()

    else:
        
        # sample KL
        kl = F.kl_div(torch.log(fake_hist + 1e-14), real_hist, reduction='batchmean') 

        # constructing uniform histogram in density format (similar to fake_hist and real_hist)
        bin_area = ((xy_range[1] - xy_range[0])/ hist_bins)**2

        uniform_hist_count = np.ones_like(fake_hist.data.cpu().numpy()) 
        uniform_hist_prob = uniform_hist_count / uniform_hist_count.sum()
        uniform_hist = uniform_hist_prob / bin_area # in density format
    
        uniform_hist = torch.tensor(uniform_hist).to(device)

        # computing sample kl between fake and uniform as a proxy of H[pd(x)] (i.e. lower entropy -> more peaky -> less uniform -> higher kl_uniform)
        kl_uniform =  F.kl_div(torch.log(fake_hist + 1e-14), uniform_hist, reduction='batchmean') 


        return kl.data.cpu().item(), kl_uniform.data.cpu().item()

def get_quantitatives(model, f_name, train_set, scale, kl_loss_type, MC, beta_kl, beta_rec, beta_neg, seed,):
    model.eval()

    res = {}
    res['sample_kl'], res['sample_entropy'] = calculate_sample_kl(model, train_set, num_samples=5000, device=model.device, hist_bins=100,
                                           use_jsd=False, xy_range=(-2 * scale, 2 * scale))
    res['jsd'] = calculate_sample_kl(model, train_set, num_samples=5000, device=model.device, hist_bins=100,
                                     use_jsd=True, xy_range=(-2 * scale, 2 * scale))
    test_grid = setup_grid(range_lim=scale * 2, n_pts=1024, device=model.device)
    res['elbo'] = calculate_elbo_with_grid(model, train_set, test_grid=test_grid, kl_loss_type=kl_loss_type, MC=MC, beta_kl=1.0,
                                           beta_recon=1.0,
                                           device=model.device, batch_size=128)
    print("#" * 50)
    print(f'beta_kl: {beta_kl}, beta_rec: {beta_rec}, beta_neg: {beta_neg}')
    print(f'grid-normalized elbo: {res["elbo"]:.4e}, kl: {res["sample_kl"]:.4f}, jsd: {res["jsd"]:.4f} sample_entropy: {res["sample_entropy"]:.4f}')
    print("#" * 50)
    with open(os.path.join(f_name, 'results_log_soft_intro_vae.txt'), 'a') as fp:
        line = f'_beta_kl_{beta_kl}_beta_neg_{beta_neg}_beta_rec_{beta_rec}_gnelbo_{res["elbo"]}_kl_{res["sample_kl"]}_jsd_{res["jsd"]}_entropy_{res["sample_entropy"]}_seed_{seed}\n'
        fp.write(line)

    model.train()
    return

## Qualitative result utilities
cmap = plt.get_cmap("tab20")

def plot_vae_density(model, test_grid, n_pts, batch_size, beta_kl=1.0,
                     beta_recon=1.0,  kl_loss_type='stochastic', MC=100,
                     device=torch.device('cpu'), 
                     save_path="", it=0, set_title=True):
    """ plots square grid and vae density """

    ## plotting exp(ELBO)
    print("plotting density...")
    model.eval()
    xx, yy, zz = test_grid  ## setting the density grid
    # compute posterior approx density
    # p(x) = E_{z~p(z)}[q(z|x)]
    zzk = [[] for _ in range(3)]

    ## repeatable eps
    eps = torch.randn(MC, model.zdim).to(device)

    with torch.no_grad():
        for zz_i in zz.split(batch_size, dim=0):
            curr_batch_size = zz_i.shape[0]
            zz_i = zz_i.to(device)
            mu, logvar, _, rec = model(zz_i, deterministic=True) 
            recon_error = calc_reconstruction_loss(zz_i, rec, loss_type='mse', reduction='none')
            while len(recon_error.shape) > 1:
                recon_error = recon_error.sum(-1)

            ## repeatable reparametrization
            z_s = mu.repeat(MC,1) + torch.exp(0.5 * logvar.repeat(MC,1)) * eps.repeat_interleave(curr_batch_size, dim=0)

            ## to avoid nan when doing exp(-logvar) which can happend for some hyperparams during grid - search
            clipped_logvar = logvar.clamp(-80) 
            kl = calc_kl_loss(model, mu, clipped_logvar, z_s=z_s,
                              mc=MC, kl_loss_type=kl_loss_type, reduction='none')
                        
            zzk_KL = -1.0 * (beta_kl * kl)  ## density approximation via ELBO (see pg.6)
            zzk_RE = -1.0 * (beta_recon * recon_error)  ## density approximation via ELBO (see pg.6)

            zzk[0] += [(zzk_RE+zzk_KL).exp()]
            zzk[1] += [(zzk_KL).exp()]
            zzk[2] += [(zzk_RE).exp()]


    fig_names = ['density', 'density_KL', 'density_rec']
                                 
    for index, fig_name in enumerate(fig_names):        


        _, ax = plt.subplots(1, 1, figsize=(6, 6))

        p_x = torch.cat(zzk[index], 0)
        p_x = p_x.view(n_pts, n_pts).data.cpu().numpy()

        # plot
        cmesh = ax.pcolormesh(xx.data.cpu().numpy(), yy.data.cpu().numpy(), p_x,
                     cmap=plt.cm.jet)
        ax.set_facecolor(plt.cm.jet(0.))

        ax.set_axis_off()

        if set_title:
            plt.title(fig_name)
            plt.colorbar(cmesh)

        plt.savefig(os.path.join(save_path,"{}_{}.png".format(fig_name, it)), bbox_inches='tight')
        plt.close()


    return

def plot_generated_data(model, f_name, it, num_random_samples=1024, set_title=True):

    # Plotting generated data 
    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    noise_batch = model.sample_noise(num_random_samples)

    fake_batch = model.sample(noise_batch)
    fake_batch_ = fake_batch.data.cpu().numpy()
    ax.scatter(fake_batch_[:, 0], fake_batch_[:, 1], s=8, c='g', cmap=cmap, label="generated")
    plt.axis('off')

    if set_title:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
        plt.title('Generated data') 

    plt.savefig(os.path.join(f_name,"sample_{}.png".format(it)), bbox_inches='tight')
    plt.close()

    return fake_batch

def plot_real_data(model, train_set, f_name, it, set_title=True, with_ood=False):

    real_batch, real_labels = train_set.next_batch(batch_size=1024, device=model.device)
    odd_batch, odd_labels = train_set.next_batch(batch_size=1024, device=model.device,
                                                   return_ood=True)

    ## Reconstructed data
    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    real_batch_ = real_batch.data.cpu().numpy()
    real_labels = real_labels.data.cpu().numpy()

    odd_batch_ = odd_batch.data.cpu().numpy()
    odd_labels = odd_labels.data.cpu().numpy()


    ax.scatter(real_batch_[:, 0], real_batch_[:, 1], s=12, label="true", cmap=cmap, marker='x', c=real_labels)
    if with_ood:
        ax.scatter(odd_batch_[:, 0], odd_batch_[:, 1], s=12, label="ood", cmap=cmap, marker='o', alpha=0.3, c=odd_labels)

    if model.encoder.prior.type == "vamp":
            # pseudo_inputs = model.encoder.means(model.encoder.idle_input)
            pseudo_inputs = model.encoder.prior.get_pseudoinputs()
            pseudo_inputs = pseudo_inputs.data.cpu().numpy()
            ax.scatter(pseudo_inputs[:, 0], pseudo_inputs[:, 1], marker='*',  c='black', label="pseudo_inputs", alpha=0.5)


    ax.set_axis_off()
    
    if set_title:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
        plt.title('Real data') 


    plt.savefig(os.path.join(f_name,"data_{}.png".format(it)), bbox_inches='tight')
    plt.close()

    return real_batch, real_labels, odd_batch, odd_labels

def get_prior_density(model, mu, logvar, w_c, X_grid, Y_grid, with_numpy=True):

    Z = np.zeros_like(X_grid)
    grid = np.stack((X_grid, Y_grid), axis=2)

    for i_c in range(model.encoder.prior.C):

        curr_mu = mu[i_c].data.cpu().numpy().tolist()
        current_var = logvar.exp()[i_c].data.cpu().numpy().tolist()

        curr_w = np.array([1])  if model.encoder.prior.C == 1 else w_c[i_c].data.cpu().numpy()
        
        try:
            if with_numpy:
                ## numpy

                diff = grid - curr_mu
                inv_cov = np.reciprocal(current_var)
                exponent = np.sum(np.dot(diff, np.diag(inv_cov)) * diff, axis=2)
                Z += curr_w*np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(np.prod(current_var)))

            else:
                ## scipy
                rv = multivariate_normal(curr_mu, [[current_var[0], 0], [0, current_var[1]]])
                # Probability Density
                pos = np.empty(X_grid.shape + (2,))
                pos[:, :, 0] = X_grid
                pos[:, :, 1] = Y_grid
                Z += curr_w*rv.pdf(pos)
        except:
            pass

    return X_grid, Y_grid, Z 

def plot_latent(prior_density, mu, w_c, data, labels, legends, f_name=" ", im_name=" ", it=0, set_title=True):

    ## plotting latent space (real_dist)
    _, ax = plt.subplots(1, 1, figsize=(6, 6))

    markers = ['x','o','+']
    for data_index in range(len(data)):

        ax.scatter(data[data_index][:, 0],data[data_index][:, 1], s=20, 
                   marker=markers[data_index], alpha=1/(1+data_index), c=labels[data_index], cmap=cmap, label = legends[data_index])


    ## plotting prior's means
    C = mu.shape[0]
    for i_c in range(mu.shape[0]):
        
        curr_mu = mu[i_c].data.cpu().numpy()
        curr_w = np.array([1])  if C == 1 else w_c[i_c].data.cpu().numpy()

        ax.plot(curr_mu[0], curr_mu[1],'ok', markersize=(curr_w*100).clip(1,15))

        if curr_w < 1/(C**2):
            ## plotting dead prior with "+"
            ax.plot(curr_mu[0], curr_mu[1],'+k', markersize=(curr_w*100).clip(1,15))
    
    ax.set_axis_off()


    if set_title:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
        plt.title(im_name) 

    X, Y, Z = prior_density
    ax.contourf(X, Y, Z, cmap='Blues', alpha=0.5)
    plt.savefig(os.path.join(f_name,"{}_{}.png".format(im_name, it)), bbox_inches='tight')
    plt.close()

    return 

def get_qualitatives(model, f_name, train_set, scale, kl_loss_type, MC,
                      it=0, set_title=True, delete_figures=True, with_ood=False):

    model.eval()

    ##############
    # plotting in and out of distribution data
    real_batch, real_labels, odd_batch, odd_labels = plot_real_data(model, train_set, f_name, it=it,
                                                                     set_title=set_title, with_ood=with_ood)

    # plotting generated data
    fake_batch = plot_generated_data(model, f_name, it=it, num_random_samples = 1024, set_title=set_title) 
    fake_labels = np.ones(len(fake_batch)) * (np.max(real_labels)+1)

    # plotting density (via ELBO)
    n_pts = 1024 
    test_grid = setup_grid(range_lim=scale * 2, n_pts=n_pts, device=model.device)
    plot_vae_density(model, test_grid, n_pts=n_pts, batch_size=256,
                    beta_kl=1.0, beta_recon=1.0, kl_loss_type=kl_loss_type, MC=MC,
                    device=model.device,
                    save_path=f_name, it=it, set_title=set_title)
    
    # plotting latent space
    mu, logvar, w_c = model.get_prior_params()

    _, _, z_s, rec_batch = model(real_batch, deterministic=True)
    real_batch_latent_ = z_s.data.cpu().numpy()


    _, _, z_s, _ = model(rec_batch, deterministic=True)
    rec_batch_latent_ = z_s.data.cpu().numpy()

    _, _, z_s, _ = model(fake_batch, deterministic=True)
    fake_batch_latent_ = z_s.data.cpu().numpy()

    latent_data = np.vstack((real_batch_latent_, rec_batch_latent_, 
                                   fake_batch_latent_,
                                   mu.data.cpu().numpy()))

    if with_ood:
        _, _, z_s, _ = model(odd_batch, deterministic=True)
        odd_batch_latent_ = z_s.data.cpu().numpy()
        latent_data = np.vstack((latent_data, odd_batch_latent_))


    min_x, max_x = np.min(latent_data[:,0]), np.max(latent_data[:,0])
    min_y, max_y = np.min(latent_data[:,1]), np.max(latent_data[:,1])

    X_grid = np.linspace(min_x - 0.1*np.abs(min_x), max_x + 0.1*np.abs(max_x), 1024)
    Y_grid = np.linspace(min_y - 0.1*np.abs(min_y), max_y + 0.1*np.abs(max_y), 1024)
    X_grid, Y_grid = np.meshgrid(X_grid, Y_grid)

    prior_density = get_prior_density(model, mu, logvar, w_c, X_grid, Y_grid)

    plot_latent(prior_density, mu, w_c, data=[real_batch_latent_, rec_batch_latent_, fake_batch_latent_], 
                        labels=[real_labels, real_labels, fake_labels,],
                        legends=['real','reconstructed', 'generated',], f_name=f_name, im_name="latent_real_fake", it=it, set_title=set_title)
    
    if with_ood:
        plot_latent(prior_density, mu, w_c, data=[real_batch_latent_, odd_batch_latent_], 
                            labels=[real_labels, odd_labels], 
                            legends = ['real','ood'] ,f_name=f_name, im_name="latent_real_ood", it=it, set_title=set_title)

    model.train()

    figures_include = ['sample', 'data', 'latent_real_fake', 'density', 'density_KL', 'density_rec']

    if with_ood:
        figures_include.insert(3, 'latent_real_ood')

    try:
        get_summary(f_name, figures_include, it=it, delete_figures=delete_figures)
    except:
        print("Figures not found")
    return

def get_summary(figures_dir, figures_include, it=0, delete_figures=True):

    text_width = 150


    img_files = [os.path.join(figures_dir,fig_name+"_{}.png".format(it)) for fig_name in figures_include]


    # taken from https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python 
    images = [Image.open(fig_name) for fig_name in img_files]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (text_width + total_width, max_height), (255, 255, 255))

    x_offset = text_width
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(os.path.join(figures_dir,"summary_{}.png".format(it)))

    if delete_figures:
        ## delete figures
        img_files = [os.path.join(figures_dir,fig_name+"_{}.png".format(it)) for fig_name in figures_include]
        for img in img_files:
            if os.path.exists(img):
                    os.remove(img)
    return
