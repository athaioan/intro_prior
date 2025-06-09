import os

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.manifold import TSNE
import cv2

from utils.losses import calc_reconstruction_loss, calc_kl_loss, reparameterize, get_mog_logvar
import scipy.stats as stats

import math

def zero_one_scaling(image: np.ndarray) -> np.ndarray:
    """Scales an image to range [0, 1]."""

    C = image.shape[0]
    if C == 1:
        image = np.tile(image, (3, 1, 1))

    if np.all(image == 0):
        return image
    image = image.astype(np.float32)
    return (image - image.min()) / (image.max() - image.min())

def show_sensitivity_on_image(image: np.ndarray,
                              sensitivity_map: np.ndarray,
                              colormap: int = cv2.COLORMAP_PARULA,
                              heatmap_weight: float = 2.0) -> np.ndarray:
    """Overlay the sensitivity map on the image."""
    # Convert sensitivity map to a heatmap.
    heatmap = cv2.applyColorMap(sensitivity_map, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap.astype(np.float32) / 255

    # Overlay original RGB image and heatmap with specified weights.
    scaled_image = zero_one_scaling(image=image)
    overlay = heatmap_weight * heatmap.transpose(2, 0, 1) + scaled_image
    overlay = zero_one_scaling(image=overlay)

    return np.clip(overlay, 0.0, 1.0).astype(np.float32)

def plot_explainability(model, train_data_loader, save_dir, beta_rec=1.0, beta_kl=1.0, MC=10, recon_loss_type='mse', kl_loss_type='stochastic',
                                          batch_size=16, it=0, nrow=16, device='cpu', mu_MoG=None, logvar_MoG=None, wc_MoG=None):

    
    for param in list(model.encoder.main.parameters()) + list(model.encoder.fc.parameters()):
                                    param.requires_grad = True 

    if model.encoder.prior.type != 'imposed':
            for param in model.encoder.prior.parameters():
                    param.requires_grad = True 

    for param in model.decoder.parameters():  
            param.requires_grad = True

    batch, _ = next(train_data_loader)
    if len(batch.size()) == 3:
        batch = batch.unsqueeze(0)

    real_batch = batch.to(device)
    noise_batch, noise_indices = model.sample_noise(batch_size, ret_ind=True)
    fake_batch = model.sample(noise_batch).detach()


    num_to_display = 8
    explanability_batch = torch.cat([real_batch[:num_to_display], fake_batch[:num_to_display]], dim=0)

    # reconstruct real and generated data
    explainability_mu,  explainability_logvar, _, explainability_rec = model(explanability_batch)

    # =========== Update E, D ================
    loss_rec = calc_reconstruction_loss(explanability_batch, explainability_rec, loss_type=recon_loss_type, reduction="mean")
    ## retrieving prior pararms
    if mu_MoG is None:
        mu_MoG, logvar_MoG, wc_MoG = model.get_prior_params()


    # used to hook the gradients
    z_s_explainability = reparameterize(explainability_mu.repeat(MC,1), explainability_logvar.repeat(MC,1)) 
    z_s_explainability.retain_grad()
    explainability_mu.retain_grad()
    explainability_logvar.retain_grad()

    # the z_s_explainability is passed to calc_kl_loss to hook the gradients wrt loss_kl
    loss_kl = calc_kl_loss(model, explainability_mu, explainability_logvar, z_s=z_s_explainability,
                           mc=MC, reduction="mean", kl_loss_type=kl_loss_type,
                           target_mu=mu_MoG, target_logvar=logvar_MoG, target_w_c=wc_MoG)

    loss = beta_rec * loss_rec + beta_kl * loss_kl 
    
    B, _, H, W = explanability_batch.shape

    overlayed_images = np.zeros((3*B, 3, H, W))
    overlayed_images_relu = np.zeros((3*B, 3, H, W))

    outter_index = 0 
    for loss_to_backward in [loss_rec, loss_kl, loss]:

        ## resetting grads
        z_s_explainability.grad = None

        loss_to_backward.backward(retain_graph=True)
        # pull the gradients out of the model
        gradients = model.encoder.get_activations_gradient()

        # average pooling the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.encoder.main(explanability_batch).detach()   

        ## unsqueezing at dimensions [0, 2, 3] to matching activations dimensions
        pooled_gradients = pooled_gradients[(..., ) + (None, ) * 2].unsqueeze(0)

        ## channel-wise activations weighting
        activations = activations * pooled_gradients
        heatmap = torch.mean(activations, dim=1)
        

        for index in range(B):
            
            ## Not applying ReLU (visualizing both positive and negative contribution) (see. https://arxiv.org/abs/2203.06026)

            sensitivity_map = zero_one_scaling(image=heatmap[index].detach().cpu().numpy())
            sensitivity_map = np.clip((sensitivity_map * 255.0).astype(np.uint8), 0.0, 255.0)
            sensitivity_map = np.array(Image.fromarray(sensitivity_map).resize((W, H), 
                                       resample=Image.LANCZOS).convert('L'))  # Scale to original image size.

            overlay_image = show_sensitivity_on_image(image=explanability_batch[index].detach().cpu().numpy().transpose(0,1,2)*255, 
                                                      sensitivity_map=sensitivity_map)
            overlayed_images[outter_index] = overlay_image

            ## Applying ReLU as in Grad-CAM
            heatmap[index] = torch.relu(heatmap[index])
            sensitivity_map = zero_one_scaling(image=heatmap[index].detach().cpu().numpy())
            sensitivity_map = np.clip((sensitivity_map * 255.0).astype(np.uint8), 0.0, 255.0)
            sensitivity_map = np.array(Image.fromarray(sensitivity_map).resize((W, H), 
                                       resample=Image.LANCZOS).convert('L'))  # Scale to original image size.
            
            overlay_image = show_sensitivity_on_image(image=explanability_batch[index].detach().cpu().numpy().transpose(0,1,2)*255, 
                                                      sensitivity_map=sensitivity_map)
            overlayed_images_relu[outter_index] = overlay_image
            
            outter_index += 1

    if explanability_batch.shape[1] != 3:
        explanability_batch = explanability_batch.repeat(1,3,1,1)

    vutils.save_image(torch.cat((explanability_batch, torch.from_numpy(overlayed_images).to(device)),dim=0),
                    '{}/{}_explainability_image.png'.format(save_dir, it), nrow=nrow)
        
    vutils.save_image(torch.cat((explanability_batch, torch.from_numpy(overlayed_images_relu).to(device)),dim=0),
                    '{}/{}_explainability_image_with_relu.png'.format(save_dir, it), nrow=nrow)
    


    z_s_explainability_grad =  z_s_explainability.grad.reshape(MC, explanability_batch.shape[0], -1).sum(axis=(0,1)).abs()
    ## logging grad histogram wrt to KL (Note that z_s_explainability was used to hook the gradients wrt to KL)
    plt.figure()
    plt.bar(range(z_s_explainability_grad.shape[0]), z_s_explainability_grad.data.cpu().numpy())
    plt.title('Latent-Z KL grads')
    plt.savefig('{}/{}_latent_KL_grads.png'.format(save_dir, it))
    plt.close('all')

    plt.imshow(np.diag(z_s_explainability_grad.data.cpu().numpy()), cmap='inferno')
    plt.colorbar()
    plt.title('Jacobian of KL')
    plt.savefig('{}/{}_KL_jacobian.png'.format(save_dir, it))
    plt.close('all')


    mu_unimodal_MoG, logvar_unimodal_MoG = get_mog_logvar(mu_MoG, logvar_MoG, wc_MoG)

    fig, axs = plt.subplots(2, 2)


    axs[0, 0].errorbar(np.arange(explainability_logvar.shape[1]), explainability_logvar.mean(axis=0).data.cpu().numpy(), yerr=explainability_logvar.std(axis=0).data.cpu(), fmt='ok', elinewidth=0.5, capsize=2, markersize=2)
    axs[0, 0].bar(range(explainability_logvar.shape[1]), explainability_logvar.mean(axis=0).data.cpu().numpy(), color='tab:blue')
    axs[0, 0].set_title('Encoder Logvars')

    axs[0, 1].errorbar(np.arange(explainability_logvar.shape[1]), explainability_mu.mean(axis=0).data.cpu().numpy(), yerr=explainability_mu.std(axis=0).data.cpu(),fmt='ok', elinewidth=0.5, capsize=2, markersize=2)
    axs[0, 1].bar(range(explainability_mu.shape[1]), explainability_mu.mean(axis=0).data.cpu().numpy(), color='tab:blue')
    axs[0, 1].set_title('Encoder Mus')

    axs[1, 0].errorbar(np.arange(explainability_logvar.shape[1]), mu_MoG.mean(axis=0).data.cpu().numpy(), yerr=mu_MoG.std(axis=0).data.cpu(), fmt='ok', label='MoG', elinewidth=0.5, capsize=2, markersize=2)
    axs[1, 0].bar(range(explainability_logvar.shape[1]), mu_MoG.mean(axis=0).data.cpu().numpy(), color='tab:blue')
    axs[1, 0].errorbar(np.arange(explainability_logvar.shape[1]),  mu_unimodal_MoG.data.cpu().numpy(), fmt='or', label='unimodal_MoG', elinewidth=0.5, capsize=2, markersize=2)
    axs[1, 0].legend(fontsize=4)
    axs[1, 0].set_title('Prior Mus')

    axs[1, 1].errorbar(np.arange(explainability_logvar.shape[1]), logvar_MoG.mean(axis=0).data.cpu().numpy(), yerr=logvar_MoG.std(axis=0).data.cpu(), fmt='ok',  label='MoG', elinewidth=0.5, capsize=2, markersize=2)
    axs[1, 1].bar(range(explainability_logvar.shape[1]), logvar_MoG.mean(axis=0).data.cpu().numpy(), color='tab:blue')
    axs[1, 1].errorbar(np.arange(explainability_logvar.shape[1]), logvar_unimodal_MoG.data.cpu().numpy(), fmt='or', label='unimodal_MoG', elinewidth=0.5, capsize=2, markersize=2)
    axs[1, 1].legend(fontsize=4)
    axs[1, 1].set_title('Prior Logvars')

    plt.subplots_adjust(wspace=0.3, hspace=0.5)

    plt.savefig('{}/{}_posterior_prior.png'.format(save_dir, it))
    plt.close('all')
    return 
    

def plot_manifold(model, train_data_loader, N_real=5_000, N_fake=500, z1_dim=0, z2_dim=1, n_1d_dims=3, device='cuda', save_path ='./latent.png', nrow=10, with_numpy=True, 
                  deterministic_mode=False, normalize=False):
    

    stacked_real_latents = None
    stacked_fake_latents = None
    stacked_labels = None
    tot = 0

    while tot < N_real:
        fetched_real_samples, fetched_labels = next(train_data_loader)
        _, _, fetched_real_latents, _ = model(fetched_real_samples.to(device), deterministic=deterministic_mode)
        fetched_real_latents = fetched_real_latents.data.cpu()


        if tot == 0:
            stacked_real_latents = fetched_real_latents
            stacked_labels = fetched_labels
        else:
            stacked_real_latents = torch.cat((stacked_real_latents, fetched_real_latents), dim=0)
            stacked_labels = torch.cat((stacked_labels, fetched_labels), dim=0)
        
        tot += fetched_real_latents.shape[0]

    stacked_real_latents = stacked_real_latents[:N_real]
    stacked_labels = stacked_labels[:N_real]

    tot = 0

    ## fake samples
    while tot < N_fake:

        # generate fake data 
        noise_batch, noise_indices = model.sample_noise(len(fetched_labels), ret_ind=True)
        fetched_fake_samples = model.sample(noise_batch)

        _, _, fetched_fake_latents, _ = model(fetched_fake_samples, deterministic=deterministic_mode)
        fetched_fake_latents = fetched_fake_latents.data.cpu()
        fetched_fake_noise = noise_batch.data.cpu()

        if tot == 0:
            stacked_fake_latents = fetched_fake_latents
            stacked_fake_noise = fetched_fake_noise
        else:
            stacked_fake_latents = torch.cat((stacked_fake_latents, fetched_fake_latents), dim=0)
            stacked_fake_noise = torch.cat((stacked_fake_noise, fetched_fake_noise), dim=0)
        
        tot += fetched_fake_latents.shape[0]

    stacked_fake_latents = stacked_fake_latents[:N_fake]
    stacked_fake_noise = stacked_fake_noise[:N_fake]

    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)

    sample_posterior=torch.cat((stacked_real_latents,stacked_fake_latents,stacked_fake_noise), dim=0)

    if normalize:
        ## standardizing posterior for TSNE
        normalized_sample_posterior = (sample_posterior)/sample_posterior.std(dim=0)
    else:
        normalized_sample_posterior = sample_posterior
    
    tsne_embed = tsne.fit_transform(normalized_sample_posterior) 
    
 
    plt.figure()
    plt.scatter(tsne_embed[:N_real, 0], tsne_embed[:N_real,1], c = stacked_labels, s=0.1)
    plt.savefig(save_path+"_manifold_real.png", bbox_inches='tight')
    plt.close('all')


    plt.figure()
    plt.scatter(tsne_embed[N_real:(N_real + N_fake), 0], tsne_embed[N_real:(N_real + N_fake),1], marker='+', s=10)
    plt.savefig(save_path+"_manifold_fake.png", bbox_inches='tight')
    plt.close('all')

    plt.figure()
    plt.scatter(tsne_embed[(N_real + N_fake):, 0], tsne_embed[(N_real + N_fake):,1], marker='+', s=10)
    plt.savefig(save_path+"_manifold_noise.png", bbox_inches='tight')
    plt.close('all')


    ### plotting latent
    from scipy.stats import multivariate_normal

    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    plt.scatter(sample_posterior[:N_real, z1_dim], sample_posterior[:N_real,z2_dim], c = stacked_labels, s=0.3) #real data posterior
    plt.scatter(sample_posterior[N_real:(N_real + N_fake), z1_dim], sample_posterior[N_real:(N_real + N_fake),z2_dim], marker='*',c='r', s=0.3) #fake data posterior
    plt.scatter(sample_posterior[(N_real + N_fake):, z1_dim], sample_posterior[(N_real + N_fake):,z2_dim], marker="X",c='k', s=0.3) # fsampled noise


    mu, logvar, w_c = model.get_prior_params()

    plot_latent = np.vstack((stacked_real_latents, stacked_fake_latents, stacked_fake_noise, mu.data.cpu().numpy()))
    min_x, max_x = np.min(plot_latent[:,z1_dim]), np.max(plot_latent[:,z1_dim])
    min_y, max_y = np.min(plot_latent[:,z2_dim]), np.max(plot_latent[:,z2_dim])

    X = np.linspace(min_x - 0.1*np.abs(min_x), max_x + 0.1*np.abs(max_x), 1024)
    Y = np.linspace(min_y - 0.1*np.abs(min_y), max_y + 0.1*np.abs(max_y), 1024)

    X_grid, Y_grid = np.meshgrid(X, Y)
    grid = np.stack((X_grid, Y_grid), axis=2)

    Z = np.zeros_like(X_grid)

    for i_c in range(model.encoder.prior.C):

        curr_mu = mu[i_c].data.cpu().numpy().tolist()
        current_var = logvar.exp()[i_c].data.cpu().numpy().tolist()

        curr_w = np.array([1]) if model.encoder.prior.C == 1 else w_c[i_c].data.cpu().numpy()

        ax.plot(curr_mu[z1_dim], curr_mu[z2_dim], "ok", markersize=(curr_w*100).clip(1,15))

        if curr_w < 1/(model.encoder.prior.C**2):
            ## plotting inactive prior component  with "+"
            ax.plot(curr_mu[z1_dim], curr_mu[z2_dim], "+k", markersize=(curr_w*100).clip(1,15))
        
        try:
            if with_numpy:
                ## numpy
                diff = grid - (curr_mu[z1_dim], curr_mu[z2_dim])
                inv_cov = np.reciprocal((current_var[z1_dim],current_var[z2_dim]))
                exponent = np.sum(np.dot(diff, np.diag(inv_cov)) * diff, axis=2)
                Z += curr_w*np.exp(-0.5 * exponent) / (2 * np.pi * np.sqrt(np.prod((current_var[z1_dim],current_var[z2_dim]))))

            else:
                ## scipy
                rv = multivariate_normal(curr_mu[z1_dim], curr_mu[z2_dim], [[current_var[z1_dim], 0], [0, current_var[z2_dim]]])
                # Probability Density
                pos = np.empty(X_grid.shape + (2,))
                pos[:, :, 0] = X_grid
                pos[:, :, 1] = Y_grid
                Z += curr_w*rv.pdf(pos)
                ### 
        except:
            pass

    ax.contourf(X_grid, Y_grid, Z, cmap='Blues', alpha=0.7)
    plt.savefig(save_path+"_latent.png", bbox_inches='tight')
    plt.close()
    
    _, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.pcolormesh(X_grid, Y_grid, Z, cmap='viridis')
    plt.savefig(save_path+"_likelihood_heatmap.png", bbox_inches='tight')
    plt.close()

    ## plotting 1D manifold
    num_plots = n_1d_dims ## number of latent dimensions to plot
    num_cols = np.minimum(5, num_plots)
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5*num_rows))
    fig.subplots_adjust(hspace=0.5)

    for z_dim in range(num_plots):
        
        x_stack = np.array([])
        for i_c in range(model.encoder.prior.C):

            x_prior = np.linspace(mu[i_c, z_dim].data.cpu().numpy() - 5*math.sqrt(logvar.exp()[i_c, z_dim].data.cpu().numpy()), 
                                  mu[i_c, z_dim].data.cpu().numpy()  + 5*math.sqrt(logvar.exp()[i_c, z_dim].data.cpu().numpy() ), 100)
            x_stack = np.hstack((x_stack, x_prior))


        row = z_dim // num_cols
        col = z_dim % num_cols
        ax = axes[row, col] if num_rows > 1 else axes[col]

        x_stack = np.sort(x_stack)
        for i_c in range(model.encoder.prior.C):
            
            ax.plot(x_stack, stats.norm.pdf(x_stack, mu[i_c, z_dim].data.cpu().numpy(), math.sqrt(logvar.exp()[i_c, z_dim].data.cpu().numpy())), linestyle='--')
        
        ax.plot(stacked_real_latents.data.cpu().numpy()[:N_real, z_dim], np.zeros_like(stacked_real_latents.data.cpu().numpy()), 'x',  markersize=3)
        ax.plot(stacked_fake_latents.data.cpu().numpy()[:N_real, z_dim], np.zeros_like(stacked_fake_latents.data.cpu().numpy()), 'o',  markersize=3)
        ax.title.set_text(f'Latent dimension {z_dim}')

    plt.savefig(save_path+"_1D_latents.png", bbox_inches='tight')
    plt.close()



    return

def extract_qualitative(model, train_set, save_dir, beta_rec=1.0, beta_kl=1.0, MC=10, recon_loss_type='mse', kl_loss_type='stochastic',
                                          batch_size=16, it=0, nrow=16, device='cpu', delete_figures=True, mu_MoG_target=None, logvar_MoG_target=None, wc_MoG_target=None, 
                                          figures_to_include=[], normalize=False):
    
    model.train()

    train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                num_workers=0)
    train_data_loader = iter(train_data_loader)

    print("visualizing explainability...")

    plot_explainability(model, train_data_loader, save_dir, beta_rec=beta_rec, beta_kl=beta_kl,
                        MC=MC, recon_loss_type=recon_loss_type, kl_loss_type=kl_loss_type,
                        batch_size=batch_size, it=it, nrow=nrow, device=device,  mu_MoG=mu_MoG_target, logvar_MoG=logvar_MoG_target, wc_MoG=wc_MoG_target)
    
    
   
    with torch.no_grad():

        print("visualizing manifold...")
        ## manifold (UMap + latent)
        plot_manifold(model, train_data_loader, device=device, 
                      save_path='{}/{}'.format(save_dir, it), nrow=nrow,
                      normalize=normalize)

        batch, _ = next(train_data_loader)

        real_batch = batch.to(device)

        # reconstruct real data
        _, _, _, rec = model(real_batch)
        
        noise_batch, noise_indices = model.sample_noise(batch_size, ret_ind=True)
        fake_batch = model.sample(noise_batch)

        max_plot = min(batch_size, 64)
        ## reconstructed data
        vutils.save_image(torch.cat([real_batch[:max_plot], rec[:max_plot]], dim=0).data.cpu(),
                        '{}/{}_rec_image.png'.format(save_dir, it), nrow=nrow)
        
        ## generated data
        vutils.save_image(fake_batch[:max_plot].data.cpu(),
                        '{}/{}_gen_image.png'.format(save_dir, it), nrow=nrow)

        ## plotting the pseudoinputs
        if model.encoder.prior.type == "vamp":

            with torch.no_grad():
                pseudoinputs = model.extract_pseudoinputs()
                _, _, _, rec_pseudoinputs = model(pseudoinputs)
                vutils.save_image(torch.cat([pseudoinputs[:max_plot], rec_pseudoinputs[:max_plot]], dim=0).data.cpu(),
                                '{}/{}_MoGs.png'.format(save_dir, it), nrow=nrow)
        elif "MoG":
            with torch.no_grad():
                mu_MoG, _, _ = model.encoder.prior.get_params()
                rec_mu = model.decode(mu_MoG)
                vutils.save_image(rec_mu[:max_plot].data.cpu(),
                                '{}/{}_MoGs.png'.format(save_dir, it), nrow=nrow)
                

    get_summary(save_dir, it, figures_to_include=figures_to_include, delete_figures=delete_figures)

    return

def get_summary(figures_dir, it, figures_to_include, delete_figures=True, margin_offset=5):


    img_files = [os.path.join(figures_dir,"{}_{}.png".format(it,fig_name)) for fig_name in figures_to_include]


    # taken from https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python 
    images = [Image.open(fig_name) for fig_name in img_files]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width+ margin_offset*len(images), max_height), (255, 255, 255))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += margin_offset + im.size[0]

    new_im.save(os.path.join(figures_dir,"summary_{}.png".format(it)))

    if delete_figures:
        ## delete figures
        img_files = [os.path.join(figures_dir, fig_name) for fig_name in os.listdir(figures_dir) if fig_name.startswith(str(it)) and fig_name.endswith('.png') and fig_name != "summary_{}.png".format(it)]
        for img in img_files:
            if os.path.exists(img):
                    os.remove(img)
    return