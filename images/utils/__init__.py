from .losses import reparameterize, calc_reconstruction_loss, calc_kl_loss, log_Normal_diag, kl_soft_assignments, \
      jacobian_norm, compute_grad_norm, get_mog_logvar, softplus
from .configs import get_configs, augment_configs, augment_intro_configs, augment_encoder_ovefit_configs


from .session import save_checkpoint, seed_everything, plot_loss_history, construct_gif, get_training_mode, get_eval_mode, LogHist, store_history_dict
from .results import plot_manifold, extract_qualitative

