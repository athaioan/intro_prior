import os 

import numpy as np
import random 

import torch


def seed_everything(seed: int):

    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("\n --------- {} seed specified ---------\n".format(seed))

    return

def plot_losses(iteration, max_iter, *losses):
    loss_names = ["REC_real", "KL_real", "KL_fake", "KL_rec", "expELBO_fale", "expELBO_rec"]
    info = "\nIter: {}/{}".format(iteration, max_iter)
    for i, loss in enumerate(losses):
        info += ' {}: {:.4f}, '.format(loss_names[i], loss.data.cpu())
    print(info)

    return


def save_checkpoint(model, result_dir, epoch, iteration):
    model_out_path = os.path.join(result_dir, "model_epoch_{}_iter_{}.pth".format(epoch, iteration))
    state = {"epoch": epoch, "model": model.state_dict()}
    torch.save(state, model_out_path)
    print("model checkpoint saved @ {}".format(model_out_path))
    return


