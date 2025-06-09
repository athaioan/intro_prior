"""Calculates the Frechet Inception Distance (FID) to evalulate GANs
The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.
When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).
The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.
See --help to see further details.
Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
https://github.com/mseitzer/pytorch-fid
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
import torchvision.transforms as transforms

from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.utils as vutils
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PIL import Image
import logging
# from dataset import ImageDatasetFromFile
import math
import torch

# from networks import IntroVAE
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from .inception import InceptionV3
from .resnet50 import resnet50

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')



""" taken from https://github.com/marcojira/fld """

class Metric:
    """Generic Metric class"""

    def __init__(self):
        # To be implemented by each metric
        self.name = None
        pass

    def compute_metric(
        self,
        train_feat,
        test_feat,
        gen_feat,
    ):
        """Computes the metric value for the given sets of features (TO BE IMPLEMENTED BY EACH METRIC)
        - train_feat: Features from set of samples used to train generative model
        - test_feat: Features from test samples
        - gen_feat: Features from generated samples

        returns: Metric value
        """
        pass



# Batch implementation for memory issues (equivalent)
BATCH_SIZE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PrecisionRecall(Metric):
    # We use 4 to ignore distance to self
    def __init__(self, mode, num_neighbors=4):
        super().__init__()

        self.name = mode  # One of ("Precision", "Recall")
        self.num_neighbors = num_neighbors

    def get_nn_dists(self, feat):
        dists = torch.zeros(feat.shape[0]).to(DEVICE)
        for i in range(math.ceil(feat.shape[0] / BATCH_SIZE)):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            curr_dists = torch.cdist(feat[start:end], feat)
            curr_dists = curr_dists.topk(
                self.num_neighbors, dim=1, largest=False
            ).values
            dists[start:end] = curr_dists[:, -1]
        return dists

    def pct_in_manifold(self, evaluated_feat, manifold_feat):
        total_in_manifold = 0
        nn_dists = self.get_nn_dists(manifold_feat)

        for i in range(math.ceil(evaluated_feat.shape[0] / BATCH_SIZE)):
            start, end = i * BATCH_SIZE, (i + 1) * BATCH_SIZE
            pairwise_dists = torch.cdist(evaluated_feat[start:end], manifold_feat)
            comparison_tensor = nn_dists.unsqueeze(0).repeat(pairwise_dists.shape[0], 1)

            num_in_manifold = (pairwise_dists < comparison_tensor).sum(dim=1)
            num_in_manifold = (num_in_manifold > 0).sum()
            total_in_manifold += num_in_manifold

        return total_in_manifold / evaluated_feat.shape[0]

    def compute_metric(
        self,
        train_feat,
        test_feat,  # Test samples not used by Precision/Recall
        gen_feat,
    ):
        train_feat = train_feat.to(DEVICE)
        gen_feat = gen_feat.to(DEVICE)

        if self.name == "Precision":
            return self.pct_in_manifold(gen_feat, train_feat).item()
        elif self.name == "Recall":
            return self.pct_in_manifold(train_feat, gen_feat).item()
        else:
            raise NotImplementedError


def get_activations_given_dataset(dataloader, model, batch_size=50, dims=2048,
                                  cuda=False, device=torch.device("cpu"), num_images=50000, eval_mode=False, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    if eval_mode:
        model.eval()
    else:
        model.train()
    
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    activations = []
    num_images_processed = 0
    for idx, batch in enumerate(dataloader):
        if len(batch) == 2 or len(batch) == 3:
            batch = batch[0]

        # gray scale data
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)
            
        if cuda:
            batch = batch.to(device)

        ## normalization
        if model.normalization_mode == 'classic_inception':
                        
            batch = F.interpolate(batch,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)

            batch = 2 * batch - 1  # Scale from range (0, 1) to range (-1, 1)

        elif model.normalization_mode == 'classic_imagenet':

            batch = transforms.Compose([transforms.Resize(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.228, 0.224, 0.225])])(batch)


        res = model(batch)
        # if idx == 0:
        #     print(batch[0].min())
        #     print(batch[0].max())
        #     print("real images shape: ", batch.shape)
        #     print("res output shape:" , res.shape)
        # res = inception.run(x, num_gpus=gpu_count, assume_frozen=True)
        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if res.ndim>2 and (res.size(2) != 1 or res.size(3) != 1):
            res = adaptive_avg_pool2d(res, output_size=(1, 1))
        activations.append(res.cpu().data.numpy().reshape(res.size(0), -1))
        num_images_processed += batch.shape[0]
        if num_images_processed > num_images:
            # print("num img proc.: ", num_images_processed, " num img req.:, ", num_images)
            break
    activations = np.concatenate(activations)
    activations = activations[:num_images]
    print("total real activations: ", activations.shape)
    # print("num images processed: ", num_images_processed)

    if verbose:
        print(' done')

    return activations


def get_activations_generate(model_s, model, batch_size=50, dims=2048,
                             cuda=False, device=torch.device("cpu"), num_images=50000, eval_mode=False, verbose=False,):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """

    if eval_mode:
        model.eval()
    else:
        model.train()

    activations = []
    num_images_processed = 0
    # for _ in tqdm(range(0, num_images, batch_size)):
    for index in range(0, num_images, batch_size):
        noise_batch = model_s.sample_noise(batch_size)
        images = model_s.sample(noise_batch)
        images = images.data.cpu().numpy()

        # ## TO DELETE
        # if index==0:
        #     for it in range(batch_size):
        #         import matplotlib.pyplot as plt
        #         plt.imsave("/proj/cvl/users/x_ioaat/projects/intro_prior/image_intro_w36/W45_fid/cifar10/store_fid_imgs/test_{}.png".format(it),(np.clip(images[it].transpose(1,2,0),0,1)*255).astype(np.uint8))
        # ## TO DELETE

        images = np.clip(images * 255, 0, 255).astype(np.uint8)

        images = images / 255.0

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.to(device)

        # gray scale data
        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        ## normalization
        if model.normalization_mode == 'classic_inception':
                        
            batch = F.interpolate(batch,
                            size=(299, 299),
                            mode='bilinear',
                            align_corners=False)

            batch = 2 * batch - 1  # Scale from range (0, 1) to range (-1, 1)

        elif model.normalization_mode == 'classic_imagenet':

            batch = transforms.Compose([transforms.Resize(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.228, 0.224, 0.225])])(batch)

        res = model(batch)

        activations.append(res.cpu().data.numpy().reshape(res.size(0), -1))

    activations = np.concatenate(activations)
    activations = activations[:num_images]
    print("total generated activations: ", activations.shape)

    if verbose:
        print(' done')

    return activations



def calculate_precision_recall_given_dataset(dataloader, model_s, batch_size, cuda, dims, device, num_images, fid_backbone="inceptionV3", eval_mode=False):
    """Calculates the Precision and Recall"""

    if fid_backbone == 'inceptionV3':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
        model.normalization_mode = 'classic_inception'

    elif fid_backbone == 'resnet50_SwAV':
        model = resnet50(pretrained_mode='SwAV')
        model.normalization_mode = 'classic_imagenet'


    if cuda:
        model.to(device)
        
    train_feat = get_activations_given_dataset(dataloader, model, batch_size, dims, cuda, device, num_images, eval_mode=eval_mode, verbose=False)
       
    gen_feat =  get_activations_generate(model_s, model, batch_size, dims, cuda, device, num_images, eval_mode=eval_mode, verbose=False)
    
    
    precision = PrecisionRecall(mode="Precision").compute_metric(torch.tensor(train_feat), None, torch.tensor(gen_feat)) # Default precision
    recall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(torch.tensor(train_feat), None, torch.tensor(gen_feat)) # Recall with k=5



    return precision, recall

def calculate_precision_recall_given_activations(train_feat, gen_feat):

    precision = PrecisionRecall(mode="Precision").compute_metric(torch.tensor(train_feat), None, torch.tensor(gen_feat)) # Default precision
    recall = PrecisionRecall(mode="Recall", num_neighbors=5).compute_metric(torch.tensor(train_feat), None, torch.tensor(gen_feat)) # Recall with k=5

    return precision, recall



