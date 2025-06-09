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

def get_activations_given_dataset(dataloader, model, batch_size=50, dims=2048,
                                  cuda=False, verbose=False, device=torch.device("cpu"), num_images=50000, eval_mode=False):
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
                             cuda=False, verbose=False, device=torch.device("cpu"), num_images=50000, eval_mode=False):
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


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, errestfloat = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics_given_dataset(dataloader, model, batch_size=50,
                                                  dims=2048, cuda=False, verbose=False, device=torch.device("cpu"),
                                                  num_images=50000, eval_mode=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_given_dataset(dataloader, model, batch_size, dims, cuda, verbose, device, num_images, eval_mode=eval_mode)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def calculate_activation_statistics_generate(model_s, model, batch_size=50,
                                             dims=2048, cuda=False, verbose=False, device=torch.device("cpu"),
                                             num_images=50000, eval_mode=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations_generate(model_s, model, batch_size, dims, cuda, verbose, device, num_images, eval_mode=eval_mode)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma, act


def _compute_statistics_of_given_dataset(dataloader, model, batch_size, dims, cuda, device, num_images, eval_mode=False):
    m, s, act = calculate_activation_statistics_given_dataset(dataloader, model, batch_size,
                                                         dims, cuda, device=device, num_images=num_images, eval_mode=eval_mode)

    return m, s, act


def _compute_statistics_of_generate(model_s, model, batch_size, dims, cuda, device, num_images, eval_mode=False ):
    m, s, act = calculate_activation_statistics_generate(model_s, model, batch_size,
                                                    dims, cuda, device=device, 
                                                    num_images=num_images, eval_mode=eval_mode)

    return m, s, act


def calculate_fid_and_act_given_dataset(dataloader, model_s, batch_size, cuda, dims, device, num_images, fid_backbone="inceptionV3", eval_mode=False):
    """Calculates the FID"""
    
    if fid_backbone == 'inceptionV3':
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx])
        model.normalization_mode = 'classic_inception'

    elif fid_backbone == 'resnet50_SwAV':
        model = resnet50(pretrained_mode='SwAV')
        model.normalization_mode = 'classic_imagenet'


    if cuda:
        model.to(device)
        

    m1, s1, train_feat = _compute_statistics_of_given_dataset(dataloader, model, batch_size,
                                                              dims, cuda, device, num_images, eval_mode)

    m2, s2, gen_feat = _compute_statistics_of_generate(model_s, model, batch_size,
                                                       dims, cuda, device, num_images, eval_mode)

    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, train_feat, gen_feat


