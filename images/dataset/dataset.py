from tkinter import E
import torchvision.transforms as transforms
import os

from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
import torch
import numpy as np

def load_dataset(dataset, data_root, split='train', border_ratio=0, expand_ratio=0):




    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        ch = 3
        data_loader = CIFAR10(root=os.path.join(data_root,"cifar10_ds"), train=(split=='train'), download=True, transform=transforms.ToTensor())
    
    elif dataset == 'cifar10_gray':
        image_size = 32
        channels = [64, 128]
        data_loader = CIFAR10(root=os.path.join(data_root,"cifar10_ds"), train=(split=='train'), 
                              download=True, transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))
        ch = 1

    elif dataset == 'svhn':

        image_size = 32
        channels = [64, 128, 256]
        data_loader = SVHN(root=os.path.join(data_root,"svhn_ds"), split=split, download=True, transform=transforms.ToTensor())
        ch = 3

    elif dataset == 'svhn_gray':

        image_size = 32
        channels = [64, 128]
        ch = 1
        data_loader =  SVHN(root=os.path.join(data_root,"svhn_ds"), split=split, download=True, 
                            transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),transforms.ToTensor()]))

    elif dataset == 'fmnist':
        image_size = 28
        channels = [64, 128]
        ch = 1
        data_loader = FashionMNIST(root=os.path.join(data_root,"fmnist_ds"), train=(split=='train'), download=True, transform=transforms.ToTensor())

    elif dataset == 'mnist':
        image_size = 28
        channels = [64, 128]
        data_loader = MNIST(root=os.path.join(data_root,"mnist_ds"), train=(split=='train'), download=True, transform=transforms.ToTensor())
        ch = 1

    else:
        raise NotImplementedError("dataset is not supported")

    border = 0 # initialize border

    if border_ratio:

        border = np.ceil(image_size * border_ratio).astype(int)

        image_size = image_size + 2*border
        if 'cifar10' in dataset:
            data_loader.data = torch.nn.functional.pad(torch.from_numpy(data_loader.data.transpose(0,3,1,2)), (border, border, border, border), value=0)
            data_loader.data =  data_loader.data.numpy().transpose(0,2,3,1)

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))]

        elif 'svhn' in dataset:
            data_loader.data = torch.nn.functional.pad(torch.from_numpy(data_loader.data), (border, border, border, border), value=0)
            data_loader.data =  data_loader.data.numpy()

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))].transpose(1,2,0)


        elif 'fmnist' or 'mnist' in dataset:
            data_loader.data = torch.nn.functional.pad(data_loader.data, (border, border, border, border), value=0)

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))]

    elif expand_ratio:

        border = np.ceil(image_size * expand_ratio).astype(int)

        if 'cifar10' in dataset:
            data_loader.data = torch.nn.functional.pad(torch.from_numpy(data_loader.data.transpose(0,3,1,2)).float(), (image_size, image_size, image_size, image_size), mode='circular')
            data_loader.data =  (data_loader.data.numpy().transpose(0,2,3,1)).astype(np.uint8)
            data_loader.data =  data_loader.data[:, image_size-border:-image_size+border, image_size-border:-image_size+border]

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))]

        elif 'svhn' in dataset:
            data_loader.data = torch.nn.functional.pad(torch.from_numpy(data_loader.data).float(), (image_size, image_size, image_size, image_size), mode='circular')
            data_loader.data =  (data_loader.data.numpy()).astype(np.uint8)
            data_loader.data =  data_loader.data[:, :, image_size-border:-image_size+border, image_size-border:-image_size+border]

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))].transpose(1,2,0)


        elif 'fmnist' or 'mnist' in dataset:
            data_loader.data = torch.nn.functional.pad(data_loader.data.float().unsqueeze(1), (image_size, image_size, image_size, image_size), mode='circular').squeeze(1)
            data_loader.data = data_loader.data.to(torch.uint8)

            data_loader.data = data_loader.data[:, image_size-border:-image_size+border, image_size-border:-image_size+border]

            sample_img = data_loader.data[np.random.randint(0, len(data_loader.data))]


        image_size = image_size + 2*border


    return data_loader, image_size, ch, channels, border


def get_samples_from_dataloader(dataloader, n_samples=10_000):
    
    dataloader = iter(dataloader)

    tot = 0

    while tot < n_samples:
        fetched_real_samples, fetched_labels = next(dataloader)


        if tot == 0:
            stacked_samples = fetched_real_samples
            stacked_labels = fetched_labels
        else:
            stacked_samples = torch.cat((stacked_samples, fetched_real_samples), dim=0)
            stacked_labels = torch.cat((stacked_labels, fetched_labels), dim=0)
        
        tot += fetched_real_samples.shape[0]

    stacked_samples = stacked_samples[:n_samples]
    stacked_labels = stacked_labels[:n_samples]

    return stacked_samples, stacked_labels