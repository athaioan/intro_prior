import numpy as np
import torch
import random


def sample_2d_ood(dataset, n_samples):
    """
    https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py
    :param dataset:
    :param n_samples:
    :return:
    """
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1 / np.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == 'spiral':

        t = np.pi/2

        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
        d1x = - torch.cos(n+t) * n + torch.rand(n_samples // 2) * 0.5
        d1y = torch.sin(n+t) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3

        label = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()

        output_ood = x + 0.1 * z


        return output_ood, label.type(torch.int)

    elif dataset == 'checkerboard':


        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2      
        x2 = x2_ + (x1.floor()+1) % 2

        label1 = np.floor(x1)+2
        label2 = x2<=0
        label = label1*2 + label2
        
        output_ood = torch.stack([x1, x2], dim=1) * 2 

        return output_ood, label.type(torch.int)

    elif dataset == 'rings':

        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * np.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * np.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * np.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * np.pi, n_samples1 + 1)[:-1]
        
        t = 0.15
        circ4_x = torch.cos(linspace4) * (1-t)
        circ4_y = torch.sin(linspace4) * (1-t)
        circ3_x = torch.cos(linspace4) * (0.75-t)
        circ3_y = torch.sin(linspace3) * (0.75-t)
        circ2_x = torch.cos(linspace2) * (0.5-t)
        circ2_y = torch.sin(linspace2) * (0.5-t)
        circ1_x = torch.cos(linspace1) * (0.25-t)
        circ1_y = torch.sin(linspace1) * (0.25-t)

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        label =  torch.cat([torch.ones(n_samples4)*0, torch.ones(n_samples3)*1, 
                            torch.ones(n_samples2)*2, torch.ones(n_samples3)*3])

        # random sample
        shuffled_indices = torch.randint(0, n_samples, size=(n_samples,))
        x = x[shuffled_indices]
        label = label[shuffled_indices]

        # Add noise
        output_ood = x + torch.normal(mean=torch.zeros_like(x), std=0.08 * torch.ones_like(x))

       
        return output_ood, label.type(torch.int)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')


def sample_2d_data(dataset, n_samples):
    """
    https://github.com/kamenbliznashki/normalizing_flows/blob/master/bnaf.py
    :param dataset:
    :param n_samples:
    :return:
    """
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1 / np.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (sq2, sq2), (-sq2, sq2), (sq2, -sq2), (-sq2, -sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x, y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    elif dataset == 'spiral':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * np.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y = torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([d1x, d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3

        output = x + 0.1 * z
        label = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)]).long()

        return output, label.type(torch.int)

    elif dataset == 'checkerboard':

        x1 = torch.rand(n_samples) * 4 - 2

        # x1 = torch.rand(n_samples) * 1 - 2 ## sub-dataset

        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        
        # x2_ = torch.rand(n_samples) - torch.randint(1, 2, (n_samples,), dtype=torch.float) * 1 ## sub-dataset

  
        x2 = x2_ + x1.floor() % 2


        label1 = np.floor(x1)+2
        label2 = x2<=0
        label = label1*2 + label2
        
        output = torch.stack([x1, x2], dim=1) * 2

        return output, label.type(torch.int)

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * np.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * np.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * np.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * np.pi, n_samples1 + 1)[:-1]

        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        label =  torch.cat([torch.ones(n_samples4)*0, torch.ones(n_samples3)*1, 
                            torch.ones(n_samples2)*2, torch.ones(n_samples3)*3])

        # random sample
        shuffled_indices = torch.randint(0, n_samples, size=(n_samples,))
        x = x[shuffled_indices]
        label = label[shuffled_indices]

        # Add noise
        output = x + torch.normal(mean=torch.zeros_like(x), std=0.08 * torch.ones_like(x))
        
        return output, label.type(torch.int)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')

## Dataset construction
### 8-Gaussian
class ToyDataset:
    def __init__(self, dim=2, scale=2, iter_per_mode=100, distr='8Gaussians'):
        self.distr = distr
        self.dim = dim
        self.scale = scale

        self.dataset = []
        for i in range(100000 // 25):
            for x in range(-2, 3):
                for y in range(-2, 3):
                    point = np.random.randn(2) * 0.05
                    point[0] += 2 * x
                    point[1] += 2 * y
                    self.dataset.append(point)
        self.dataset = np.array(self.dataset, dtype='float32')
        np.random.shuffle(self.dataset)
        self.dataset /= 2.828  # stdev
        self.range = 1
        self.curr_iter = 0
        self.curr_mode = 0
        self.iter_per_mode = iter_per_mode
        self.centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1. / np.sqrt(2), 1. / np.sqrt(2)),
            (1. / np.sqrt(2), -1. / np.sqrt(2)),
            (-1. / np.sqrt(2), 1. / np.sqrt(2)),
            (-1. / np.sqrt(2), -1. / np.sqrt(2))
        ]

        self.centers = [(self.scale * x, self.scale * y) for x, y in self.centers]

        self.scale = 1 if distr == '8Gaussian' else 2


    def next_batch(self, batch_size=64, device=None, sig=0.02, return_ood=False):

        dist_1 = ['spiral', 'checkerboard', 'rings']

        if return_ood:
            ## return ood data
            if self.distr in dist_1:

                dataset_ood, labels = sample_2d_ood(self.distr, batch_size)
                return dataset_ood.to(device), labels.to(device)
        
            else:
                theta = np.pi/8
                rot_mat = np.asarray([[np.cos(theta), -np.sin(theta)], 
                                        [np.sin(theta), np.cos(theta)]])

                # constructing the centers
                dataset_ood = []
                labels = []
                # Gaussian sampling on centers
                for _ in range(batch_size):
                    point = np.random.randn(2) * sig
                    center_index = random.choice(range(len(self.centers)))
                    center = self.centers[center_index]
                    center = rot_mat @ np.asarray(self.centers[center_index])
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset_ood.append(point)
                    labels.append(center_index)
                dataset_ood = np.array(dataset_ood, dtype='float32')
                dataset_ood /= 1.414  # stdev

                dataset_ood = torch.FloatTensor(dataset_ood)
                labels = torch.IntTensor(labels)

                return dataset_ood.to(device), labels.to(device)
        
        else:
            ## return in-distribution data

            if self.distr in dist_1:
                dataset_ood, labels = sample_2d_data(self.distr, batch_size)
                return dataset_ood.to(device), labels.to(device)
            else:
                # constructing the centers
                dataset = []
                labels = []
                # Gaussian sampling on centers
                for _ in range(batch_size):
                    point = np.random.randn(2) * sig
                    center_index = random.choice(range(len(self.centers)))
                    center = self.centers[center_index]
                    point[0] += center[0]
                    point[1] += center[1]
                    dataset.append(point)
                    labels.append(center_index)
                dataset = np.array(dataset, dtype='float32')
                dataset /= 1.414  # stdev

                dataset = torch.FloatTensor(dataset)
                labels = torch.IntTensor(labels)

                return dataset.to(device), labels.to(device)


def get_samples_from_dataloader(dataloader, n_samples=10_000, batch_size=64, device='cpu'):
    
    tot = 0

    while tot < n_samples:
        fetched_real_samples, fetched_labels = dataloader.next_batch(batch_size=batch_size, device=device)

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