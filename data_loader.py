
import math
import numpy

import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
from collections import OrderedDict


class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, x, transform):
        super(Dataset, self).__init__(x, torch.zeros(x.size(0)))
        self.transform = transform

    def __getitem__(self, index):
        input = self.transform(self.data_tensor[index])
        target = self.target_tensor[index]
        return input, target


class Data_loader():
    # Make dataloader for data
    x = torch.from_numpy(numpy.array(list(X))).permute(0, 3, 1, 2)
    loader = torch.utils.data.DataLoader(
        Dataset(x, transform=self.forward_transform),
        batch_size=1,
        pin_memory=True,
    )


