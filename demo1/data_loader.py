import numpy
import math
import torch
import torch.utils.data as Data
import torchvision as tv


class FitToQuantum():
    def __init__(self, quantum=64):
        self.quantum = float(quantum)

    def __call__(self, img):
        quantum = self.quantum
        size = img.size()

        if img.size(1) % int(quantum) == 0:
            pad_w = 0
        else:
            pad_w = int((quantum - img.size(1) % int(quantum)) / 2)

        if img.size(2) % int(quantum) == 0:
            pad_h = 0
        else:
            pad_h = int((quantum - img.size(2) % int(quantum)) / 2)

        res = torch.zeros(size[0],
            int(math.ceil(size[1]/quantum) * quantum),
            int(math.ceil(size[2]/quantum) * quantum))
        res[:, pad_w:(pad_w + size[1]), pad_h:(pad_h + size[2])].copy_(img)
        return res


mean = torch.Tensor((0.485, 0.456, 0.406))
stdv = torch.Tensor((0.229, 0.224, 0.225))
forward_transform = tv.transforms.Compose([
    tv.transforms.Normalize(mean=mean, std=stdv),
    FitToQuantum(),
])

class Dataset(torch.utils.data.TensorDataset):
    def __init__(self, x, transform):
        super(Dataset, self).__init__(x, torch.zeros(x.size(0)))
        self.transform = transform

    def __getitem__(self, index):
        input = self.transform(self.data_tensor[index])
        target = self.target_tensor[index]
        return input, target


def get_DataLoader(X):
    x = torch.from_numpy(numpy.array(list(X))).permute(0, 3, 1, 2)
    loader = torch.utils.data.DataLoader(
        Dataset(x, transform=forward_transform),
        batch_size=1,
        pin_memory=True,
    )
    return loader


