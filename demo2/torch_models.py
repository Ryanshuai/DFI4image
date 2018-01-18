import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

model_urls = {
    'vgg19g': 'https://www.dropbox.com/s/cecy6wtjy97wt3d/vgg19g-4aff041b.pth?dl=1',
}

class Vgg19g(nn.Module):
  def __init__(self, pretrained=True):
    super(Vgg19g, self).__init__()
    self.features_1 = nn.Sequential(OrderedDict([
      ('conv1_1', nn.Conv2d(3, 64, kernel_size = 3, padding = 1)),
      ('relu1_1', nn.ReLU(inplace = True)),
      ('conv1_2', nn.Conv2d(64, 64, kernel_size = 3, padding = 1)),
      ('relu1_2', nn.ReLU(inplace = True)),
      ('pool1', nn.MaxPool2d(2, 2)),
      ('conv2_1', nn.Conv2d(64, 128, kernel_size = 3, padding = 1)),
      ('relu2_1', nn.ReLU(inplace = True)),
      ('conv2_2', nn.Conv2d(128, 128, kernel_size = 3, padding = 1)),
      ('relu2_2', nn.ReLU(inplace = True)),
      ('pool2', nn.MaxPool2d(2, 2)),
      ('conv3_1', nn.Conv2d(128, 256, kernel_size = 3, padding = 1)),
      ('relu3_1', nn.ReLU(inplace = True)),
    ]))
    self.features_2 = nn.Sequential(OrderedDict([
      ('conv3_2', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_2', nn.ReLU(inplace = True)),
      ('conv3_3', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_3', nn.ReLU(inplace = True)),
      ('conv3_4', nn.Conv2d(256, 256, kernel_size = 3, padding = 1)),
      ('relu3_4', nn.ReLU(inplace = True)),
      ('pool3', nn.MaxPool2d(2, 2)),
      ('conv4_1', nn.Conv2d(256, 512, kernel_size = 3, padding = 1)),
      ('relu4_1', nn.ReLU(inplace = True)),
    ]))
    self.features_3 = nn.Sequential(OrderedDict([
      ('conv4_2', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_2', nn.ReLU(inplace = True)),
      ('conv4_3', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_3', nn.ReLU(inplace = True)),
      ('conv4_4', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu4_4', nn.ReLU(inplace = True)),
      ('pool4', nn.MaxPool2d(2, 2)),
      ('conv5_1', nn.Conv2d(512, 512, kernel_size = 3, padding = 1)),
      ('relu5_1', nn.ReLU(inplace = True)),
    ]))

    if pretrained:
      state_dict = torch.utils.model_zoo.load_url(model_urls['vgg19g'])
      self.load_state_dict(state_dict)

  def forward(self, x):
    features_1 = self.features_1(x)
    features_2 = self.features_2(features_1)
    features_3 = self.features_3(features_2)
    return features_1, features_2, features_3

class ReconModel(nn.Module):
  def __init__(self, drop_rate=0):
    super(ReconModel, self).__init__()

    self.recon5 = _PoolingBlock(3, 512, 512, drop_rate = drop_rate)
    self.upool4 = _TransitionUp(512, 512)
    self.recon4 = _PoolingBlock(3, 1024, 512, drop_rate = drop_rate)
    self.upool3 = _TransitionUp(512, 256)
    self.recon3 = _PoolingBlock(3, 512, 256, drop_rate = drop_rate)
    self.upool2 = _TransitionUp(256, 128)
    self.recon2 = _PoolingBlock(2, 128, 128, drop_rate = drop_rate)
    self.upool1 = _TransitionUp(128, 64)
    self.recon1 = _PoolingBlock(1, 64, 64, drop_rate = drop_rate)
    self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

  def forward(self, x, mc_samples=1):
    # Non MC inference
    if mc_samples == 1 or not any([isinstance(module, nn.Dropout) for module in self.modules()]):
      res = self._forward(x)
      return res, res*0

    # MC inference
    for module in self.modules():
      if isinstance(module, nn.Dropout):
        module.train()

    means = None
    covars = None
    size = None

    for i in range(mc_samples):
      output_var = self._forward(x)
      output = output_var.data
      if size is None:
        size = output.size()

      output = output.permute(0, 2, 3, 1).contiguous().view(-1, 1, 3)
      if means is None:
        means = output.clone()
      else:
        means.add_(output)

      if covars is None:
        covars = torch.bmm(output.permute(0, 2, 1), output)
      else:
        covars.baddbmm_(output.permute(0, 2, 1), output)

    means.div_(mc_samples)
    covars.div_(mc_samples).sub_(torch.bmm(means.permute(0, 2, 1), means))

    # Set stdv to be frobenius norm
    stdvs = covars.view(-1, 9).norm(p=2, dim=1)
    stdvs.sqrt_()
    stdvs.clamp_(0, 1)

    # Reshape
    means = means.view(-1, size[2], size[3], 3).permute(0, 3, 1, 2)
    stdvs = stdvs.view(-1, 1, size[2], size[3])

    means_var = Variable(means, volatile=True)
    stdvs_var = Variable(stdvs.repeat(1, 3, 1, 1), volatile=True)
    return means_var, stdvs_var

  def _forward(self, x):
    features_1, features_2, features_3 = x

    recon5 = self.recon5(features_3)
    upool4 = self.upool4(recon5)

    recon4 = self.recon4(torch.cat([upool4, features_2], 1))
    upool3 = self.upool3(recon4)

    recon3 = self.recon3(torch.cat([upool3, features_1], 1))
    upool2 = self.upool2(recon3)

    recon2 = self.recon2(upool2)
    upool1 = self.upool1(recon2)

    recon1 = self.recon1(upool1)
    recon0 = self.recon0(recon1)

    return recon0
