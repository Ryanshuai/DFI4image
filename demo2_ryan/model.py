import numpy
import torch
import torch.nn as nn
from collections import OrderedDict


class TVLoss(nn.Module):
    def __init__(self, eps=1e-3, beta=2):
        super(TVLoss, self).__init__()
        self.eps = eps
        self.beta = beta

    def forward(self, input):
        x_diff = input[:, :, :-1, :-1] - input[:, :, :-1, 1:]
        y_diff = input[:, :, :-1, :-1] - input[:, :, 1:, :-1]

        sq_diff = torch.clamp(x_diff * x_diff + y_diff * y_diff, self.eps, 10000000)  #限制上下界
        return torch.norm(sq_diff, self.beta / 2.0) ** (self.beta / 2.0)  #求p范数


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

        model_urls = 'https://www.dropbox.com/s/cecy6wtjy97wt3d/vgg19g-4aff041b.pth?dl=1'
        if pretrained:
            state_dict = torch.utils.model_zoo.load_url(model_urls, model_dir='models/')
            self.load_state_dict(state_dict)

    def forward(self, image_batch):
        features_1 = self.features_1(image_batch)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        features_1 = features_1.view(-1)
        features_2 = features_2.view(-1)
        features_3 = features_3.view(-1)
        features = torch.cat((features_1, features_2, features_3))
        return features.numpy()


class vgg19g_DeepFeature():
    def __init__(self):
        self.vgg= Vgg19g(pretrained = True)
        self.vgg.eval()
        self.vgg.cuda()

        # Parameters
        self.tv_lambda = 10
        self.max_iter = 500

    def get_Deep_Feature(self, image_batch):  ##image_batch [BS,W,H,C]
        with torch.cuda.device(0):
            deep_feature_batch = self.vgg.forward(image_batch)
            return deep_feature_batch.numpy()

    def Deep_Feature_inverse(self, target_deepFeature_batch, image_batch):  ##deep_feature_batch [BS,???]
        with torch.cuda.device(0):
            image_batch_tenor = torch.from_numpy(numpy.array(image_batch))
            recon_image_batch = nn.Parameter(image_batch_tenor.cuda(), requires_grad=True)  # 使得recon_var为一个参数

            # Create optimizer and loss functions
            optimizer = torch.optim.LBFGS(params=[recon_image_batch], max_iter=self.max_iter)
            optimizer.n_steps = 0
            mse_loss = nn.MSELoss(size_average=False).cuda()
            criterion_tv = TVLoss().cuda()

            # Optimize
            def closure():
                self.vgg.zero_grad()
                if recon_image_batch.grad is not None:
                    recon_image_batch.grad.data.fill_(0)

                recon_deepFeature_batch = self.vgg(recon_image_batch)

                mse = mse_loss(recon_deepFeature_batch, target_deepFeature_batch)
                tv = self.tv_lambda * criterion_tv(recon_image_batch)
                loss = mse + tv
                loss.backward()

                optimizer.n_steps += 1
                return loss

            optimizer.step(closure)

            return recon_image_batch.numpy()
