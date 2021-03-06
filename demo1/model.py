
import math
import numpy

import torch
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
from collections import OrderedDict
from data_loader import  get_DataLoader, Transform


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

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3


class vgg19g_DeepFeature(object):
    def __init__(self):
        self.forward_model = Vgg19g(pretrained = True)
        self.forward_model.eval()

        self.transform = Transform()
        # Parameters
        self.tv_lambda = 10
        self.max_iter = 500

    def get_Deep_Feature(self, image_list):  #得到一个图片的list的深度特征
        # Storage for features
        flattened_features = None

        x = torch.from_numpy(numpy.array(list(image_list))).permute(0, 3, 1, 2)  # 这行想去掉TODO
        loader = get_DataLoader(x,self.transform.forward_transform)

        with torch.cuda.device(3):
            self.forward_model.cuda()

            for i, input in enumerate(loader):
                #print('Image %d of %d' % (i+1, x.size(0)))
                input_var = Variable(input, volatile=True).cuda()
                feature_vars = self.forward_model(input_var)

                # Add to tally of features
                if flattened_features is None:
                    flattened_features = torch.cat([f.data.sum(0).view(-1) for f in feature_vars], 0)
                else:
                    flattened_features.add_(torch.cat([f.data.sum(0).view(-1) for f in feature_vars], 0))
                del input_var
                del feature_vars

            flattened_features.div_(x.size(0))

            flattened_features = flattened_features.cpu()
            self.forward_model.cpu()

        return flattened_features.numpy()

    def Deep_Feature_inverse(self, Feature, initial_image, **options):  #得到一个深度特征对应的图片
        verbose = options.get('verbose', 0)
        x = torch.from_numpy(numpy.array(initial_image))
        x = x.permute(2, 0, 1)  # 交换轴
        orig_size = x.size()

        x = self.transform.forward_transform(x)  #
        x = x.contiguous().view(1, *x.size())  # 使连续，并改变尺寸

        with torch.cuda.device(3):
            self.forward_model.cuda()
            recon_var = nn.Parameter(x.cuda(), requires_grad=True)  # 使得recon_var为一个参数

            # Get size of features
            orig_feature_vars = self.forward_model(recon_var)
            sizes = ([f.data[:1].size() for f in orig_feature_vars])
            cat_offsets = torch.cat([torch.Tensor([0]),
                                   torch.cumsum(torch.Tensor([f.data[:1].nelement() for f in orig_feature_vars]),
                                                0)])

          # Reshape provided features to match original features
            cat_features = torch.from_numpy(Feature).view(-1)
            features = tuple(Variable(cat_features[int(start_i):int(end_i)].view(size)).cuda()
                           for size, start_i, end_i in zip(sizes, cat_offsets[:-1], cat_offsets[1:]))

            # Create optimizer and loss functions
            optimizer = torch.optim.LBFGS(
            params=[recon_var],
            max_iter=options['max_iter'] if 'max_iter' in options else self.max_iter,
            )
            optimizer.n_steps = 0
            criterion3 = nn.MSELoss(size_average=False).cuda()
            criterion4 = nn.MSELoss(size_average=False).cuda()
            criterion5 = nn.MSELoss(size_average=False).cuda()
            criterion_tv = TVLoss().cuda()

            # Optimize
            def step():
                self.forward_model.zero_grad()
                if recon_var.grad is not None:
                    recon_var.grad.data.fill_(0)

                output_var = self.forward_model(recon_var)
                loss3 = criterion3(output_var[0], features[0])
                loss4 = criterion4(output_var[1], features[1])
                loss5 = criterion5(output_var[2], features[2])
                loss_tv = self.tv_lambda * criterion_tv(recon_var)
                loss = loss3 + loss4 + loss5 + loss_tv
                loss.backward()

                if verbose and optimizer.n_steps % 25 == 0:
                    print('Step: %d  total: %.1f  conv3: %.1f  conv4: %.1f  conv5: %.1f  tv: %.3f' %
                        (optimizer.n_steps, loss.data[0], loss3.data[0], loss4.data[0], loss5.data[0],
                         loss_tv.data[0]))

                optimizer.n_steps += 1
                return loss

            optimizer.step(step)
            self.forward_model.cpu()
            recon = recon_var.data[0].cpu()

        # Return the new image
        def unfit_from_quantum(img, orig_size, quantum=64):
        # get the image with size of orig_size from img
            if orig_size[1] % int(quantum) == 0:
                pad_w = 0
            else:
                pad_w = int((quantum - orig_size[1] % int(quantum)) / 2)

            if orig_size[2] % int(quantum) == 0:
                pad_h = 0
            else:
                pad_h = int((quantum - orig_size[2] % int(quantum)) / 2)

            res = img[:, pad_w:(pad_w + orig_size[1]), pad_h:(pad_h + orig_size[2])].clone()
            return res

        recon = self.transform.reverse_transform(recon)
        recon = unfit_from_quantum(recon, orig_size)
        recon = recon.squeeze()
        recon = recon.permute(1, 2, 0)
        return recon.numpy()

