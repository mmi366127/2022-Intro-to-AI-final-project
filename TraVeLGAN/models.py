
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


def _L2normalize(v, eps = 1e-12):
    return v / (torch.norm(v) + eps)


def max_sinular_value(W, u, power_iterations = 1):
    _u = u
    for _ in range(power_iterations):
        _v = _L2normalize(torch.matmul(_u, W.data))
        _u = _L2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)))

    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u


class SNLinear(nn.Linear):
    def __init__(self, in_features, out_features, **kwargs):
        super(SNLinear, self).__init__(in_features, out_features, **kwargs)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_sinular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, x):
        return F.linear(x, self.W_, self.bias)


class ConvSN2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvSN2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_sinular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma
    
    def forward(self, x):
        return F.conv2d(x, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ConvTransposeSN2d(nn.ConvTranspose2d):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(ConvTransposeSN2d, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.register_buffer('u', torch.Tensor(1, in_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_sinular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, x):
        return F.conv_transpose2d(x, self.W_, self.bias, self.stride, self.padding, 0, self.groups, self.dilation)


class Siamese(nn.Module):
    def __init__(self, input_size, output_size = 128):
        
        super(Siamese, self).__init__()
        
        self.C, self.H, self.W = input_size
        
        # width after conv layers
        self.W1 = self.W - 2
        self.W2 = (self.W1 + 1) // 2
        self.W3 = (self.W2 + 1) // 2

        # downscaling conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.C, out_channels = 256, kernel_size = (self.H, 3), stride = (1, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 9), stride = (1, 2), padding = (0, 4)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 7), stride = (1, 2), padding = (0, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # Dense layer
        self.flatten = nn.Flatten()
        self.Dense = nn.Linear(256 * self.W3, output_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.Dense(x)
        return x


class Generater(nn.Module):
    def __init__(self, input_size):
        
        super(Generater, self).__init__()

        self.C, self.H, self.W = input_size

        # downscaling conv layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels = self.C, out_channels = 256, kernel_size = (self.H, 3), stride = (1, 1), padding = (0, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 9), stride = (1, 2), padding = (0, 4)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1, 7), stride = (1, 2), padding = (0, 3)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        # upscaling conv layers
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor = (1, 2), mode = 'bilinear', align_corners = False),
            ConvSN2d(in_channels = 256, out_channels = 256, kernel_size = (1, 7), stride = (1, 1), padding = 'same', bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor = (1, 2), mode = 'bilinear', align_corners = False),
            ConvSN2d(in_channels = 512, out_channels = 256, kernel_size = (1, 9), stride = (1, 1), padding = 'same', bias = False),
            nn.LeakyReLU(0.2)
        )

        self.deconv3 = nn.Sequential(
            ConvTransposeSN2d(in_channels = 512, out_channels = 1, kernel_size = (self.H, 1)),
            nn.Tanh()
        )


    def forward(self, x):
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        x4 = self.deconv1(x3)
        x4 = torch.cat((x4, x2), dim = 1)
        
        x5 = self.deconv2(x4)
        x5 = torch.cat((x5, x1), dim = 1)
        
        x6 = self.deconv3(x5)
        
        return x6


class Discriminator(nn.Module):
    def __init__(self, input_size):
    
        super(Discriminator, self).__init__()
        
        self.C, self.H, self.W = input_size

        # width after conv layers
        self.W1 = self.W - 2
        self.W2 = (self.W1 + 1) // 2 
        self.W3 = (self.W2 + 1) // 2

        self.conv1 = nn.Sequential(
            ConvSN2d(in_channels = self.C, out_channels = 512, kernel_size = (self.H, 3), stride = (1, 1)),
            nn.LeakyReLU(0.2)
        )

        self.conv2 = nn.Sequential(
            ConvSN2d(in_channels = 512, out_channels = 512, kernel_size = (1, 9), stride = (1, 2), padding = (0, 4)),
            nn.LeakyReLU(0.2)
        )

        self.conv3 = nn.Sequential(
            ConvSN2d(in_channels = 512, out_channels = 512, kernel_size = (1, 7), stride = (1, 2), padding = (0, 3)),
            nn.LeakyReLU(0.2)
        )

        self.flatten = nn.Flatten()
        self.Dense = SNLinear(512 * self.W3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.Dense(x)
        return x



if __name__ == '__main__':

    x = torch.randn(20, 1, 192, 24)

    model_S = Siamese(input_size = (1, 192, 24))
    model_G = Generater(input_size = (1, 192, 24))
    model_D = Discriminator(input_size = (1, 192, 24))

    s = model_S(x)
    g = model_G(x)
    d = model_D(x)

    print(s.size(), g.size(), d.size())

    