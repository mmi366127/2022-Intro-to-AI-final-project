
from audioop import bias
from cmath import tanh
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32


class RandomCNN(nn.Module):
    def __init__(self):
        super(RandomCNN, self).__init__()

        # 2-D CNN
        self.conv1 = nn.Conv2d(1, OUT_CHANNELS, kernel_size=(3, 1), stride=1, padding=0)
        self.LeakyReLU = nn.LeakyReLU(0.2)

        # Set the random parameters to be constant.
        weight = torch.randn(self.conv1.weight.data.shape)
        self.conv1.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias = torch.zeros(self.conv1.bias.data.shape)
        self.conv1.bias = torch.nn.Parameter(bias, requires_grad=False)

    def forward(self, x_delta):
        out = self.LeakyReLU(self.conv1(x_delta))
        return out


    # some loss functions 
    @classmethod
    def compute_content_loss(cls, a_C, a_G):
        """
        Compute the content cost
        Arguments:
        a_C -- tensor of dimension (1, n_C, n_H, n_W)
        a_G -- tensor of dimension (1, n_C, n_H, n_W)
        Returns:
        J_content -- scalar that you compute using equation 1 above
        """
        m, n_C, n_H, n_W = a_G.shape

        # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
        a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
        a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

        # Compute the cost
        J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

        return J_content

    @classmethod
    def gram(cls, A):
        """
        Argument:
        A -- matrix of shape (n_C, n_L)
        Returns:
        GA -- Gram matrix of shape (n_C, n_C)
        """
        GA = torch.matmul(A, A.t())

        return GA

    @classmethod
    def gram_over_time_axis(cls, A):
        """
        Argument:
        A -- matrix of shape (1, n_C, n_H, n_W)
        Returns:
        GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
        """
        m, n_C, n_H, n_W = A.shape

        # Reshape the matrix to the shape of (n_C, n_L)
        # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
        A_unrolled = A.view(m * n_C * n_H, n_W)
        GA = torch.matmul(A_unrolled, A_unrolled.t())

        return GA

    @classmethod
    def compute_layer_style_loss(cls, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_C, n_H, n_W)
        a_G -- tensor of dimension (1, n_C, n_H, n_W)
        Returns:
        J_style_layer -- tensor representing a scalar style cost.
        """
        m, n_C, n_H, n_W = a_G.shape

        # Reshape the matrix to the shape of (n_C, n_L)
        # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

        # Calculate the gram
        # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
        # not along n_H * n_W. But is the result the same? No.
        GS = cls.gram_over_time_axis(a_S)
        GG = cls.gram_over_time_axis(a_G)

        # Computing the loss
        J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

        return J_style_layer


"""
a_random = Variable(torch.randn(1, 1, 257, 430)).float()
model = RandomCNN()
a_O = model(a_random)
print(a_O.shape)
"""

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
    pass
