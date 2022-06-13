

import torch.nn as nn
import numpy as np
import torch


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        # Custom Implementation because the Voice Conversion Cycle GAN
        # paper assumes GLU won't reduce the dimension of tensor by 2.

    def forward(self, input):
        return input * torch.sigmoid(input)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        # Custom Implementation because PyTorch PixelShuffle requires,
        # 4D input. Whereas, in this case we have have 3D array
        self.upscale_factor = upscale_factor

    def forward(self, input):
        n = input.shape[0]
        c_out = input.shape[1] // 2
        w_new = input.shape[2] * 2
        return input.view(n, c_out, w_new)


class ResidualLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualLayer, self).__init__()

        self.conv1d_layer = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding), 
            nn.InstanceNorm1d(num_features = out_channels, affine = True)
        )

        self.conv_layer_gates = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm1d(num_features = out_channels, affine = True)
        )

        self.conv1d_out_layer = nn.Sequential(
            nn.Conv1d(in_channels = out_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm1d(num_features = in_channels, affine = True)
        )

    def forward(self, x):

        h1_norm = self.conv1d_layer(x)
        h1_gates_norm = self.conv_layer_gates(x)

        # GLU
        h1_glu = h1_norm * torch.sigmoid(h1_gates_norm)

        h2_norm = self.conv1d_out_layer(h1_glu)
        
        return x + h2_norm


class downSample_Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample_Generator, self).__init__()

        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm2d(num_features = out_channels, affine = True)
        )

        self.convLayer_gates = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm2d(num_features = out_channels, affine = True)
        )

    def forward(self, input):
        # GLU
        return self.convLayer(input) * torch.sigmoid(self.convLayer_gates(input))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        # 2D Conv Layer 
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (5, 15), stride = (1, 1), padding = (2, 7))

        self.conv1_gates = nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (5, 15), stride = 1, padding = (2, 7))

        # 2D Downsample Layer
        self.downSample1 = downSample_Generator(in_channels = 128, out_channels = 256, kernel_size = 5, stride = 2, padding = 2)

        self.downSample2 = downSample_Generator(in_channels = 256, out_channels = 256, kernel_size = 5, stride = 2, padding = 2)

        # 2D -> 1D Conv
        self.conv2dto1dLayer = nn.Sequential(
            nn.Conv1d(in_channels = 2304, out_channels = 256, kernel_size = 1, stride = 1, padding = 0),
            nn.InstanceNorm1d(num_features = 256, affine = True)
        )

        # Residual Blocks
        self.residualLayer1 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)

        self.residualLayer2 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)

        self.residualLayer3 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        
        self.residualLayer4 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        
        self.residualLayer5 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        
        self.residualLayer6 = ResidualLayer(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)

        # 1D -> 2D Conv
        self.conv1dto2dLayer = nn.Sequential(
            nn.Conv1d(in_channels = 256, out_channels = 2304, kernel_size = 1, stride = 1, padding = 0),
            nn.InstanceNorm1d(num_features = 2304, affine = True)
        )

        # UpSample Layer
        self.upSample1 = self.upSample(in_channels = 256, out_channels = 1024, kernel_size = 5, stride = 1, padding = 2)

        self.upSample2 = self.upSample(in_channels = 256, out_channels = 512, kernel_size = 5, stride = 1, padding = 2)

        self.lastConvLayer = nn.Conv2d(in_channels = 128, out_channels = 1, kernel_size = (5, 15), stride = (1, 1), padding = (2, 7))

    def downSample(self, in_channels, out_channels, kernel_size, stride, padding):

        self.ConvLayer = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.InstanceNorm1d(num_features = out_channels, affine = True),
            GLU()
        )

        return self.ConvLayer

    def upSample(self, in_channels, out_channels, kernel_size, stride, padding):
        
        self.convLayer = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding),
            nn.PixelShuffle(upscale_factor = 2),
            nn.InstanceNorm2d(num_features = out_channels // 4, affine = True),
            GLU()
        )

        return self.convLayer

    def forward(self, input):
        # GLU
        # print("Generator forward input: ", input.shape)
        input = input.unsqueeze(1)
        # print("Generator forward input: ", input.shape)
        conv1 = self.conv1(input) * torch.sigmoid(self.conv1_gates(input))
        # print("Generator forward conv1: ", conv1.shape)

        # DownloadSample
        downsample1 = self.downSample1(conv1)
        # print("Generator forward downsample1: ", downsample1.shape)
        downsample2 = self.downSample2(downsample1)
        # print("Generator forward downsample2: ", downsample2.shape)

        # 2D -> 1D
        # reshape
        reshape2dto1d = downsample2.view(downsample2.size(0), 2304, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)
        # print("Generator forward reshape2dto1d: ", reshape2dto1d.shape)
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        # print("Generator forward conv2dto1d_layer: ", conv2dto1d_layer.shape)

        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # print("Generator forward residual_layer_6: ", residual_layer_6.shape)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        # print("Generator forward conv1dto2d_layer: ", conv1dto2d_layer.shape)
        # reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 9, -1)
        # print("Generator forward reshape1dto2d: ", reshape1dto2d.shape)

        # UpSample
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        # print("Generator forward upsample_layer_1: ", upsample_layer_1.shape)
        upsample_layer_2 = self.upSample2(upsample_layer_1)
        # print("Generator forward upsample_layer_2: ", upsample_layer_2.shape)

        output = self.lastConvLayer(upsample_layer_2)
        # print("Generator forward output: ", output.shape)
        output = output.squeeze(1)
        # print("Generator forward output: ", output.shape)
        return output


if __name__ == '__main__':

    x = torch.randn((20, 1, 192, 24))

    model = Generator()

    x = model(x)

    print(x.size())