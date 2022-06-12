

from torch.nn.utils import clip_grad_norm_
# from sklearn.interpolate import interp1d
# from sklearn.metrics import roc_curve
from scipy.optimize import brentq
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch



N_FFT = 512
N_CHANNELS = round(1 + N_FFT/2)
OUT_CHANNELS = 32

# Random CNN model

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

# TraVeLGan model

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
        
#  model



class SpeakerEncoder(nn.Module):
    def __init__(self, device, loss_device):
        super().__init__()
        self.loss_device = loss_device
        
        # Network defition
        self.lstm = nn.LSTM(input_size=mel_n_channels,
                            hidden_size=model_hidden_size, 
                            num_layers=model_num_layers, 
                            batch_first=True).to(device)
        self.linear = nn.Linear(in_features=model_hidden_size, 
                                out_features=model_embedding_size).to(device)
        self.relu = torch.nn.ReLU().to(device)
        
        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = nn.Parameter(torch.tensor([10.])).to(loss_device)
        self.similarity_bias = nn.Parameter(torch.tensor([-5.])).to(loss_device)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss().to(loss_device)
        
    def do_gradient_ops(self):
        # Gradient scale
        self.similarity_weight.grad *= 0.01
        self.similarity_bias.grad *= 0.01
            
        # Gradient clipping
        clip_grad_norm_(self.parameters(), 3, norm_type=2)
    
    def forward(self, utterances, hidden_init=None):
        """
        Computes the embeddings of a batch of utterance spectrograms.
        
        :param utterances: batch of mel-scale filterbanks of same duration as a tensor of shape 
        (batch_size, n_frames, n_channels) 
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers, 
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """
        # Pass the input through the LSTM layers and retrieve all outputs, the final hidden state
        # and the final cell state.
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        
        # We take only the hidden state of the last layer
        embeds_raw = self.relu(self.linear(hidden[-1]))
        
        # L2-normalize it
        embeds = embeds_raw / (torch.norm(embeds_raw, dim=1, keepdim=True) + 1e-5)        

        return embeds
    
    def similarity_matrix(self, embeds):
        """
        Computes the similarity matrix according the section 2.1 of GE2E.

        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the similarity matrix as a tensor of shape (speakers_per_batch,
        utterances_per_speaker, speakers_per_batch)
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Inclusive centroids (1 per speaker). Cloning is needed for reverse differentiation
        centroids_incl = torch.mean(embeds, dim=1, keepdim=True)
        centroids_incl = centroids_incl.clone() / (torch.norm(centroids_incl, dim=2, keepdim=True) + 1e-5)

        # Exclusive centroids (1 per utterance)
        centroids_excl = (torch.sum(embeds, dim=1, keepdim=True) - embeds)
        centroids_excl /= (utterances_per_speaker - 1)
        centroids_excl = centroids_excl.clone() / (torch.norm(centroids_excl, dim=2, keepdim=True) + 1e-5)

        # Similarity matrix. The cosine similarity of already 2-normed vectors is simply the dot
        # product of these vectors (which is just an element-wise multiplication reduced by a sum).
        # We vectorize the computation for efficiency.
        sim_matrix = torch.zeros(speakers_per_batch, utterances_per_speaker,
                                 speakers_per_batch).to(self.loss_device)
        mask_matrix = 1 - np.eye(speakers_per_batch, dtype=np.int)
        for j in range(speakers_per_batch):
            mask = np.where(mask_matrix[j])[0]
            sim_matrix[mask, :, j] = (embeds[mask] * centroids_incl[j]).sum(dim=2)
            sim_matrix[j, :, j] = (embeds[j] * centroids_excl[j]).sum(dim=1)
        
        ## Even more vectorized version (slower maybe because of transpose)
        # sim_matrix2 = torch.zeros(speakers_per_batch, speakers_per_batch, utterances_per_speaker
        #                           ).to(self.loss_device)
        # eye = np.eye(speakers_per_batch, dtype=np.int)
        # mask = np.where(1 - eye)
        # sim_matrix2[mask] = (embeds[mask[0]] * centroids_incl[mask[1]]).sum(dim=2)
        # mask = np.where(eye)
        # sim_matrix2[mask] = (embeds * centroids_excl).sum(dim=2)
        # sim_matrix2 = sim_matrix2.transpose(1, 2)
        
        sim_matrix = sim_matrix * self.similarity_weight + self.similarity_bias
        return sim_matrix
    
    def loss(self, embeds):
        """
        Computes the softmax loss according the section 2.1 of GE2E.
        
        :param embeds: the embeddings as a tensor of shape (speakers_per_batch, 
        utterances_per_speaker, embedding_size)
        :return: the loss and the EER for this batch of embeddings.
        """
        speakers_per_batch, utterances_per_speaker = embeds.shape[:2]
        
        # Loss
        sim_matrix = self.similarity_matrix(embeds)
        sim_matrix = sim_matrix.reshape((speakers_per_batch * utterances_per_speaker, 
                                         speakers_per_batch))
        ground_truth = np.repeat(np.arange(speakers_per_batch), utterances_per_speaker)
        target = torch.from_numpy(ground_truth).long().to(self.loss_device)
        loss = self.loss_fn(sim_matrix, target)
        
        # EER (not backpropagated)
        with torch.no_grad():
            inv_argmax = lambda i: np.eye(1, speakers_per_batch, i, dtype=np.int)[0]
            labels = np.array([inv_argmax(i) for i in ground_truth])
            preds = sim_matrix.detach().cpu().numpy()

            # Snippet from https://yangcha.github.io/EER-ROC/
            fpr, tpr, thresholds = roc_curve(labels.flatten(), preds.flatten())           
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            
        return loss, eer


if __name__ == '__main__':
    
    x = torch.rand((20, 1, 192, 24))

    model = Siamese(input_size = (1, 192, 24))

    print(x.size())

    x = model(x)

    print(x.size())
