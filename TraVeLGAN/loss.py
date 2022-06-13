
import torch.nn.functional as F
import torch

def mse(x, y):
    return torch.mean((x - y)**2)

def loss_travel(siam_x_1, siam_x_1_gen, siam_x_2, siam_x_2_gen):
    L1 = torch.mean(torch.norm(((siam_x_1 - siam_x_2) - (siam_x_1_gen - siam_x_2_gen)), p = 2, dim = -1))
    L2 = torch.mean(torch.sum((F.normalize(siam_x_1 - siam_x_2, p = 2, dim = -1) * F.normalize(siam_x_1_gen - siam_x_2_gen, p = 2, dim = -1)), dim = -1))
    return L1 + L2

def loss_margin(siam_x_1, siam_x_2, delta):
    logits = torch.sqrt(torch.sum((siam_x_1 - siam_x_2)**2 ,dim = -1, keepdims = True))
    return torch.mean(F.relu(delta - logits))

def d_loss_f(fake):
    return torch.mean(F.relu(1 + fake))

def d_loss_r(real):
    return torch.mean(F.relu(1 - real))

def g_loss_f(fake):
    return torch.mean(-fake)
    
