
import torch.nn.functional as F
import torch

def mae(x, y):
    return torch.mean(torch.abs(x - y))


def mse(x, y):
    return torch.mean((x - y)**2)


def loss_travel(siam_x_1, siam_x_1_gen, siam_x_2, siam_x_2_gen):
    L1 = torch.mean(((siam_x_1 - siam_x_2) - (siam_x_1_gen - siam_x_2_gen))**2)
    L2 = torch.mean(torch.sum(-(F.normalize(siam_x_1 - siam_x_2, p = 2, dim = -1) * F.normalize(siam_x_1_gen - siam_x_2_gen, p = 2, dim = -1)), dim = -1))
    return L1 + L2


def loss_siamses(siam_x_1, siam_x_2, zero, delta):
    logits = torch.sqrt(torch.sum((siam_x_1 - siam_x_2)**2 ,dim = -1, keepdims = True))
    return torch.mean(torch.square(torch.maximum(delta - logits, zero)))


def d_loss_f(fake, zero):
    return torch.mean(torch.maximum(1 + fake, zero))


def d_loss_r(real, zero):
    return torch.mean(torch.maximum(1 - real, zero))


def g_loss_f(fake):
    return torch.mean(-fake)
    
