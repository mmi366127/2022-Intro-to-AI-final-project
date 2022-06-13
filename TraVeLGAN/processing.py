

from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim
import librosa.display
import numpy as np
import librosa
import torch 


N_FFT = hop * 6
hop = 192

sampleRate = 22050

min_level_db = -100
ref_level_db = 20

# Mel-Spectrum object and function
melspecobj = MelSpectrogram(n_fft = N_FFT, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, win_length = 6 * hop, hop_length = hop, pad = 0, power = 2, normalized = False)
melspecfunc = melspecobj.forward
"""
Old design not using
# GrinffinLim object and function
# griff = GriffinLim(n_fft = N_FFT, win_length = 6 * hop, hop_length = hop, power = 2)
# grifffunc = griff.forward

# # Inverse-Mel-Spectrum object and function
# invMelS = InverseMelScale(n_stft = N_FFT // 2 + 1, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, max_iter = 10000)
# invMelsfunc = invMelS.forward
"""


def normalize(S):
    return np.clip((((S - min_level_db) / -min_level_db)*2.)- 1.0, -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1) + 1.0) / 2.0) * -min_level_db) + min_level_db

def wav2spectrum(x):
    x = np.array(torch.squeeze(melspecfunc(torch.Tensor(x))).detach().cpu())
    x = librosa.power_to_db(x) - ref_level_db
    return normalize(x)

def GRAD(spec, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.003):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*hop)-hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def spectrum2wav(spectrum):
    # Reconstruct the audio from spectrum
    """
    Old design not using
    # x = denormalize(spectrum) + ref_level_db
    # x = librosa.db_to_power(x)
    # x = torch.Tensor(x)
    # S = invMelsfunc(x)
    # S = grifffunc(S)
    """

    x = denormalize(spectrum) + ref_level_db
    x = librosa.db_to_power(x)
    wv = GRAD(np.expand_dims(x, 0), melspecfunc, maxiter = 2000, evaiter = 10, tol = 1e-8)
    return np.array(S.detach().cpu())
    # return np.array(S.detach().cpu())
