

from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim
import librosa.display
import numpy as np
import librosa
import torch 


hop = 192
N_FFT = hop * 6
sampleRate = 22050

min_level_db = -100
ref_level_db = 20

# Mel-Spectrum object and function
melspecobj = MelSpectrogram(n_fft = N_FFT, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, win_length = 6 * hop, hop_length = hop, pad = 0, power = 2, normalized = True)
melspecfunc = melspecobj.forward

# GrinffinLim object and function
griff = GriffinLim(n_fft = N_FFT, win_length = 6 * hop, hop_length = hop, power = 2)
grifffunc = griff.forward

# Inverse-Mel-Spectrum object and function
invMelS = InverseMelScale(n_stft = N_FFT // 2 + 1, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, max_iter = 10000)
invMelsfunc = invMelS.forward


def normalize(S):
    return np.clip((((S - min_level_db) / -min_level_db)*2.)- 1.0, -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1) + 1.0) / 2.0) * -min_level_db) + min_level_db

def wav2spectrum(x):
    x = np.array(torch.squeeze(melspecfunc(torch.Tensor(x))).detach().cpu())
    x = librosa.power_to_db(x) - ref_level_db
    return normalize(x)

def spectrum2wav(spectrum):
    # Reconstruct the audio from spectrum
    x = denormalize(spectrum) + ref_level_db
    x = librosa.db_to_power(x)
    x = torch.Tensor(x)
    S = invMelsfunc(x)
    S = grifffunc(S)
    return np.array(S.detach().cpu())





