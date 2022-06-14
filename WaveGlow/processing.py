from torchaudio.transforms import MelSpectrogram
import librosa.display
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import librosa
import torch.nn as nn
import torch 
from librosa.filters import mel as librosa_mel_fn
from torch_stft import STFT
from scipy.io.wavfile import read, write
from vars import *
from models import WaveGlow

"""
Some Codes are from https://github.com/NVIDIA/tacotron2/tree/185cd24e046cc1304b4f8e564734d2498c6e2e6f
"""

mel_basis = librosa_mel_fn(sample_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
mel_basis = torch.from_numpy(mel_basis).float()
stft = STFT(filter_length = filter_length, hop_length = hop_length, win_length = win_length, window = 'hann')

def compression(x, C = 1,clip_val = 1e-5):
    return torch.log((torch.clamp(x, min = clip_val)) * C)

def decompression(x, C = 1):
    return torch.exp(x) / C

def spectral_normalize(magnitudes):
    output = compression(magnitudes)
    return output

def spectral_de_normalize(magnitudes):
    output = decompression(magnitudes)
    return output

def mel_spectrogram(x):
    magnitudes, phases = stft.transform(x)
    magnitudes = magnitudes.data
    mel_output = torch.matmul(mel_basis, magnitudes)
    mel_output = spectral_normalize(mel_output)
    return mel_output

def wav2spectrum(x):
    x = x / MAX_WAV_VALUE
    x = x.unsqueeze(0)
    x = torch.Tensor(x)
    S = mel_spectrogram(x)
    S = torch.squeeze(S, 0)
    return S.detach().cpu()


if __name__ == '__main__':
    sampleRate, data = read('./LJ001-0001.wav')
    x = torch.from_numpy(data).float()
    # # # print(sampleRate)
    # x = wav2spectrum(x)
    # x = spectrum2wav(x)
    # x = add_audio(x, 20)
#    writeFile('./test.wav', x)

    # sw = Mel2Samp('./LJ001-0001.wav', 22050, filter_length, hop_length, win_length, sampleRate, mel_fmin, mel_fmax)

    x = wav2spectrum(torch.Tensor(x))
    wg_path = './waveglow_256channels_universal_v5.pt'
    wg = WaveGlow.load_state_dict(torch.load(wg_path))
    wg = wg.remove_weightnorm(wg)
    # x = wav2spectrum(x)
    wg.cuda().eval()
    mel = torch.Tensor(x)
    mel = torch.autograd.Variable(mel.cuda())
    mel = torch.unsqueeze(mel, dim = 0)
    # print(mel.shape)
    with torch.no_grad():
        audio = wg.infer(mel, sigma = 1.0)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    # audio = add_audio(audio, 40)
    audio = audio.astype('int16')

    write('./test.wav', sample_rate, audio)