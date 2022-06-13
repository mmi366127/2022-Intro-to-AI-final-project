

import numpy as np 
import librosa

N_FFT = 512

sampleRate = 22050

def wav2spectrum(x):
    S = librosa.stft(x, n_fft = N_FFT)
    S = np.log1p(np.abs(S))
    return S


def spectrum2wav(S):
    # reconstruct the waveform from spectrum
    a = np.exp(S) - 1
    p = 2 * np.pi * np.random.random_sample(S.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft = N_FFT))
    return x


def wav2spectrum_keep_phase(x, N_FFT):
    S = librosa.stft(x, n_fft = N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, p


def spectrum2wav_keep_phase(spectrum, p, N_FFT):
    a = np.exp(spectrum) - 1
    for i in range(100):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft = N_FFT))
    return x

