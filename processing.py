
from torchaudio.transforms import MelScale, Spectrogram
import torchaudio.transforms as transforms
from packaging import version
import librosa.display
import numpy as np
import soundfile
import librosa
import torch 


sampleRate = 16000        # sample rate

hop = 192                 # hop size (window size = 6 * hop)
N_FFT = 6 * hop           # N_FFT

min_level_db = -100       # reference values to normalize data
ref_level_db = 20         


# Write file 
def writeFile(fileName, x):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(fileName, x, sampleRate)
    else:
        soundfile.write(fileName, x, sampleRate)


# Load and resample the sound 
def loadFile(filename):
    x, sr = librosa.load(filename)
    x = librosa.resample(y = x, orig_sr = sr, target_sr = sampleRate)
    return x

# Transform to Mel Scale spectrum
def wav2spectrum(x):
    S = librosa.feature.melspectrogram(y = x, sr = sampleRate, n_fft = N_FFT, hop_length = hop, win_length = hop * 6, n_mels = 256)
    return S

# Reconstruct the audio from Mel spectrum
def spectrum2wav(spectrum):
    S = librosa.feature.inverse.mel_to_stft(spectrum)
    x = librosa.griffinlim(S)
    return x

def wav2spectrum_keep_phase(x, N_FFT):
    S = librosa.stft(x, n_fft = N_FFT)
    p = np.angle(S)
    S = np.log1p(np.abs(S))
    return S, p


def spectrum2wav_keep_phase(spectrum, p, N_FFT):
    a = np.exp(spectrum) - 1
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft = N_FFT))
    return x