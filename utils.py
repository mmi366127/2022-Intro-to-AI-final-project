
from audioop import add
from plistlib import load
from torchaudio.transforms import MelSpectrogram, InverseMelScale, InverseSpectrogram, GriffinLim, Vol
import matplotlib.pyplot as plt
from packaging import version
from pytube import YouTube
import librosa.display
import numpy as np
import subprocess
import soundfile
import librosa
import torch 


hop = 192
N_FFT = hop * 6 

sampleRate = 22050

min_level_db = -100
ref_level_db = 20

def download_youtube_to_wav(url, parentdir):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio = True).first()
    outpath = video.download(parentdir)
    print(outpath[:-4])
    subprocess.run(['ffmpeg', '-i', outpath, outpath[:-4] + '.wav'])


melspecobj = MelSpectrogram(n_fft = N_FFT, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, win_length = 6 * hop, hop_length = hop, pad = 0, power = 2, normalized = True)
melspecfunc = melspecobj.forward

griff = GriffinLim(n_fft = N_FFT, win_length = 6 * hop, hop_length = hop, power = 2)
grifffunc = griff.forward

invMelS = InverseMelScale(n_stft = N_FFT // 2 + 1, n_mels = hop, sample_rate = sampleRate, f_min = 0.0, max_iter = 10000)
invMelsfunc = invMelS.forward


def writeFile(fileName, x):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(fileName, x, sampleRate)
    else:
        soundfile.write(fileName, x, sampleRate)

def loadFile(filename):
    x, sr = librosa.load(filename, sr = sampleRate)
    return x

def normalize(S):
    return np.clip((((S - min_level_db) / -min_level_db)*2.)- 1.0, -1, 1)

def denormalize(S):
    return (((np.clip(S, -1, 1) + 1.0) / 2.0) * -min_level_db) + min_level_db

def wav2spectrum(x):
    x = np.array(torch.squeeze(melspecfunc(torch.Tensor(x))).detach().cpu())
    x = librosa.power_to_db(x) - ref_level_db
    return normalize(x)
    # S = librosa.stft(x, n_fft = N_FFT)
    # S = np.log1p(np.abs(S))
    # return S

def spectrum2wav(spectrum):
    # Reconstruct the audio from spectrum
    x = denormalize(spectrum) + ref_level_db
    x = librosa.db_to_power(x)
    x = torch.Tensor(x)
    S = invMelsfunc(x)
    S = grifffunc(S)
    return np.array(S.detach().cpu())
    # return np.transpose(S, (1, 0))
    # a = np.exp(spectrum) - 1
    # p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    # for i in range(50):
    #     S = a * np.exp(1j * p)
    #     x = librosa.istft(S)
    #     p = np.angle(librosa.stft(x, n_fft = N_FFT))
    # return x

def scale_audio(x, scale):
    x = torch.tensor(x)
    return Vol(gain = scale)(x).detach().cpu().numpy()

def add_audio(x, db):
    x = torch.tensor(x)
    return Vol(gain = db, gain_type = 'db')(x).detach().cpu().numpy()

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

def plot_curve(content_loss, style_loss, total_loss, content_title = 'content loss', style_title = 'style loss', total_title = 'total loss', same_y_scale = True):

    x = np.arange(len(content_loss))
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))

    # plot total loss
    color = 'tab:blue'
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel(total_title, color = color)

    axes[0].plot(x, total_loss, color = color)
    axes[0].tick_params(axis = 'y', labelcolor = color)

    axes[1].set_xlabel('epoch', color = color)
    axes[1].set_ylabel(content_title, color = color)
    
    axes[1].plot(x, content_loss, color = color)
    axes[1].tick_params(axis = 'y', labelcolor = color)

    color = 'tab:red'
    
    if same_y_scale:
        axes[1].set_ylabel(style_title, color = color)
        axes[1].plot(x, style_loss, color = color)
    else:
        ax3 = plt.twinx(axes[1])
        ax3.set_ylabel(style_title, color = color)
        ax3.plot(x, style_loss, color = color)
        ax3.tick_params(axis = 'y', labelcolor = color)

    fig.tight_layout()
    plt.show()
    

def plot_spectrogram_with_raw_signal(signal, sr, title = 'spectrum'):
    plt.title('Spectrogram')
    plt.specgram(signal,Fs = sr)    
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.show()

def plot_spectrogram(spec):
    D = np.abs(spec)
    db = librosa.amplitude_to_db(D, ref = np.max)
    librosa.display.specshow(db, sr = sampleRate, y_axis = 'log', x_axis = 'time')


if __name__ == '__main__':

    x = loadFile('./input/laplus/5second.wav')
    print(sampleRate)
    x = wav2spectrum(x)
    x = spectrum2wav(x)
    x = add_audio(x, 20)
    writeFile('./test.wav', x)



    # plot_curve([1, 2, 3], [4, 5, 6], [5, 7, 9])


