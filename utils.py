
import matplotlib.pyplot as plt
from packaging import version
import librosa.display
import numpy as np
import soundfile
import librosa
import torch 


def writeFile(fileName, x, sampleRate):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(fileName, x, sampleRate)
    else:
        soundfile.write(fileName, x, sampleRate)


def loadFile(filename):
    x, sampleRate = librosa.load(filename)
    return x, sampleRate


def wav2spectrum(x, N_FFT):
    S = librosa.stft(x, n_fft = N_FFT)
    S = np.log1p(np.abs(S))
    return S

def spectrum2wav(spectrum, N_FFT):
    # Reconstruct the audio from spectrum
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
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
    for i in range(50):
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

def plot_spectrogram(spec, sr):
    D = np.abs(spec)
    db = librosa.amplitude_to_db(D, ref = np.max)
    librosa.display.specshow(db, sr = sr, y_axis = 'log', x_axis = 'time')


if __name__ == '__main__':


    plot_curve([1, 2, 3], [4, 5, 6], [5, 7, 9])


