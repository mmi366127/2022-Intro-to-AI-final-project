
from torchaudio.transforms import Vol
import matplotlib.pyplot as plt
from packaging import version
from pytube import YouTube
import librosa.display
import numpy as np
import subprocess
import soundfile
import librosa
import torch


def download_youtube_to_wav(url, parentdir):
    yt = YouTube(url)
    video = yt.streams.filter(only_audio = True).first()
    outpath = video.download(parentdir)
    print(outpath[:-4])
    subprocess.run(['ffmpeg', '-i', outpath, outpath[:-4] + '.wav'])

def scale_audio(x, scale):
    x = torch.tensor(x)
    return Vol(gain = scale)(x).detach().cpu().numpy()

def add_audio(x, db):
    x = torch.tensor(x)
    return Vol(gain = db, gain_type = 'db')(x).detach().cpu().numpy()

def writeFile(fileName, x, sampleRate):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(fileName, x, sampleRate)
    else:
        soundfile.write(fileName, x, sampleRate)

def loadFile(filename, sampleRate):
    x, sr = librosa.load(filename, sr = sampleRate)
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


def plot_spectrum(A, B, title_A = 'Source', title_B = 'Generated'):

    fig, axs = plt.subplots(ncols = 2, figsize = (10, 10))
    axs[0].imshow(np.flip(A, -2), cmap = None)
    axs[0].axis('off')
    axs[0].set_title(title_A)
    axs[1].imshow(np.flip(B, -2), cmap = None)
    axs[1].axis('off')
    axs[1].set_title(title_B)
    plt.show()



def plot_spectrogram(spec, sampleRate):
    D = np.abs(spec)
    db = librosa.amplitude_to_db(D, ref = np.max)
    librosa.display.specshow(db, sr = sampleRate, y_axis = 'log', x_axis = 'time')


