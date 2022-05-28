
import matplotlib.pyplot as plt
from packaging import version
import numpy as np
import soundfile
import librosa
import torch 


def writeFile(fileName, x, sampleRate):
    if version.parse(librosa.__version__) < version.parse('0.8.0'):
        librosa.output.write_wav(fileName, x, sampleRate)
    else:
        soundfile.write(fileName, x, sampleRate)



def wav2spectrum(fileName, N_FFT):
    x, sampleRate = librosa.load(fileName)
    S = librosa.stft(x, n_fft = N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S))

    return S, sampleRate


def spectrum2wav(spectrum, sampleRate, fileName, N_FFT):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    p = 2 * np.pi * np.random.random_sample(spectrum.shape) - np.pi
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft = N_FFT))
    writeFile(fileName, x, sampleRate)


def wav2spectrum_keep_phase(filename, N_FFT):
    x, sr = librosa.load(filename)
    S = librosa.stft(x, n_fft = N_FFT)
    p = np.angle(S)

    S = np.log1p(np.abs(S))
    return S, p, sr


def spectrum2wav_keep_phase(spectrum, p, sampleRate, fileName, N_FFT):
    # Return the all-zero vector with the same shape of `a_content`
    a = np.exp(spectrum) - 1
    for i in range(50):
        S = a * np.exp(1j * p)
        x = librosa.istft(S)
        p = np.angle(librosa.stft(x, n_fft = N_FFT))
    writeFile(fileName, x, sampleRate)


def compute_content_loss(a_C, a_G):
    """
    Compute the content cost
    Arguments:
    a_C -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_content -- scalar that you compute using equation 1 above
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    a_C_unrolled = a_C.view(m * n_C, n_H * n_W)
    a_G_unrolled = a_G.view(m * n_C, n_H * n_W)

    # Compute the cost
    J_content = 1.0 / (4 * m * n_C * n_H * n_W) * torch.sum((a_C_unrolled - a_G_unrolled) ** 2)

    return J_content


def gram(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_L)
    Returns:
    GA -- Gram matrix of shape (n_C, n_C)
    """
    GA = torch.matmul(A, A.t())

    return GA


def gram_over_time_axis(A):
    """
    Argument:
    A -- matrix of shape (1, n_C, n_H, n_W)
    Returns:
    GA -- Gram matrix of A along time axis, of shape (n_C, n_C)
    """
    m, n_C, n_H, n_W = A.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)
    A_unrolled = A.view(m * n_C * n_H, n_W)
    GA = torch.matmul(A_unrolled, A_unrolled.t())

    return GA


def compute_layer_style_loss(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_C, n_H, n_W)
    a_G -- tensor of dimension (1, n_C, n_H, n_W)
    Returns:
    J_style_layer -- tensor representing a scalar style cost.
    """
    m, n_C, n_H, n_W = a_G.shape

    # Reshape the matrix to the shape of (n_C, n_L)
    # Reshape a_C and a_G to the (m * n_C, n_H * n_W)

    # Calculate the gram
    # !!!!!! IMPORTANT !!!!! Here we compute the Gram along n_C,
    # not along n_H * n_W. But is the result the same? No.
    GS = gram_over_time_axis(a_S)
    GG = gram_over_time_axis(a_G)

    # Computing the loss
    J_style_layer = 1.0 / (4 * (n_C ** 2) * (n_H * n_W)) * torch.sum((GS - GG) ** 2)

    return J_style_layer



def plot_curve(content_loss, style_loss, total_loss, title = 'loss', same_y_scale = True):

    x = np.arange(len(content_loss))
    
    fig, axes = plt.subplots(1, 2, figsize = (12, 6))

    # plot total loss
    color = 'tab:blue'
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('total loss', color = color)

    axes[0].plot(x, total_loss, color = color)
    axes[0].tick_params(axis = 'y', labelcolor = color)

    axes[1].set_xlabel('epoch', color = color)
    axes[1].set_ylabel('content loss', color = color)
    
    axes[1].plot(x, content_loss, color = color)
    axes[1].tick_params(axis = 'y', labelcolor = color)

    color = 'tab:red'
    
    if same_y_scale:
        axes[1].set_ylabel('style loss', color = color)
        axes[1].plot(x, style_loss, color = color)
    else:
        ax3 = plt.twinx(axes[1])
        ax3.set_ylabel('style loss', color = color)
        ax3.plot(x, style_loss, color = color)
        ax3.tick_params(axis = 'y', labelcolor = color)

    fig.tight_layout()
    plt.show()
    



if __name__ == '__main__':


    plot_curve([1, 2, 3], [4, 5, 6], [5, 7, 9])


