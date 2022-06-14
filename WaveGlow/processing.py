from librosa.filters import mel as librosa_mel_fn
from torch_stft import STFT
from WaveGlow.vars import *
import torch 


"""
Some Codes are from https://github.com/NVIDIA/tacotron2/tree/185cd24e046cc1304b4f8e564734d2498c6e2e6f
"""

mel_basis = librosa_mel_fn(sr = sample_rate, n_fft = filter_length, n_mels = n_mel_channels, fmin = mel_fmin, fmax = mel_fmax)
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
    x = x[None, :]
    x = torch.Tensor(x)
    S = mel_spectrogram(x)
    S = torch.squeeze(S, 0)
    return S.detach().cpu()

class Denoiser(torch.nn.Module):
    """ Removes model bias from audio produced with waveglow """

    def __init__(self, waveglow, filter_length = 1024, n_overlap = 4,
                 win_length = 1024, mode='zeros'):
        super(Denoiser, self).__init__()
        # print(filter_length, int(filter_length / n_overlap), win_length)
        self.stft = STFT(filter_length = filter_length,
                         hop_length = int(filter_length/n_overlap),
                         win_length = win_length).cuda()
        if mode == 'zeros':
            mel_input = torch.zeros(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        elif mode == 'normal':
            mel_input = torch.randn(
                (1, 80, 88),
                dtype=waveglow.upsample.weight.dtype,
                device=waveglow.upsample.weight.device)
        else:
            raise Exception("Mode {} if not supported".format(mode))

        with torch.no_grad():
            bias_audio = waveglow.infer(mel_input, sigma=0.0).float()
            bias_spec, _ = self.stft.transform(bias_audio)

        self.register_buffer('bias_spec', bias_spec[:, :, 0][:, :, None])

    def forward(self, audio, strength=0.1):
        audio_spec, audio_angles = self.stft.transform(audio.cuda().float())
        audio_spec_denoised = audio_spec - self.bias_spec * strength
        audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
        audio_denoised = self.stft.inverse(audio_spec_denoised, audio_angles)
        return audio_denoised

#def denoise(waveglow, audio, mode = 'zeros', strength = 0.1):
#    global stft
#    if mode == 'zeros':
#        mel_input = torch.zeros(
#            (1, 80, 88),
#            dtype=waveglow.upsample.weight.dtype,
#            device=waveglow.upsample.weight.device)
#    elif mode == 'normal':
#        mel_input = torch.randn(
#            (1, 80, 88),
#            dtype=waveglow.upsample.weight.dtype,
#            device=waveglow.upsample.weight.device)
#    stft = stft.cuda()
#    with torch.no_grad():
#        bias_audio = waveglow.infer(mel_input, sigma = 0.0).float()
#        bias_spec, _ = stft.transform(bias_audio)
#
#    bias_spec = bias_spec[:, :, 0][:, :, None]
#    audio_spec, audio_angles = stft.transform(audio.cuda().float())
#    audio_spec_denoised = audio_spec - bias_spec * strength
#    audio_spec_denoised = torch.clamp(audio_spec_denoised, 0.0)
#    audio_denosised = stft.inverse(audio_spec_denoised, audio_angles)
#
#    return audio_denosised

if __name__ == '__main__':


    from scipy.io.wavfile import read, write


    sampleRate, data = read('./LJ001-0001.wav')
    # data, sampleRate = librosa.load('./LJ001-0001.wav')
    x = torch.from_numpy(data).float()
    # # # print(sampleRate)
    # x = wav2spectrum(x)
    # x = spectrum2wav(x)
    # x = add_audio(x, 20)
#    writeFile('./test.wav', x)

    # sw = Mel2Samp('./LJ001-0001.wav', 22050, filter_length, hop_length, win_length, sampleRate, mel_fmin, mel_fmax)

    x = wav2spectrum(torch.Tensor(x))
    wg_path = './waveglow_256channels_universal_v5.pt'
    wg = torch.load(wg_path)['model']
    wg = wg.remove_weightnorm(wg)
    # x = wav2spectrum(x)
    wg.cuda().eval()
    mel = torch.Tensor(x)
    mel = torch.autograd.Variable(mel.cuda())
    mel = torch.unsqueeze(mel, dim = 0)
    denoiser = Denoiser(wg)
    for i in denoiser.parameters():
        print(i)
    print('done')
    # print(mel.shape)
    with torch.no_grad():
        audio = wg.infer(mel, sigma = 1.0)
        audio = denoise(wg, audio)
        audio = audio * MAX_WAV_VALUE
    audio = audio.squeeze()
    audio = audio.cpu().numpy()
    # audio = add_audio(audio, 40)
    audio = audio.astype('int16')

    write('./test.wav', sample_rate, audio)
