

n_mel_channels = 80
filter_length = 1024
hop_length = 256
hop = 256
sample_rate = 22050
win_length = 1024
mel_fmin = 0.0
mel_fmax = 8000.0
MAX_WAV_VALUE = 32768.0


# model configuration for the pretrained model

model_config = {
    "n_mel_channels": 80,
    "n_flows": 12,
    "n_group": 8,
    "n_early_every": 4,
    "n_early_size": 2,
    "WN_config": {
        "n_layers": 8,
        "n_channels": 256,
        "kernel_size": 3
    }
}
