import matplotlib.pyplot as plt
import torch
import torchaudio.datasets as datasets
import librosa
import numpy as np


def load_librispeech(num_samples=1):
    dataset = datasets.LIBRISPEECH(root='./data/', url="dev-clean", download=True)
    
    data = []
    for i in range(num_samples):
        current = dataset[i][0]
        
        x = []
        for j in range(len(current[0]) // 16384 - 1):
            y = current[:, 16384*j + 8192:16384*(j + 1) + 8192]
            y = y / y.std() + y.mean()
            x.append(y)
        data.append(torch.stack(x))

    return data


def get_noisy_audio(clean, std=0.1):
    noisy = []
    for c in clean:
        noisy.append(c + torch.randn_like(c)*std)

    return noisy


def to_fourier(x):
    x = x.squeeze()
    x = librosa.stft(x.cpu().numpy(), n_fft=1023, hop_length=64)
    x = torch.from_numpy(x)
    x = torch.stack([x.real, x.imag])
    return x


def from_fourier(x):
    x = x.squeeze()
    x = torch.complex(x[0], x[1])
    x = librosa.istft(x.cpu().numpy(), n_fft=1023, hop_length=64, length=16384)
    x = torch.from_numpy(x)
    return x


def plot_waveform(waveform: torch.Tensor, sample_rate=16000):
    waveform = waveform.squeeze().numpy()

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(waveform, sr=sample_rate)
    plt.title("Waveform")
    plt.show()


def plot_spectrogram(waveform: torch.Tensor, sample_rate=16000):
    waveform = waveform.squeeze().numpy()

    plt.figure(figsize=(10, 4))
    spec = librosa.stft(waveform, n_fft=512)
    spec = np.abs(spec)
    spec = librosa.amplitude_to_db(spec, ref=np.max)
    librosa.display.specshow(spec, sr=sample_rate, x_axis='time', y_axis='linear', hop_length=128)
    plt.title("Spectrogram")
    plt.show()
