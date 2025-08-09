import numpy as np
from scipy import signal, ndimage
import matplotlib.pyplot as plt
import textwrap


def bandpass(x, low, high, fs):
    band = [2 * low/fs, 2 * high/fs]
    b, a = signal.butter(3, band, btype="bandpass")
    x = signal.filtfilt(b, a, x, axis=-1)

    return x


def lowpass(x, high, fs):
    band = 2 * high/fs
    b, a = signal.butter(3, band, btype="low")
    x = signal.filtfilt(b, a, x, axis=-1)

    return x


def substract_median(x, channel_axis=0):
    median = np.median(x, axis=channel_axis, keepdims=True)
    return x - median


def normalize(x, k=32):
    local_mean = ndimage.uniform_filter1d(x, size=k, axis=1, mode='nearest')

    local_var = ndimage.uniform_filter1d(x**2, size=k, axis=1, mode='nearest') - local_mean**2
    local_std = np.sqrt(np.maximum(local_var, 0))  # Ensure non-negative variance

    normalized = x / (local_std + 1e-8)

    return normalized, local_std


# TODO add DAS plotting utils

# def plot_row(images, labels=[], clip=2, cbar=True, path=''):
#     cols = len(images)
#     fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 8))

#     for i, img in enumerate(images):
#         if clip is None:
#             clip = np.percentile(img, 99)
      
#         im = axes[i].imshow(img, origin='lower', aspect='auto', cmap='seismic', interpolation='none', vmin=-clip, vmax=clip)
#         # axes[i].axis('off')
        
#         if labels and i < len(labels):
#             label = "\n".join(textwrap.wrap(labels[i], 22))
#             axes[i].set_title(label, fontsize=10)

#         if cbar:
#             cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.01)
#             cbar.ax.set_position([0.91, 0.1, 0.2, 0.8])

#     if path:
#         plt.savefig(path, bbox_inches='tight')
    
#     plt.tight_layout()
#     plt.show()


# plot channels (give ch list)
