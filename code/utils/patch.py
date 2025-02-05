import torch
import math
import numpy as np
import scipy.stats as stats


def patch(x, patch_size=16, stride=4):
    _, _, W, H = x.shape

    patches = []
    for i in range(0, W - patch_size + 1, stride):  
        for j in range(0, H - patch_size + 1, stride):
            patch = x[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)

    return patches


def unpatch(patches, patch_size=16, stride=4):
    N, C, _, _ = patches.shape

    num_patches_w = math.isqrt(N)
    num_patches_h = N // num_patches_w
    W = stride*(num_patches_w - 1) + patch_size
    H = stride*(num_patches_h - 1) + patch_size

    x = torch.zeros((1, C, W, H), device=patches.device)
    overlap_map = torch.zeros((1, C, W, H), device=patches.device)  # count overlaps for mean calculation
    k = 0
    for i in range(0, W - patch_size + 1, stride):
        for j in range(0, H - patch_size + 1, stride):
            x[:, :, i:i+patch_size, j:j+patch_size] += patches[k]
            overlap_map[:, :, i:i+patch_size, j:j+patch_size] += 1
            k += 1

    x /= overlap_map
    return x


def filter_patches(patches, alpha):
    flattened_patches = patches.flatten(start_dim=1)

    kurtosis = stats.kurtosis(flattened_patches.cpu(), axis=1)
    threshold = np.percentile(kurtosis, alpha*100)
    
    selected_columns = kurtosis > threshold
    patches = patches[selected_columns, :]

    return patches
