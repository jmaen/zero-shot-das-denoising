import torch
import torch.nn.functional as F
import numpy as np
import scipy.stats as stats


def patch(x, patch_size=16, stride=4):
    _, _, W, H = x.shape
    pad_w = (patch_size - W % patch_size) % patch_size
    pad_h = (patch_size - H % patch_size) % patch_size

    x = F.pad(x, (0, pad_h, 0, pad_w), mode="constant", value=0)

    _, _, W, H = x.shape

    patches = []
    for i in range(0, W - patch_size + 1, stride):  
        for j in range(0, H - patch_size + 1, stride):
            patch = x[:, :, i:i+patch_size, j:j+patch_size]
            patches.append(patch)
    patches = torch.cat(patches, dim=0)

    return patches


def unpatch(patches, shape, patch_size=16, stride=4):
    C = patches.shape[1]

    # num_patches_w = math.isqrt(N)
    # num_patches_h = N // num_patches_w
    # W = stride*(num_patches_w - 1) + patch_size
    # H = stride*(num_patches_h - 1) + patch_size

    W, H = shape

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


# https://github.com/cuiyang512/Unsupervised-DAS-Denoising
def yc_patch_inv(X1, n1, n2, l1=24, l2=24, o1=6, o2=6):
    tmp1 = np.mod(n1 - l1, o1)
    tmp2 = np.mod(n2 - l2, o2)
    if (tmp1 != 0) and (tmp2 != 0):
        A = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))
        mask = np.zeros((n1 + o1 - tmp1, n2 + o2 - tmp2))

    if (tmp1 != 0) and (tmp2 == 0):
        A = np.zeros((n1 + o1 - tmp1, n2))
        mask = np.zeros((n1 + o1 - tmp1, n2))

    if (tmp1 == 0) and (tmp2 != 0):
        A = np.zeros((n1, n2 + o2 - tmp2))
        mask = np.zeros((n1, n2 + o2 - tmp2))

    if (tmp1 == 0) and (tmp2 == 0):
        A = np.zeros((n1, n2))
        mask = np.zeros((n1, n2))

    N1, N2 = np.shape(A)
    ids = 0
    for i1 in range(0, N1 - l1 + 1, o1):
        for i2 in range(0, N2 - l2 + 1, o2):
            # print(i1,i2)
            #       [i1,i2,ids]
            A[i1:i1 + l1, i2:i2 + l2] = A[i1:i1 + l1, i2:i2 + l2] + np.reshape(X1[:, ids], (l1, l2))
            mask[i1:i1 + l1, i2:i2 + l2] = mask[i1:i1 + l1, i2:i2 + l2] + np.ones((l1, l2))
            ids = ids + 1

    A = A / mask;
    A = A[0:n1, 0:n2]
    return A