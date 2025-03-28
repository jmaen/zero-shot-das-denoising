import os
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils


def load_image(path, size=256):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    img = transform(img)
    return img.unsqueeze(0)


def load_images(dir, size=256):
    images = []
    for file in sorted(os.listdir(dir)):
        image = load_image(os.path.join(dir, file), size)
        images.append(image)

    return torch.cat(images)


def load_celeba(num_samples=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = torchvision.datasets.CelebA(root='./data/image/', download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)

    return next(iter(data_loader))[0]


def save_image(img, path):
    if img.min() < 0 or img.max() > 1:
        # img = (img - img.min()) / (img.max() - img.min())
        img = img / 2 + 0.5
        img = img.clamp(0, 1)

    utils.save_image(img, path)


def get_noisy_image(img, psnr=20, max=2):
    mse = max**2 / (10 ** (psnr / 10))
    std = np.sqrt(mse)

    noise = torch.randn(img.size()) * std
    img = img + noise
    # img = torch.clamp(img, -1, 1)
    return img


def plot_row(images, labels=[], path=''):
    cols = len(images)
    _, axes = plt.subplots(1, cols, figsize=(cols * 4, 8))

    for i, img in enumerate(images):
        img = torch.squeeze(img)

        if img.min() < 0 or img.max() > 1:
            # img = (img - img.min()) / (img.max() - img.min())
            img = img / 2 + 0.5
            img = img.clamp(0, 1)
        
        img = img.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if labels and i < len(labels):
            label = "\n".join(textwrap.wrap(labels[i], 22))
            axes[i].set_title(label, fontsize=10)

    if path:
        plt.savefig(path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
