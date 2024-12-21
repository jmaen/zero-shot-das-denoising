import os
import textwrap
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as utils


def load_image(path):
    img = Image.open(path)
    transform = transforms.ToTensor()
    img = transform(img)
    return img.unsqueeze(0)


def load_images(dir, transform=None):
    images = []
    for file in os.listdir(dir):
        image = load_image(os.path.join(dir, file))

        if transform is not None:
            image = transform(image)

        images.append(image)

    return torch.cat(images)


def load_celeba(num_samples=1):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = torchvision.datasets.CelebA(root='./data/', download=True, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples, shuffle=True)

    return next(iter(data_loader))[0]


def save_image(img, path):
    utils.save_image(img, path)


def get_noisy_image(img, psnr=20):
    mse = 1 / (10 ** (psnr / 10))
    std = np.sqrt(mse)

    noise = torch.randn(img.size()) * std
    img = img + noise
    return torch.clamp(img, 0, 1)


def plot_row(images, labels=[], path=''):
    cols = len(images)
    _, axes = plt.subplots(1, cols, figsize=(cols * 4, 8))

    for i, img in enumerate(images):
        img = torch.squeeze(img)

        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        
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
