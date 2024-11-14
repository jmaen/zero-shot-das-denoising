import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.utils as utils


def load_image(path):
    img = Image.open(path)
    transform = transforms.ToTensor()
    img = transform(img)
    return img.unsqueeze(0)


def save_image(img, path):
    utils.save_image(img, path)


def get_noisy_image(img, std=0.1):
    noise = torch.randn(img.size()) * std
    img = img + noise
    return torch.clamp(img, 0, 1)


def plot_row(images, labels=[], path=''):
    _, axes = plt.subplots(1, len(images), figsize=(10, 10))

    for i, img in enumerate(images):
        img = torch.squeeze(img)

        if img.min() < 0 or img.max() > 1:
            img = (img - img.min()) / (img.max() - img.min())
        
        img = img.permute(1, 2, 0).cpu().numpy()
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        if labels and i < len(labels):
            axes[i].set_title(labels[i], fontsize=10)

    if path:
        plt.savefig(path, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
