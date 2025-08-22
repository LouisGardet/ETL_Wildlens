import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parent / "src"))

import torch
import torchvision
import matplotlib.pyplot as plt


def denormalize(img_tensor):
    """
    Applique la dénormalisation pour une image normalisée entre [-1, 1]
    afin de la ramener dans l'espace RGB visible [0, 1].
    """
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    return img_tensor * std + mean


def show_batch(dataloader, n=8):
    """
    Affiche les n premières images d’un batch extrait d’un DataLoader PyTorch.

    Args:
        dataloader (torch.utils.data.DataLoader): le dataloader contenant les images.
        n (int): nombre d’images à afficher (par défaut 8).
    """
    images, labels = next(iter(dataloader))

    # Appliquer la dénormalisation
    images = denormalize(images)

    # Afficher avec torchvision
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(torchvision.utils.make_grid(images[:n]).permute(1, 2, 0))
    plt.axis('off')
    plt.show()