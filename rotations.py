import torch
from torch.utils.data import Dataset
import numpy as np


def rotate(img, label):
    """Rotate input image with 0, 90, 180, and 270 degrees.
    Args:
        img (Tensor): input image of shape (C, H, W).
    Returns:
        list[Tensor]: A list of four rotated images.
    """
    if label == 0:
        return img
    elif label == 1:
        return torch.flip(img.transpose(1, 2), [1])
    elif label == 2:
        return torch.flip(img, [1, 2])
    else:
        return torch.flip(img, [1]).transpose(1, 2)


class RotationPredDataset(Dataset):
    """Dataset for rotation prediction.
    """

    def __init__(self, dataset, random_seed=0):
        super().__init__()
        self.dataset = dataset
        self.rng = np.random.default_rng(random_seed)

    def __getitem__(self, idx):
        img, _, _ = self.dataset[idx]
        rot_label = self.rng.integers(0, high=4)
        img = rotate(img, rot_label)
        return img, rot_label, idx

    def __len__(self):
        return len(self.dataset)
