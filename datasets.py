from torch.utils.data import Dataset, Subset
from torchvision import datasets
import numpy as np


def split_dataset_into_train_and_val(dataset, random_seed=0):
    rng = np.random.default_rng(random_seed)
    val_size = int(len(dataset) * 0.1)
    val_indices = rng.choice(len(dataset), val_size, replace=False)
    val_indices_set = set(val_indices)
    train_indices = [i for i in range(len(dataset)) if i not in val_indices_set]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def get_dataset(dataset, split, random_seed):
    if split == 'test':
        return dataset

    train_dataset, val_dataset = split_dataset_into_train_and_val(dataset, random_seed=random_seed)
    return train_dataset if split == 'train' else val_dataset


class FashionMNIST(Dataset):
    def __init__(self, root_path, download=True, split='train', transform=None, random_seed=0):
        super().__init__()
        train = split in ['train', 'val']
        dataset = datasets.FashionMNIST(root_path, download=download, train=train, transform=transform)
        self.dataset = get_dataset(dataset, split, random_seed)

        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        return img, target, idx

    def __len__(self):
        return len(self.dataset)
