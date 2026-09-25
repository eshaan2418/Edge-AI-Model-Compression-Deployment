from __future__ import annotations

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


def get_cifar10_datasets(data_dir: str) -> tuple[Dataset, Dataset]:
    """Return CIFAR-10 train and test datasets with standard transforms."""
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    return train, test
