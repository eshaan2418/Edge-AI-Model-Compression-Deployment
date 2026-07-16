from __future__ import annotations

import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

# Per-dataset normalization statistics. "fake" reuses CIFAR-10 stats so the
# synthetic smoke dataset is shaped and normalized exactly like CIFAR-10.
_STATS = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tiny_imagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    "fake": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

# Number of image classes per supported dataset.
_NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "tiny_imagenet": 200, "fake": 10}

# Default number of synthetic samples per split when no ``limit_samples`` given.
_FAKE_DEFAULT_SAMPLES = 128


def _normalize_name(dataset: str) -> str:
    n = dataset.lower().replace("-", "_")
    if n in ("tiny_imagenet_200", "tinyimagenet"):
        return "tiny_imagenet"
    if n in ("synthetic", "debug", "random"):
        return "fake"
    return n


def _transforms(dataset: str, crop_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    mean, std = _STATS[dataset]
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tf, test_tf


def build_loaders(
    dataset: str,
    data_dir: str,
    *,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
    limit_samples: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test ``DataLoader``s for a supported dataset.

    Supported datasets: ``cifar10``, ``cifar100``, ``tiny_imagenet``. Tiny
    ImageNet is not bundled with torchvision; it is expected to already be laid
    out as ``{data_dir}/tiny-imagenet-200/{train,val}`` in ``ImageFolder`` form.

    The ``fake`` dataset (aliases ``synthetic`` / ``debug`` / ``random``)
    generates random CIFAR-shaped tensors (``[3, 32, 32]``, 10 classes) with no
    files or network access — used for fast, fully offline smoke tests.

    ``limit_samples`` restricts each split to its first N examples (a fast
    smoke-test knob); ``None`` uses the full dataset.
    """
    name = _normalize_name(dataset)
    if name not in _STATS:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {sorted(_STATS)}")

    if name == "fake":
        train_tf, test_tf = _transforms(name, crop_size=32)
        n = limit_samples if limit_samples is not None else _FAKE_DEFAULT_SAMPLES
        train_ds = torchvision.datasets.FakeData(
            size=n, image_size=(3, 32, 32), num_classes=10, transform=train_tf
        )
        test_ds = torchvision.datasets.FakeData(
            size=n, image_size=(3, 32, 32), num_classes=10, transform=test_tf
        )
    elif name == "tiny_imagenet":
        train_tf, test_tf = _transforms(name, crop_size=64)
        import os

        root = os.path.join(data_dir, "tiny-imagenet-200")
        train_ds = torchvision.datasets.ImageFolder(os.path.join(root, "train"), transform=train_tf)
        test_ds = torchvision.datasets.ImageFolder(os.path.join(root, "val"), transform=test_tf)
    else:
        train_tf, test_tf = _transforms(name, crop_size=32)
        ds_cls = (
            torchvision.datasets.CIFAR10 if name == "cifar10" else torchvision.datasets.CIFAR100
        )
        train_ds = ds_cls(root=data_dir, train=True, download=download, transform=train_tf)
        test_ds = ds_cls(root=data_dir, train=False, download=download, transform=test_tf)

    if limit_samples is not None:
        train_ds = Subset(train_ds, range(min(limit_samples, len(train_ds))))
        test_ds = Subset(test_ds, range(min(limit_samples, len(test_ds))))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    return train_loader, test_loader


def build_cifar10_loaders(
    data_dir: str,
    *,
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Convenience wrapper for CIFAR-10 train/test loaders."""
    return build_loaders(
        "cifar10",
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        download=download,
    )


def num_classes_for(dataset: str) -> int:
    """Return the number of classes for a supported dataset name."""
    return _NUM_CLASSES[_normalize_name(dataset)]
