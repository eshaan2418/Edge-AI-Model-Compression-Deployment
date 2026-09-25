from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import get_cifar10_datasets
from .models import create_model


@dataclass
class TrainConfig:
    data_dir: str = "data"
    batch_size: int = 128
    num_epochs: int = 1
    lr: float = 0.1
    weight_decay: float = 5e-4
    num_workers: int = 2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Cap batches per train/eval loop for fast smoke tests (None = full dataset).
    max_batches: int | None = None
    ckpt_path: str = "models/baseline_resnet18.pt"


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    for images, targets in tqdm(loader, desc="train", leave=False):
        if max_batches is not None and n >= max_batches:
            break
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(logits.detach(), targets)
        n += 1

    n = max(n, 1)
    return running_loss / n, running_acc / n


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    n = 0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval", leave=False):
            if max_batches is not None and n >= max_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            running_loss += loss.item()
            running_acc += accuracy(logits, targets)
            n += 1
    n = max(n, 1)
    return running_loss / n, running_acc / n


def main(cfg: TrainConfig | None = None) -> None:
    cfg = cfg or TrainConfig()
    device = cfg.device
    train_ds, test_ds = get_cifar10_datasets(cfg.data_dir)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.num_epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, cfg.max_batches
        )
        te_loss, te_acc = evaluate(model, test_loader, device, cfg.max_batches)
        print(
            f"epoch={epoch + 1}/{cfg.num_epochs} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={te_loss:.4f} val_acc={te_acc:.4f}"
        )

    # Save a checkpoint
    ckpt_path = Path(cfg.ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


def build_config_from_args(argv: list[str] | None = None) -> TrainConfig:
    p = argparse.ArgumentParser(description="Train the PyTorch CIFAR-10 baseline (ResNet-18).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Cap batches per train/eval loop for a fast smoke test.",
    )
    p.add_argument("--ckpt-path", default="models/baseline_resnet18.pt")
    args = p.parse_args(argv)
    return TrainConfig(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        device=args.device,
        max_batches=args.max_batches,
        ckpt_path=args.ckpt_path,
    )


def cli_main(argv: list[str] | None = None) -> None:
    main(build_config_from_args(argv))


if __name__ == "__main__":
    cli_main()
