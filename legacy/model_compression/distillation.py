from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import get_cifar10_datasets
from .models import create_model


@dataclass
class DistillConfig:
    data_dir: str = "data"
    batch_size: int = 128
    num_epochs: int = 1
    lr: float = 0.05
    weight_decay: float = 5e-4
    temperature: float = 4.0
    alpha: float = 0.5  # weight for distillation vs CE
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    teacher_ckpt: str = "models/baseline_resnet18.pt"


def kd_loss_fn(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    ce = nn.functional.cross_entropy(student_logits, targets)
    log_p = nn.functional.log_softmax(student_logits / temperature, dim=1)
    q = nn.functional.softmax(teacher_logits / temperature, dim=1)
    kd = nn.functional.kl_div(log_p, q, reduction="batchmean") * (temperature**2)
    return alpha * kd + (1 - alpha) * ce


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(dim=1)
            running_loss += loss.item()
            running_acc += (preds == targets).float().mean().item()
    n = len(loader)
    return running_loss / n, running_acc / n


def run_distillation(cfg: DistillConfig | None = None) -> None:
    cfg = cfg or DistillConfig()
    device = cfg.device
    train_ds, test_ds = get_cifar10_datasets(cfg.data_dir)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    teacher = create_model().to(device)
    ckpt = torch.load(cfg.teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    student = create_model().to(device)
    optimizer = optim.SGD(
        student.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay
    )

    for epoch in range(cfg.num_epochs):
        student.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"distill e{epoch + 1}", leave=False):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                t_logits = teacher(images)

            s_logits = student(images)
            loss = kd_loss_fn(s_logits, t_logits, targets, cfg.temperature, cfg.alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        tr_loss = running_loss / len(train_loader)
        te_loss, te_acc = evaluate(student, test_loader, device)
        print(
            f"epoch={epoch + 1}/{cfg.num_epochs} train_loss={tr_loss:.4f} "
            f"val_loss={te_loss:.4f} val_acc={te_acc:.4f}"
        )

    torch.save({"model_state": student.state_dict()}, "models/student_kd.pt")
    print("Saved student model to models/student_kd.pt")
