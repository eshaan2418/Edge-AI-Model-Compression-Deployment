from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from edge_ai_compression.core.registry import ModelRegistry
from edge_ai_compression.data.loaders import build_cifar10_loaders


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    ce = F.cross_entropy(student_logits, targets)
    log_p = F.log_softmax(student_logits / temperature, dim=1)
    q = F.softmax(teacher_logits / temperature, dim=1)
    kd = F.kl_div(log_p, q, reduction="batchmean") * (temperature**2)
    return alpha * kd + (1 - alpha) * ce


@torch.no_grad()
def _eval_accuracy(model: nn.Module, test_loader, device: str) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_acc += (preds == targets).float().mean().item()
        n += 1
    return total_loss / max(n, 1), total_acc / max(n, 1)


def distill_student_inplace(
    student: nn.Module,
    teacher_model: str,
    teacher_ckpt: str,
    data_dir: str,
    batch_size: int,
    device: str,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
) -> nn.Module:
    train_loader, test_loader = build_cifar10_loaders(
        data_dir, batch_size=batch_size, num_workers=2
    )

    teacher = ModelRegistry.create(teacher_model).to(device)
    ckpt = torch.load(teacher_ckpt, map_location=device)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    student = student.to(device)
    optimizer = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        student.train()
        running = 0.0
        for images, targets in tqdm(train_loader, desc=f"distill {epoch+1}/{epochs}", leave=False):
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                t_logits = teacher(images)
            s_logits = student(images)
            loss = kd_loss(s_logits, t_logits, targets, temperature, alpha)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        tr_loss = running / len(train_loader)
        te_loss, te_acc = _eval_accuracy(student, test_loader, device)
        print(
            f"distill epoch={epoch+1}/{epochs} train_loss={tr_loss:.4f} "
            f"val_loss={te_loss:.4f} val_acc={te_acc:.4f}"
        )

    return student
