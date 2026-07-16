from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def confusion_matrix_from_preds(
    logits_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for lg, tg in zip(logits_list, targets_list, strict=False):
        pred = lg.argmax(dim=1).cpu().numpy()
        t = tg.cpu().numpy()
        for p, y in zip(pred.tolist(), t.tolist(), strict=False):
            cm[y, p] += 1
    return cm


def per_class_accuracy(cm: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    c = cm.shape[0]
    for i in range(c):
        denom = cm[i].sum()
        out[f"class_{i}"] = float(cm[i, i] / denom) if denom > 0 else 0.0
    return out


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE for multi-class: max predicted prob as confidence."""
    conf = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += np.abs(acc[m].mean() - conf[m].mean()) * (m.mean())
    return float(ece)


def negative_log_likelihood(probs: np.ndarray, labels: np.ndarray) -> float:
    p = np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    model.eval()
    logits_chunks: list[torch.Tensor] = []
    targets_chunks: list[torch.Tensor] = []
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits_chunks.append(model(x).cpu())
        targets_chunks.append(y.cpu())
        if max_batches is not None and i + 1 >= max_batches:
            break
    return logits_chunks, targets_chunks


def failure_shift_metrics(cm_baseline: np.ndarray, cm_compressed: np.ndarray) -> dict[str, Any]:
    """Aggregate how confusion mass moves between baseline and compressed."""
    b_diag = np.diag(cm_baseline).sum()
    c_diag = np.diag(cm_compressed).sum()
    return {
        "baseline_correct_total": int(b_diag),
        "compressed_correct_total": int(c_diag),
        "off_diagonal_mass_baseline": float(cm_baseline.sum() - b_diag),
        "off_diagonal_mass_compressed": float(cm_compressed.sum() - c_diag),
    }


def build_failure_cases(
    baseline_logits: list[torch.Tensor],
    compressed_logits: list[torch.Tensor],
    targets: list[torch.Tensor],
    *,
    max_items: int = 500,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    idx_global = 0
    for lb, lc, yb in zip(baseline_logits, compressed_logits, targets, strict=True):
        pb = F.softmax(lb, dim=1)
        pc = F.softmax(lc, dim=1)
        b_pred = lb.argmax(dim=1)
        c_pred = lc.argmax(dim=1)
        conf_b = pb.max(dim=1).values
        conf_c = pc.max(dim=1).values
        for j in range(yb.shape[0]):
            if len(cases) >= max_items:
                return {"cases": cases, "truncated": True}
            y = int(yb[j].item())
            bp = int(b_pred[j].item())
            cp = int(c_pred[j].item())
            if bp == y and cp != y:
                kind = "baseline_right_compressed_wrong"
            elif bp != y and cp == y:
                kind = "compressed_right_baseline_wrong"
            else:
                kind = "other"
            cases.append(
                {
                    "index": idx_global,
                    "true": y,
                    "baseline_pred": bp,
                    "compressed_pred": cp,
                    "baseline_conf": float(conf_b[j].item()),
                    "compressed_conf": float(conf_c[j].item()),
                    "kind": kind,
                }
            )
            idx_global += 1
    return {"cases": cases, "truncated": False}


def full_diagnostic_report(
    baseline: nn.Module,
    compressed: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    *,
    max_batches: int | None = 20,
) -> dict[str, Any]:
    bl, tl = collect_predictions(baseline, loader, device, max_batches=max_batches)
    cl, _ = collect_predictions(compressed, loader, device, max_batches=max_batches)
    logits_b = torch.cat(bl, dim=0)
    logits_c = torch.cat(cl, dim=0)
    targets = torch.cat(tl, dim=0)
    y_np = targets.numpy()
    pb = F.softmax(logits_b, dim=1).numpy()
    pc = F.softmax(logits_c, dim=1).numpy()

    cm_b = confusion_matrix_from_preds(bl, tl, num_classes)
    cm_c = confusion_matrix_from_preds(cl, tl, num_classes)

    report: dict[str, Any] = {
        "confusion_matrix_baseline": cm_b.tolist(),
        "confusion_matrix_compressed": cm_c.tolist(),
        "per_class_accuracy_baseline": per_class_accuracy(cm_b),
        "per_class_accuracy_compressed": per_class_accuracy(cm_c),
        "ece_baseline": expected_calibration_error(pb, y_np),
        "ece_compressed": expected_calibration_error(pc, y_np),
        "nll_baseline": negative_log_likelihood(pb, y_np),
        "nll_compressed": negative_log_likelihood(pc, y_np),
        "failure_shift": failure_shift_metrics(cm_b, cm_c),
        "failure_cases": build_failure_cases(bl, cl, tl, max_items=500),
    }
    return report
