# src/evaluation.py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

@torch.no_grad()
def compute_metrics(probs: torch.Tensor, y_true: torch.Tensor, num_classes: int):
    """
    probs: [N, C] softmaxed predictions
    y_true: [N] int class ids
    """
    preds = probs.argmax(dim=1)
    correct = (preds == y_true).sum().item()
    acc = correct / max(1, y_true.numel())

    # macro F1 + per-class F1
    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (y_true == c)).sum().item()
        fp = ((preds == c) & (y_true != c)).sum().item()
        fn = ((preds != c) & (y_true == c)).sum().item()
        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s)) if len(f1s) else 0.0

    out = {"accuracy": acc, "macro_f1": macro_f1}
    for c, f1 in enumerate(f1s):
        out[f"f1_{c}"] = float(f1)
    return out

def _one_hot(labels: torch.Tensor, num_classes: int, smoothing: float = 0.0):
    with torch.no_grad():
        y = torch.zeros((labels.size(0), num_classes), device=labels.device)
        y.scatter_(1, labels.unsqueeze(1), 1.0)
        if smoothing > 0:
            y = (1 - smoothing) * y + smoothing / num_classes
    return y

@torch.no_grad()
def validate(model: nn.Module,
             loader: torch.utils.data.DataLoader,
             loss_fn: nn.Module,
             cfg: dict,
             device: torch.device):
    model.eval()
    total_loss = 0.0
    all_probs, all_y_true = [], []

    use_kl = cfg["loss"].get("name", "ce").lower() == "kl"
    num_classes = cfg["model"]["num_classes"]

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        logits = model(x)

        if use_kl:
            # KL expects probs as targets; accept either ints or probs from dataset
            if y.ndim == 1:
                y_probs = _one_hot(y.long(), num_classes, cfg["loss"].get("label_smoothing", 0.0))
            else:
                y_probs = y
            loss = loss_fn(logits, y_probs)
            probs = torch.softmax(logits, dim=1)
            y_true = y.argmax(dim=1) if y.ndim == 2 else y.long()
        else:
            loss = loss_fn(logits, y.long())
            probs = torch.softmax(logits, dim=1)
            y_true = y.long()

        total_loss += loss.item() * x.size(0)
        all_probs.append(probs.cpu())
        all_y_true.append(y_true.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    all_y_true = torch.cat(all_y_true, dim=0)
    metrics = compute_metrics(all_probs, all_y_true, num_classes)
    val_loss = total_loss / len(loader.dataset)
    return val_loss, metrics, all_probs, all_y_true  # return probs/labels for reports
