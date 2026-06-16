# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class KLLoss(nn.Module):
    """KL(p_target || q_pred) with logits input and prob target."""
    def __init__(self, reduction="batchmean"):
        super().__init__()
        self.kldiv = nn.KLDivLoss(reduction=reduction)
    def forward(self, logits, target_probs):
        log_q = F.log_softmax(logits, dim=1)
        return self.kldiv(log_q, target_probs)

def build_loss(cfg: dict) -> nn.Module:
    name = cfg["loss"].get("name", "ce").lower()
    if name == "kl":
        return KLLoss(reduction="batchmean")
    elif name == "ce":
        weight = cfg["loss"].get("class_weight", None)
        weight = torch.tensor(weight) if weight else None
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Unsupported loss: {name}")
