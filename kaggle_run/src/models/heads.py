# heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPHead(nn.Module):
    """
    Generic MLP head for a single feature vector per sample.
    Input:  (B, feat_dim)
    Output: (B, num_classes)  (logits)
    """
    def __init__(self, in_dim: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FusionConcatHead(nn.Module):
    """
    Concatenate two modality embeddings and classify.
    Inputs: (B, d1), (B, d2)
    Output: (B, num_classes)
    """
    def __init__(self, d1: int, d2: int, num_classes: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d1 + d2, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z1, z2):
        z = torch.cat([z1, z2], dim=1)
        return self.net(z)
