"""
model_2d.py — EfficientNet-based classifier for HMS 4-channel spectrograms.

Loads any timm backbone with in_chans=4 and replaces the classifier head
with a linear layer outputting raw logits for 6 classes.

Usage
-----
    from model_2d import EfficientNetEEG
    model = EfficientNetEEG(backbone="efficientnet_b0", num_classes=6, pretrained=True)
    logits = model(batch_images)   # (B, 6)
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError as e:
    raise ImportError("timm is required: pip install timm") from e


class EfficientNetEEG(nn.Module):
    """
    Timm backbone adapted for 4-channel spectrogram input.

    The first conv layer is rebuilt to accept `in_chans=4` instead of 3.
    When `pretrained=True`, RGB weights are averaged across the channel dimension
    and replicated to initialise the 4th channel (timm handles this internally
    via the `in_chans` argument).

    Parameters
    ----------
    backbone : str
        Any timm model name (e.g. "efficientnet_b0", "efficientnet_b2").
    num_classes : int
        Number of output logits (6 for HMS competition).
    pretrained : bool
        Load ImageNet weights via timm.
    drop_rate : float
        Dropout rate applied before the classifier head.
    """

    def __init__(
        self,
        backbone:    str  = "efficientnet_b0",
        num_classes: int  = 6,
        pretrained:  bool = True,
        drop_rate:   float = 0.3,
    ):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=4,       # 4 spectrogram chains instead of RGB
            num_classes=0,    # remove timm's default head; we add our own
            drop_rate=drop_rate,
        )
        n_features = self.encoder.num_features
        self.head = nn.Linear(n_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : FloatTensor of shape (B, 4, H, W)

        Returns
        -------
        logits : FloatTensor of shape (B, num_classes)  — no softmax applied
        """
        features = self.encoder(x)   # (B, n_features)
        return self.head(features)   # (B, num_classes)


def build_model_2d(cfg) -> nn.Module:
    """Construct model from a Config2D instance."""
    return EfficientNetEEG(
        backbone=cfg.backbone,
        num_classes=cfg.num_classes,
        pretrained=cfg.pretrained,
    )
