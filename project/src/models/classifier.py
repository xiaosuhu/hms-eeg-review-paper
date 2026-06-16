import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .backbones import TemporalCNN1D, ResNet1D, ResNet1DGRU, SmallCNN2D, EEGNet
from .heads import MLPHead, FusionConcatHead


# -------- Single-modality wrappers --------
class EEGClassifier(nn.Module):
    """
    Expects EEG input (B, C, T). Returns logits (B, num_classes).
    """
    def __init__(self, num_classes: int, backbone: str = "temporal_cnn", feat_dim: int = 256, in_channels: int = 20, backbone_dropout: float = 0.0, head_dropout: float = 0.1):
        super().__init__()
        if backbone == "temporal_cnn":
            self.backbone = TemporalCNN1D(in_channels=in_channels, feat_dim=feat_dim, dropout=backbone_dropout)
        elif backbone == "resnet1d":
            self.backbone = ResNet1D(in_channels=in_channels, feat_dim=feat_dim)
        elif backbone == "eegnet":
            self.backbone = EEGNet(in_channels=in_channels, dropout=backbone_dropout, feat_dim=feat_dim)
        elif backbone == "resnet1d_gru":
            self.backbone = ResNet1DGRU(in_channels=in_channels, feat_dim=feat_dim)
        else:
            raise ValueError(f"Unknown EEG backbone: {backbone}")
        self.head = MLPHead(in_dim=feat_dim, num_classes=num_classes, dropout=head_dropout)

    def forward(self, x_eeg):
        # x_eeg: (B,C,T)
        z = self.backbone(x_eeg)      # (B,feat_dim)
        logits = self.head(z)         # (B,num_classes)
        return logits


class SpecClassifier(nn.Module):
    """
    Expects spectrogram input (B, C, H, W). Returns logits (B, num_classes).
    If your DataLoader yields (B,C,W,H), permute before calling or inside forward.
    """
    def __init__(self, num_classes: int, backbone: str = "smallcnn2d", feat_dim: int = 256, in_channels: int = 3, input_is_BCWH: bool = False):
        super().__init__()
        if backbone == "smallcnn2d":
            self.backbone = SmallCNN2D(in_channels=in_channels, feat_dim=feat_dim)
        else:
            raise ValueError(f"Unknown Spec backbone: {backbone}")
        self.head = MLPHead(in_dim=feat_dim, num_classes=num_classes)
        self.input_is_BCWH = input_is_BCWH  # True if your input is (B,C,W,H)

    def forward(self, x_spec):
        # Ensure (B,C,H,W)
        if self.input_is_BCWH:
            x_spec = x_spec.permute(0, 1, 3, 2)  # (B,C,W,H) -> (B,C,H,W)
        z = self.backbone(x_spec)    # (B,feat_dim)
        logits = self.head(z)        # (B,num_classes)
        return logits


# -------- Multimodal (EEG + Spectrogram) --------
class MultiModalClassifier(nn.Module):
    """
    Fuses EEG (B,C,T) and Spectrogram (B,C,H,W) embeddings via concatenation.
    Returns logits (B, num_classes).
    """
    def __init__(
        self,
        num_classes: int,
        eeg_backbone: str = "temporal_cnn",
        spec_backbone: str = "smallcnn2d",
        eeg_feat_dim: int = 256,
        spec_feat_dim: int = 256,
        eeg_in_channels: int = 20,
        spec_in_channels: int = 3,
        spec_input_is_BCWH: bool = False,
    ):
        super().__init__()
        # EEG branch
        if eeg_backbone == "temporal_cnn":
            self.eeg_backbone = TemporalCNN1D(in_channels=eeg_in_channels, feat_dim=eeg_feat_dim)
        elif eeg_backbone == "resnet1d":
            self.eeg_backbone = ResNet1D(in_channels=eeg_in_channels, feat_dim=eeg_feat_dim)
        else:
            raise ValueError(f"Unknown EEG backbone: {eeg_backbone}")

        # Spec branch
        if spec_backbone == "smallcnn2d":
            self.spec_backbone = SmallCNN2D(in_channels=spec_in_channels, feat_dim=spec_feat_dim)
        else:
            raise ValueError(f"Unknown Spec backbone: {spec_backbone}")

        # Fusion head
        self.fusion_head = FusionConcatHead(d1=eeg_feat_dim, d2=spec_feat_dim, num_classes=num_classes)
        self.spec_input_is_BCWH = spec_input_is_BCWH

    def forward(self, x_eeg, x_spec):
        # x_eeg:  (B,C,T)
        # x_spec: (B,C,H,W) or (B,C,W,H) if spec_input_is_BCWH=True
        if self.spec_input_is_BCWH:
            x_spec = x_spec.permute(0,1,3,2)  # (B,C,W,H) -> (B,C,H,W)
        z_eeg  = self.eeg_backbone(x_eeg)    # (B,de)
        z_spec = self.spec_backbone(x_spec)  # (B,ds)
        logits = self.fusion_head(z_eeg, z_spec)  # (B,num_classes)
        return logits

    
# EEGFeatureExtractor returns [B, 64], and you're using 4 chain pairs
class EEGMultiChannelWrapper(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super().__init__()
        self.fe = feature_extractor
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):  # x: [B, 8, 2000]
        x = x.permute(0, 2, 1)  # [B, C=8, T]
        z1 = (self.fe(x[:, 0:1, :]) + self.fe(x[:, 1:2, :])) / 2
        z2 = (self.fe(x[:, 2:3, :]) + self.fe(x[:, 3:4, :])) / 2
        z3 = (self.fe(x[:, 4:5, :]) + self.fe(x[:, 5:6, :])) / 2
        z4 = (self.fe(x[:, 6:7, :]) + self.fe(x[:, 7:8, :])) / 2
        y = torch.cat([z1, z2, z3, z4], dim=1)  # [B, 64*4]
        y = F.relu(self.fc1(y))
        return F.softmax(self.fc2(y), dim=1)


def build_model(cfg: dict) -> nn.Module:
    name = cfg["name"]
    num_classes = cfg["num_classes"]
    in_channels = cfg.get("in_channels", 20)
    feat_dim = cfg.get("feat_dim", 256)
    backbone_dropout = cfg.get("dropout_backbone", 0.0)
    head_dropout = cfg.get("dropout_head", 0.1)

    if name == "eeg_1d_small":
        return EEGClassifier(num_classes=num_classes, backbone="temporal_cnn",
                             feat_dim=128, in_channels=in_channels,
                             backbone_dropout=backbone_dropout, head_dropout=head_dropout)
    elif name == "eeg_1d":
        return EEGClassifier(num_classes=num_classes, backbone="temporal_cnn",
                             feat_dim=feat_dim, in_channels=in_channels,
                             backbone_dropout=backbone_dropout, head_dropout=head_dropout)
    elif name == "eeg_1d_resnet":
        return EEGClassifier(num_classes=num_classes, backbone="resnet1d",
                             feat_dim=feat_dim, in_channels=in_channels,
                             backbone_dropout=backbone_dropout, head_dropout=head_dropout)
    elif name == "eeg_1d_resnet_gru":
        return EEGClassifier(num_classes=num_classes, backbone="resnet1d_gru",
                             feat_dim=feat_dim, in_channels=in_channels,
                             backbone_dropout=backbone_dropout, head_dropout=head_dropout)
    elif name == "spec_2d":
        return SpecClassifier(num_classes=num_classes, backbone="smallcnn2d",
                              feat_dim=feat_dim, in_channels=cfg.get("spec_in_channels", 3))
    elif name == "eeg_1d_eegnet":
        return EEGClassifier(num_classes=num_classes, backbone="eegnet",
                             feat_dim=128, in_channels=in_channels,
                             backbone_dropout=backbone_dropout, head_dropout=head_dropout)
    elif name == "multimodal":
        return MultiModalClassifier(num_classes=num_classes,
                                    eeg_feat_dim=feat_dim, spec_feat_dim=feat_dim,
                                    eeg_in_channels=in_channels)
    else:
        raise ValueError(f"Unknown model name: {name!r}")