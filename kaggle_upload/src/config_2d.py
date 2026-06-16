"""
config_2d.py — Experiment configuration for the 2D spectrogram CNN pipeline.

Paired with dataset_2d.py and model_2d.py.
Inline this block into a Kaggle notebook by copy-pasting the class definition.
"""
from dataclasses import dataclass, field
import torch


@dataclass
class Config2D:
    # ── Data paths ─────────────────────────────────────────────────────────
    spectrogram_dir: str = "train_spectrograms"   # dir containing {spec_id}.parquet
    metadata_csv: str    = "data_meta_splits/train_test_split.csv"

    # ── Spectrogram window ─────────────────────────────────────────────────
    spec_window_seconds: int = 300   # seconds to extract around the label offset
    img_height: int          = 128   # resize target H (freq axis)
    img_width:  int          = 256   # resize target W (time axis)

    # ── Model ──────────────────────────────────────────────────────────────
    backbone:    str  = "efficientnet_b0"   # any timm model
    pretrained:  bool = True
    num_classes: int  = 6

    # ── Training ───────────────────────────────────────────────────────────
    batch_size:   int   = 32
    num_epochs:   int   = 50
    lr:           float = 1e-3
    weight_decay: float = 1e-4

    # ── Early stopping ─────────────────────────────────────────────────────
    patience: int = 10
    monitor:  str = "macro_f1"   # "macro_f1" | "val_kl"

    # ── Misc ───────────────────────────────────────────────────────────────
    seed:        int = 42
    num_workers: int = 4
    val_fold:    int = 4          # inner_fold == val_fold → validation set

    # set automatically in __post_init__
    device: str = field(init=False)

    def __post_init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def as_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
