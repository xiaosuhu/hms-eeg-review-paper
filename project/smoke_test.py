"""
Smoke test for EEGDataset.
Run from the HMS_EEG/ project root:
    python project/smoke_test.py
"""
import sys
from pathlib import Path

# Add HMS_EEG/ to sys.path so 'project.src.*' imports resolve
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
import torch
from torch.utils.data import DataLoader

from project.src.data import EEGDataset


def main():
    cfg_path = ROOT / "project" / "config" / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cpu")
    print(f"device        : {device}")

    meta_path = ROOT / cfg["data"]["train_meta"]
    eeg_dir   = ROOT / cfg["data"]["eeg_dir"]

    train_ds = EEGDataset(str(meta_path), str(eeg_dir), split="train")
    val_ds   = EEGDataset(str(meta_path), str(eeg_dir), split="val")
    test_ds  = EEGDataset(str(meta_path), str(eeg_dir), split="test")

    print(f"train samples : {len(train_ds)}")
    print(f"val   samples : {len(val_ds)}")
    print(f"test  samples : {len(test_ds)}")

    loader = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    samples, labels = next(iter(loader))
    samples = samples.to(device)

    print(f"batch x shape : {tuple(samples.shape)}")
    print(f"batch y       : {labels.tolist()}")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
