"""
dataset_2d.py — PyTorch Dataset for HMS spectrogram (2D CNN pipeline).

Each spectrogram parquet contains:
  - 'time'  : seconds elapsed (e.g. 0.0, 2.0, 4.0 …) — ~300 rows for a 600s file
  - LL_*    : Left-Lateral chain,  ~100 frequency bins
  - RL_*    : Right-Lateral chain, ~100 frequency bins
  - LP_*    : Left-Parasagittal chain,  ~100 frequency bins
  - RP_*    : Right-Parasagittal chain, ~100 frequency bins

__getitem__ returns:
  {
    "image"     : FloatTensor (4, img_height, img_width),  # log1p-normalised
    "soft_label": FloatTensor (6,),                        # vote proportions
    "label"     : int,                                     # argmax hard label
  }
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

VOTE_COLS  = ["seizure_vote", "lpd_vote", "gpd_vote",
              "lrda_vote",    "grda_vote", "other_vote"]
CHAINS     = ["LL", "RL", "LP", "RP"]


def _chain_cols(df: pd.DataFrame, chain: str) -> list[str]:
    """Return frequency-bin columns for one chain, sorted by frequency."""
    cols = [c for c in df.columns if c.startswith(f"{chain}_")]
    cols.sort(key=lambda c: float(c.split("_", 1)[1]))
    return cols


def _load_spec_window(
    parquet_path: str,
    offset_sec: float,
    window_sec: float,
) -> np.ndarray:
    """
    Load one spectrogram window.

    Returns ndarray of shape (4, freq_bins, time_steps) with raw power values.
    Edge cases — offset near end of file — are handled by zero-padding on the right.
    """
    df = pd.read_parquet(parquet_path)

    # infer time resolution from the time column
    time_vals = df["time"].to_numpy(dtype=np.float32)
    if len(time_vals) < 2:
        dt = 2.0
    else:
        dt = float(np.median(np.diff(time_vals)))
    dt = max(dt, 1e-6)

    n_steps = max(1, round(window_sec / dt))

    # convert offset to row index
    start_idx = int(np.searchsorted(time_vals, offset_sec))
    end_idx   = start_idx + n_steps

    chains_out = []
    for chain in CHAINS:
        cols = _chain_cols(df, chain)
        if not cols:
            # chain missing — return zeros for this chain
            chains_out.append(np.zeros((len(cols) or 1, n_steps), dtype=np.float32))
            continue

        arr = df[cols].to_numpy(dtype=np.float32).T  # (freq_bins, total_time)
        freq_bins = arr.shape[0]

        # slice, pad right if window extends past file end
        if start_idx >= arr.shape[1]:
            window = np.zeros((freq_bins, n_steps), dtype=np.float32)
        else:
            sliced = arr[:, start_idx:end_idx]        # (freq, actual_steps)
            if sliced.shape[1] < n_steps:
                pad = n_steps - sliced.shape[1]
                sliced = np.pad(sliced, ((0, 0), (0, pad)), mode="constant")
            window = sliced

        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        chains_out.append(window)

    return np.stack(chains_out, axis=0)   # (4, freq_bins, n_steps)


class SpectrogramDataset(Dataset):
    """
    Dataset for HMS spectrogram images.

    Parameters
    ----------
    metadata_csv : path to train_test_split.csv
    spectrogram_dir : directory containing {spectrogram_id}.parquet files
    cfg : Config2D instance (or any object with the required attributes)
    split : "train" | "val" | "all"
    """

    def __init__(self, metadata_csv: str, spectrogram_dir: str, cfg, split: str = "train"):
        meta = pd.read_csv(metadata_csv)
        meta = meta[meta["split"] == "trainval"].reset_index(drop=True)

        if split == "train":
            meta = meta[meta["inner_fold"] != cfg.val_fold].reset_index(drop=True)
        elif split == "val":
            meta = meta[meta["inner_fold"] == cfg.val_fold].reset_index(drop=True)
        # "all" keeps everything

        self.meta           = meta
        self.spec_dir       = spectrogram_dir
        self.window_sec     = cfg.spec_window_seconds
        self.img_height     = cfg.img_height
        self.img_width      = cfg.img_width

        # pre-compute soft labels
        votes    = meta[VOTE_COLS].to_numpy(dtype=np.float32)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.soft_labels = (votes / row_sums)          # (N, 6)
        self.hard_labels = self.soft_labels.argmax(1)  # (N,)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row    = self.meta.iloc[idx]
        spec_id = int(row["spectrogram_id"])
        offset  = float(row["spectrogram_label_offset_seconds"])

        path = os.path.join(self.spec_dir, f"{spec_id}.parquet")

        # (4, freq_bins, time_steps)  raw power
        img = _load_spec_window(path, offset, self.window_sec)

        # log1p to compress power dynamic range
        img = np.log1p(img)

        # resize to (4, H, W) via bilinear interpolation
        img_t = torch.from_numpy(img).unsqueeze(0)         # (1, 4, freq, time)
        img_t = F.interpolate(
            img_t,
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)                                        # (4, H, W)

        soft = torch.from_numpy(self.soft_labels[idx])     # (6,)
        label = int(self.hard_labels[idx])

        return {"image": img_t, "soft_label": soft, "label": label}


# ── Smoke test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("SpectrogramDataset smoke test (synthetic data)")

    # build a tiny synthetic spectrogram parquet
    import tempfile, pathlib

    freq_bins = 100
    n_time    = 300
    time_vals = np.arange(n_time, dtype=np.float32) * 2.0   # 2s steps → 600s file

    rng  = np.random.default_rng(0)
    data = {"time": time_vals}
    for chain in CHAINS:
        for i in range(freq_bins):
            data[f"{chain}_{i * 0.2:.2f}"] = rng.exponential(1.0, size=n_time).astype(np.float32)

    spec_df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp:
        spec_dir = pathlib.Path(tmp) / "train_spectrograms"
        spec_dir.mkdir()
        spec_id  = 999999
        spec_df.to_parquet(spec_dir / f"{spec_id}.parquet", index=False)

        # build a minimal metadata CSV
        meta_rows = []
        for fold in range(5):
            meta_rows.append({
                "spectrogram_id": spec_id,
                "spectrogram_label_offset_seconds": 0.0,
                "inner_fold": fold,
                "split": "trainval",
                **{c: 1 if i == fold % 6 else 0
                   for i, c in enumerate(VOTE_COLS)},
            })
        meta_df = pd.DataFrame(meta_rows)
        meta_csv = pathlib.Path(tmp) / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        # minimal cfg stub
        class _Cfg:
            val_fold             = 4
            spec_window_seconds  = 300
            img_height           = 128
            img_width            = 256

        for split in ("train", "val"):
            ds = SpectrogramDataset(str(meta_csv), str(spec_dir), _Cfg(), split=split)
            sample = ds[0]
            img, soft, label = sample["image"], sample["soft_label"], sample["label"]
            print(f"  split={split:<5}  len={len(ds)}"
                  f"  image={tuple(img.shape)}  soft_label={tuple(soft.shape)}"
                  f"  label={label}  dtype={img.dtype}")
            assert img.shape  == (4, 128, 256), f"Bad image shape: {img.shape}"
            assert soft.shape == (6,),          f"Bad label shape: {soft.shape}"
            assert abs(soft.sum().item() - 1.0) < 1e-5, "Soft labels don't sum to 1"

    print("All assertions passed.")
