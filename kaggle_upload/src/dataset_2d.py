"""
dataset_2d.py — PyTorch Dataset for HMS spectrogram (2D CNN pipeline).

Pre-processing (run once):
    preprocess_spectrograms(spectrogram_dir, cfg.spec_cache_dir)
    Converts each {spec_id}.parquet → {spec_id}.npy of shape (400, total_time_cols),
    where 400 = 4 chains × 100 freq bins stacked vertically.

__getitem__ returns:
  {
    "image"     : FloatTensor (4, 100, 300),   # clip→log→z-score, 4 chains
    "soft_label": FloatTensor (6,),             # vote proportions
    "label"     : int,                          # argmax hard label
  }
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed

VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote",
             "lrda_vote",    "grda_vote", "other_vote"]


# ── Pre-processing ───────────────────────────────────────────────────────────

def _preprocess_one(spec_id: int, spectrogram_dir: str, cache_dir: str) -> None:
    dst = os.path.join(cache_dir, f"{spec_id}.npy")
    if os.path.exists(dst):
        return
    src = os.path.join(spectrogram_dir, f"{spec_id}.parquet")
    df  = pd.read_parquet(src)
    df  = df.fillna(0)
    if "time" in df.columns:
        df = df.drop(columns=["time"])
    # (total_time, 400) → transpose → (400, total_time)
    arr = df.to_numpy(dtype=np.float32).T
    np.save(dst, arr)


def preprocess_spectrograms(
    spectrogram_dir: str,
    cache_dir: str,
    n_jobs: int = 4,
) -> None:
    """
    Convert every {spec_id}.parquet in spectrogram_dir to {spec_id}.npy in cache_dir.

    Output shape per file: (400, total_time_cols)
      400 = 4 EEG chains × 100 frequency bins, stacked vertically in parquet column order
            (LL_*, RL_*, LP_*, RP_*).
    Skips files that already exist in cache_dir.
    """
    os.makedirs(cache_dir, exist_ok=True)
    spec_ids = [
        int(f[:-8])
        for f in os.listdir(spectrogram_dir)
        if f.endswith(".parquet")
    ]
    Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(_preprocess_one)(sid, spectrogram_dir, cache_dir)
        for sid in spec_ids
    )
    print(f"Preprocessed {len(spec_ids)} spectrograms → {cache_dir}")


# ── Dataset ──────────────────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """
    Dataset for HMS spectrogram images.

    Requires preprocess_spectrograms() to have been called first so that
    cfg.spec_cache_dir contains the .npy files.

    Parameters
    ----------
    metadata_csv    : path to train_test_split.csv
    spectrogram_dir : directory containing {spectrogram_id}.parquet (used by
                      preprocess_spectrograms, not read directly here)
    cfg             : Config2D instance (or any object with the required attributes)
    split           : "train" | "val" | "all"
    """

    def __init__(self, metadata_csv: str, spectrogram_dir: str, cfg, split: str = "train"):
        meta = pd.read_csv(metadata_csv)
        meta = meta[meta["split"] == "trainval"].reset_index(drop=True)

        if split == "train":
            meta = meta[meta["inner_fold"] != cfg.val_fold].reset_index(drop=True)
        elif split == "val":
            meta = meta[meta["inner_fold"] == cfg.val_fold].reset_index(drop=True)
        # "all" keeps everything

        self.meta          = meta
        self.spec_cache_dir = cfg.spec_cache_dir

        # pre-compute soft labels
        votes    = meta[VOTE_COLS].to_numpy(dtype=np.float32)
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        self.soft_labels = (votes / row_sums)          # (N, 6)
        self.hard_labels = self.soft_labels.argmax(1)  # (N,)

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        row     = self.meta.iloc[idx]
        spec_id = int(row["spectrogram_id"])
        offset  = float(row["spectrogram_label_offset_seconds"])

        # load pre-cached npy: (400, total_time_cols)
        spec = np.load(os.path.join(self.spec_cache_dir, f"{spec_id}.npy"))

        # extract 300-column window; 2-second time resolution → col = offset // 2
        col_start = int(offset // 2)
        window    = spec[:, col_start:col_start + 300]   # (400, ≤300)

        # right-pad to exactly 300 columns if the window runs off the end
        if window.shape[1] < 300:
            pad    = 300 - window.shape[1]
            window = np.pad(window, ((0, 0), (0, pad)), mode="constant")

        # normalize: clip → log → z-score
        window = np.clip(window, np.exp(-4), np.exp(8))
        window = np.log(window)
        mu     = window.mean()
        sigma  = window.std()
        window = (window - mu) / (sigma + 1e-6)

        # split 400 stacked rows back into 4 chains: (400, 300) → (4, 100, 300)
        window = window.reshape(4, 100, 300)

        img   = torch.from_numpy(window).float()        # (4, 100, 300)
        soft  = torch.from_numpy(self.soft_labels[idx]) # (6,)
        label = int(self.hard_labels[idx])

        return {"image": img, "soft_label": soft, "label": label}


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, pathlib

    print("SpectrogramDataset smoke test (synthetic data)")

    n_time   = 400   # 400 time columns → 800 seconds at 2-s resolution
    n_chains = 4
    n_freq   = 100   # freq bins per chain

    rng  = np.random.default_rng(0)
    data = {"time": np.arange(n_time, dtype=np.float32) * 2.0}
    for chain in ["LL", "RL", "LP", "RP"]:
        for i in range(n_freq):
            data[f"{chain}_{i}"] = rng.exponential(1.0, size=n_time).astype(np.float32)
    spec_df = pd.DataFrame(data)

    with tempfile.TemporaryDirectory() as tmp:
        spec_dir   = pathlib.Path(tmp) / "train_spectrograms"
        cache_dir  = pathlib.Path(tmp) / "spec_cache"
        spec_dir.mkdir(); cache_dir.mkdir()

        spec_id = 999999
        spec_df.to_parquet(spec_dir / f"{spec_id}.parquet", index=False)

        # run pre-processing
        preprocess_spectrograms(str(spec_dir), str(cache_dir), n_jobs=1)
        npy = np.load(cache_dir / f"{spec_id}.npy")
        assert npy.shape == (400, n_time), f"Expected (400, {n_time}), got {npy.shape}"
        print(f"  preprocess OK  npy.shape={npy.shape}")

        # build minimal metadata CSV
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
        meta_df  = pd.DataFrame(meta_rows)
        meta_csv = pathlib.Path(tmp) / "meta.csv"
        meta_df.to_csv(meta_csv, index=False)

        class _Cfg:
            val_fold      = 4
            spec_cache_dir = str(cache_dir)
            img_height    = 100
            img_width     = 300

        for split in ("train", "val"):
            ds     = SpectrogramDataset(str(meta_csv), str(spec_dir), _Cfg(), split=split)
            sample = ds[0]
            img, soft, label = sample["image"], sample["soft_label"], sample["label"]
            print(f"  split={split:<5}  len={len(ds)}"
                  f"  image={tuple(img.shape)}  soft_label={tuple(soft.shape)}"
                  f"  label={label}  dtype={img.dtype}")
            assert img.shape  == (4, 100, 300), f"Bad image shape: {img.shape}"
            assert soft.shape == (6,),          f"Bad label shape: {soft.shape}"
            assert abs(soft.sum().item() - 1.0) < 1e-5, "Soft labels don't sum to 1"

    print("All assertions passed.")
