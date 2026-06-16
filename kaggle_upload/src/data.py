import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import cycle
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

def load_split_data(meta_path, fold=None):
    meta = pd.read_csv(meta_path)
    if fold is None:  # only test set
        return None, None, meta[meta["split"] == "test"]

    train_df = meta[(meta["split"] == "trainval") & (meta["inner_fold"] != fold)]
    val_df   = meta[(meta["split"] == "trainval") & (meta["inner_fold"] == fold)]
    test_df  = meta[meta["split"] == "test"]
    return train_df, val_df, test_df

# This is the basic data loading class for a later data loader tool in pytorch
class EEGDataset(Dataset):
    def __init__(self, metadata_path, npy_dir=None, split="train",
                 window_len=10000, transform=None, parquet_dir=None):
        assert (npy_dir is not None) or (parquet_dir is not None), \
            "Provide either npy_dir or parquet_dir."
        self.meta = pd.read_parquet(metadata_path) if metadata_path.endswith(".parquet") else pd.read_csv(metadata_path)
        if split == "train":
            self.meta = self.meta[(self.meta["split"] == "trainval") & (self.meta["inner_fold"] != 4)].reset_index(drop=True)
        elif split == "val":
            self.meta = self.meta[(self.meta["split"] == "trainval") & (self.meta["inner_fold"] == 4)].reset_index(drop=True)
        elif split == "test":
            self.meta = self.meta[self.meta["split"] == "test"].reset_index(drop=True)
        self.npy_dir = npy_dir
        self.parquet_dir = parquet_dir
        self.window_len = window_len
        self.transform = transform
        self.tars = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
        self.meta["expert_consensus"] = self.meta["expert_consensus"].map(self.tars).fillna(-1).astype(int)
        self.targets = self.meta["expert_consensus"].tolist()
        VOTE_COLS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        votes = self.meta[VOTE_COLS].to_numpy(dtype=np.float32)
        row_sums = votes.sum(axis=1, keepdims=True)
        votes = np.where(row_sums == 0, np.full_like(votes, 1.0 / 6), votes / np.where(row_sums == 0, 1.0, row_sums))
        self.votes = votes

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        eeg_id = int(row['eeg_id'])
        offset = int(row['eeg_label_offset_seconds']) * 200
        label = int(row['expert_consensus'])

        if self.parquet_dir is not None:
            fpath = os.path.join(self.parquet_dir, f"{eeg_id}.parquet")
            eeg = pd.read_parquet(fpath).values.astype(np.float32)
            # shape: [T, 20] — transpose to [20, T]
            eeg = eeg.T
        else:
            eeg = np.load(os.path.join(self.npy_dir, f"{eeg_id}.npy"),
                          mmap_mode='r')
            eeg = eeg.T  # [T, 20] -> [20, T]
        window = eeg[:, offset:offset + self.window_len]

        if self.transform:
            window = self.transform(window)

        # bipolar montage: parquet column order Fp1=0,F3=1,C3=2,P3=3,F7=4,T3=5,T5=6,O1=7,
        #   Fz=8,Cz=9,Pz=10,Fp2=11,F4=12,C4=13,P4=14,F8=15,T4=16,T6=17,O2=18,EKG=19
        window = np.stack([
            window[0]  - window[5],   # Fp1-T3
            window[5]  - window[7],   # T3-O1
            window[0]  - window[2],   # Fp1-C3
            window[2]  - window[7],   # C3-O1
            window[11] - window[13],  # Fp2-C4
            window[13] - window[18],  # C4-O2
            window[11] - window[16],  # Fp2-T4
            window[16] - window[18],  # T4-O2
        ], axis=0)  # [8, T]

        window = window.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        mean = window.mean(axis=1, keepdims=True)
        std  = window.std(axis=1, keepdims=True)
        std  = np.where(std < 1e-6, 1e-6, std)
        window = (window - mean) / std
        soft = torch.tensor(self.votes[idx], dtype=torch.float32)
        return {"x": window, "y": label, "soft_y": soft}

    def count_nan_samples(self, n=200):
        """Check first n samples for NaN in the raw npy files."""
        nan_count = 0
        for i in range(min(n, len(self.meta))):
            row = self.meta.iloc[i]
            fname = f"{int(row['eeg_id'])}_offset{int(row['eeg_label_offset_seconds'])}.npy"
            arr = np.load(os.path.join(self.eeg_dir, fname), mmap_mode='r')
            if np.isnan(arr).any():
                nan_count += 1
        return nan_count, n

class EEGDatasetV2(Dataset):
    """EEG dataset with GroupKFold-friendly interface: accepts a pre-split df,
    uses center crop, clip+scale, and mu-law encoding instead of z-score."""

    def __init__(self, df, parquet_dir, window_len=10000):
        self.data = df.reset_index(drop=True)
        self.parquet_dir = parquet_dir
        self.window_len = window_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        eeg_id = int(row['eeg_id'])

        fpath = os.path.join(self.parquet_dir, f"{eeg_id}.parquet")
        eeg = pd.read_parquet(fpath).values.astype(np.float32)  # [T, 20]
        rows = len(eeg)
        offset = (rows - self.window_len) // 2
        eeg = eeg[offset:offset + self.window_len]  # [10000, 20]
        eeg = eeg.T                                  # [20, 10000]

        # 8 bipolar channels — same pairs and column indices as EEGDataset
        window = np.stack([
            eeg[0]  - eeg[5],   # Fp1-T3
            eeg[5]  - eeg[7],   # T3-O1
            eeg[0]  - eeg[2],   # Fp1-C3
            eeg[2]  - eeg[7],   # C3-O1
            eeg[11] - eeg[13],  # Fp2-C4
            eeg[13] - eeg[18],  # C4-O2
            eeg[11] - eeg[16],  # Fp2-T4
            eeg[16] - eeg[18],  # T4-O2
        ], axis=0)  # [8, 10000]

        window = window.astype(np.float32)
        window = np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0)
        window = np.clip(window, -1024, 1024) / 32.0

        mu = 256
        window = (np.sign(window) * np.log(1 + mu * np.abs(window))
                  / np.log(mu + 1)).astype(np.float32)

        return {
            "x": window,
            "y": int(row['target']),
            "soft_y": torch.tensor(row['soft_y'], dtype=torch.float32),
        }


# This is the basic data loading class for a later data loader tool in pytorch
class SpecDataset(Dataset):
    def __init__(self, metadata_path, npy_dir, window_len=300):
        self.meta = pd.read_parquet(metadata_path) if metadata_path.endswith(".parquet") else pd.read_csv(metadata_path)
        self.npy_dir = npy_dir
        self.window_len = window_len
        self.tars = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
        self.meta["expert_consensus"] = self.meta["expert_consensus"].map(self.tars).fillna(-1).astype(int)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        spec_id = int(row['spectrogram_id'])
        offset = int(row['spectrogram_label_offset_seconds'])
        label = int(row['expert_consensus'])

        # Load .npy file and extract window
        spec_path = os.path.join(self.npy_dir, f"{spec_id}.npy")
        signal = decode_signal(spec_path, offset=offset, dtype=np.float32)  # [T, C] -> [C, T]

        return signal, label

def decode_signal(path, offset=None, dtype=np.float32):
        # Load and decode .npy file
        sig = np.load(path).astype(dtype)  # Already decoded from npy
        sig = sig.T

        # Crop based on offset
        if offset is not None:
            offset = offset // 2
            sig = sig[:, offset:offset + 300]
            # Pad to ensure shape (400, 300)
            if sig.shape[1] < 300:
                pad_width = 300 - sig.shape[1]
                sig = np.pad(sig, ((0, 0), (0, pad_width)), mode='constant')

        # Clip and log-transform
        # sig = np.clip(sig, np.exp(-4.0), np.exp(8.0))
        # sig = np.log(sig)

        # Normalize
        sig -= sig.mean()
        sig /= sig.std() + 1e-6

        # Convert to 3-channel image shape [C, H, W]
        sig = np.stack([sig]*3, axis=0)

        return torch.tensor(sig, dtype=torch.float32)


class EEGDatasetWaveNet(Dataset):
    def __init__(
        self,
        df_data,
        parquet_dir=None,               # if you want to lazy-load parquets by path
        eegs=None,                      # OR pass a preloaded dict {eeg_id: np.ndarray [10000, len(FEATS)]}
        mode='train',
        FEATS=('Fp1','T3','C3','O1','Fp2','C4','T4','O2'),
        TARGETS=('seizure_vote','lpd_vote','gpd_vote','lrda_vote','grda_vote','other_vote'),
        downsample=5,
        # filter params
        use_lowpass=True,
        cutoff_freq=20, sampling_rate=200, order=4,
        # performance
        cache_loaded_eegs=True
    ):
        assert (parquet_dir is not None) or (eegs is not None), "Provide parquet_dir or preloaded eegs."
        self.data = df_data.reset_index(drop=True)
        self.parquet_dir = parquet_dir
        self.eegs = {} if eegs is None else eegs
        self.mode = mode
        self.downsample = int(downsample)
        self.FEATS = list(FEATS)
        self.TARGETS = list(TARGETS)
        self.use_lowpass = use_lowpass
        self.cutoff_freq = cutoff_freq
        self.sampling_rate = sampling_rate
        self.order = order
        self.cache_loaded_eegs = cache_loaded_eegs

        # precompute map
        self.FEAT2IDX = {x: i for i, x in enumerate(self.FEATS)}

        # design filter once if used
        if self.use_lowpass:
            nyquist = 0.5 * self.sampling_rate
            normal_cutoff = self.cutoff_freq / nyquist
            self._butter_ba = butter(self.order, normal_cutoff, btype='low', analog=False)
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X_np, y_np = self._build_sample(index)
        return (
            torch.tensor(X_np[::self.downsample, :], dtype=torch.float32),
            torch.tensor(y_np, dtype=torch.float32)
        )

    # ---------- Integrated helpers ----------
    def _eeg_from_parquet(self, parquet_path):
        """Read middle 50s (10000 rows) and return np.float32 [10000, len(FEATS)]."""
        eeg = pd.read_parquet(parquet_path, columns=self.FEATS)
        rows = len(eeg)
        if rows < 10_000:
            # pad (rare) or raise—here we center-crop with edge padding
            pad = 10_000 - rows
            top = pad // 2
            bottom = pad - top
            eeg = pd.concat([eeg.iloc[:1].repeat(top), eeg, eeg.iloc[-1:].repeat(bottom)], ignore_index=True)
            rows = len(eeg)

        offset = (rows - 10_000) // 2
        eeg = eeg.iloc[offset:offset+10_000]

        data = np.zeros((10_000, len(self.FEATS)), dtype=np.float32)
        for j, col in enumerate(self.FEATS):
            x = eeg[col].astype('float32').to_numpy(copy=False)
            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:  # not all NaN
                x = np.nan_to_num(x, nan=(0.0 if np.isnan(m) else m))
            else:
                x[:] = 0.0
            data[:, j] = x
        return data

    def _butter_lowpass_filter(self, data):
        b, a = self._butter_ba
        # lfilter expects last dim as axis=0 by default; we want time on axis=0, channels on axis=1
        return lfilter(b, a, data, axis=0)

    # ---------- Sample builder ----------
    def _build_sample(self, index):
        row = self.data.iloc[index]

        # get EEG matrix [10000, len(FEATS)]
        if row.eeg_id in self.eegs:
            eeg = self.eegs[row.eeg_id]
        else:
            if self.parquet_dir is None:
                raise ValueError(f"Missing EEG for {row.eeg_id} and no parquet_dir provided.")
            # infer path; adjust if your filenames differ
            parquet_path = f"{self.parquet_dir}/{int(row.eeg_id)}.parquet"
            eeg = self._eeg_from_parquet(parquet_path)
            if self.cache_loaded_eegs:
                self.eegs[row.eeg_id] = eeg

        # build 8 bipolar channels
        sample = np.zeros((10_000, 8), dtype=np.float32)
        F = self.FEAT2IDX
        sample[:, 0] = eeg[:, F['Fp1']] - eeg[:, F['T3']]
        sample[:, 1] = eeg[:, F['T3']] - eeg[:, F['O1']]
        sample[:, 2] = eeg[:, F['Fp1']] - eeg[:, F['C3']]
        sample[:, 3] = eeg[:, F['C3']] - eeg[:, F['O1']]
        sample[:, 4] = eeg[:, F['Fp2']] - eeg[:, F['C4']]
        sample[:, 5] = eeg[:, F['C4']] - eeg[:, F['O2']]
        sample[:, 6] = eeg[:, F['Fp2']] - eeg[:, F['T4']]
        sample[:, 7] = eeg[:, F['T4']] - eeg[:, F['O2']]

        # standardize/clip like your original
        sample = np.clip(sample, -1024, 1024)
        sample = np.nan_to_num(sample, nan=0.0) / 32.0

        # optional low-pass
        if self.use_lowpass:
            sample = self._butter_lowpass_filter(sample)

        # targets
        if self.mode != 'test':
            y = row[self.TARGETS].to_numpy(dtype=np.float32, copy=True)
        else:
            y = np.zeros(6, dtype=np.float32)

        return sample, y


