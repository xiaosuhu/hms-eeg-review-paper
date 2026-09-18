"""
Three-condition training framework for HMS EEG classification.

Condition 1  clean           Train on train_clean.csv (4,800 rows, n_votes≥10)
Condition 2  clean_weighted  Same data + WeightedRandomSampler (compensates 12× imbalance)
Condition 3  two_step        Step1: pretrain on full 106 K rows → Step2: finetune on clean

All conditions share the same test set: train_clean.csv split=='test' (1,139 rows).
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ..data import (
    EEGDatasetWaveNet,
    get_sampler,
    load_clean_data,
    load_full_data,
    _CLASS_NAMES,
)
from .classifier import build_model


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloader(df, cfg, is_train=True):
    """Build a DataLoader for one training phase.

    Parameters
    ----------
    df       : DataFrame — rows to use (already split by caller)
    cfg      : dict — full config (uses cfg['data'] and cfg['train'])
    is_train : bool — True → shuffle / sampler; False → no shuffle

    Returns
    -------
    torch.utils.data.DataLoader
    """
    mode = 'train' if is_train else 'val'
    dataset = EEGDatasetWaveNet(
        df_data=df,
        parquet_dir=cfg['data']['parquet_dir'],
        mode=mode,
        downsample=cfg['data'].get('downsample', 5),
        use_lowpass=cfg['data'].get('use_lowpass', True),
        cache_loaded_eegs=True,
    )

    training_mode = cfg['train'].get('training_mode', 'clean')

    if is_train and training_mode == 'clean_weighted':
        sampler = get_sampler(df, label_col='label')
        return DataLoader(
            dataset,
            batch_size=cfg['train']['batch_size'],
            sampler=sampler,
            num_workers=cfg['train'].get('num_workers', 4),
            pin_memory=True,
        )

    return DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size'],
        shuffle=is_train,
        num_workers=cfg['train'].get('num_workers', 4),
        pin_memory=True,
    )


# ── Single-phase training loop ────────────────────────────────────────────────

def train_one_phase(train_df, val_df, cfg, epochs, lr, model=None):
    """Train (or finetune) a model for one phase.

    Parameters
    ----------
    train_df : DataFrame — training rows
    val_df   : DataFrame — validation / test rows for early stopping
    cfg      : dict — full config
    epochs   : int — maximum epochs for this phase
    lr       : float — initial learning rate
    model    : nn.Module or None — None → build fresh model from cfg['model']

    Returns
    -------
    (model, history)  where model has best weights loaded and history is a
    list of dicts with keys epoch, train_kl, val_kl, macro_f1.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model is None:
        model = build_model(cfg['model'])
    model = model.to(device)

    train_dl = build_dataloader(train_df, cfg, is_train=True)
    val_dl   = build_dataloader(val_df,   cfg, is_train=False)

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = AdamW(model.parameters(), lr=lr,
                      weight_decay=cfg['train'].get('weight_decay', 1e-2))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    use_amp  = cfg['train'].get('amp', True) and device.type == 'cuda'
    scaler   = torch.amp.GradScaler('cuda', enabled=use_amp)
    patience = cfg['train'].get('early_stop_patience', 10)

    best_f1    = -1.0
    best_epoch = 0
    wait       = 0
    best_state = None
    history    = []

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, soft_y in train_dl:
            # EEGDatasetWaveNet yields (B, T, C); model expects (B, C, T)
            x      = x.permute(0, 2, 1).to(device)
            soft_y = soft_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits = model(x)
                loss   = criterion(F.log_softmax(logits, dim=1), soft_y)
            scaler.scale(loss).backward()
            if cfg['train'].get('grad_clip'):
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(),
                                         cfg['train']['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_kl_total = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, soft_y in val_dl:
                x      = x.permute(0, 2, 1).to(device)
                soft_y = soft_y.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(x)
                soft_np = soft_y.cpu().numpy()
                prob    = F.softmax(logits, dim=1).cpu().numpy()
                kl = (np.clip(soft_np, 1e-7, 1)
                      * np.log(np.clip(soft_np, 1e-7, 1)
                               / np.clip(prob, 1e-7, 1))).sum(axis=1).mean()
                val_kl_total += kl
                all_preds .extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(soft_y.argmax(1).cpu().tolist())

        avg_train = train_loss   / len(train_dl)
        val_kl    = val_kl_total / len(val_dl)
        macro_f1  = f1_score(all_labels, all_preds,
                             average='macro', zero_division=0)
        elapsed   = time.time() - t0

        history.append(dict(epoch=epoch, train_kl=avg_train,
                            val_kl=val_kl, macro_f1=macro_f1))
        print(f"  Epoch {epoch:03d} | train_kl {avg_train:.4f} | "
              f"val_kl {val_kl:.4f} | macro_f1 {macro_f1:.4f} | {elapsed:.0f}s")

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            best_epoch = epoch
            wait       = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"    ✓ new best macro_f1={best_f1:.4f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"  Phase complete — best macro_f1={best_f1:.4f} at epoch {best_epoch}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ── 2-step training orchestrator ─────────────────────────────────────────────

def train_two_step(cfg, clean_path, full_path, save_path='best_two_step.pt'):
    """Pretrain on full data then finetune on clean data.

    Parameters
    ----------
    cfg        : dict — full config (must include two_step section)
    clean_path : str  — path to train_clean.csv
    full_path  : str  — path to train_test_split.csv
    save_path  : str  — where to save the final model weights

    Returns
    -------
    (model, hist1, hist2)
    """
    ts           = cfg.get('two_step', {})
    step1_epochs = ts.get('step1_epochs', 20)
    step2_epochs = ts.get('step2_epochs', 30)
    step1_lr     = ts.get('step1_lr', 1e-3)
    step2_lr     = ts.get('step2_lr', 1e-4)

    val_df = load_clean_data(clean_path, split='test')

    # ── Step 1: pretrain on full unfiltered data ──────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 1 — Pretrain on full data "
          f"({step1_epochs} epochs, lr={step1_lr:.0e})")
    print(f"{'='*60}")
    train_full = load_full_data(full_path, split='trainval')
    _print_phase_stats("Step1 train", train_full)
    _print_phase_stats("Val/test",    val_df)

    model, hist1 = train_one_phase(
        train_full, val_df, cfg,
        epochs=step1_epochs, lr=step1_lr, model=None,
    )

    # ── Step 2: finetune on clean data ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Step 2 — Finetune on clean data "
          f"({step2_epochs} epochs, lr={step2_lr:.0e})")
    print(f"{'='*60}")
    train_clean = load_clean_data(clean_path, split='trainval')
    _print_phase_stats("Step2 train", train_clean)
    _print_phase_stats("Val/test",    val_df)

    model, hist2 = train_one_phase(
        train_clean, val_df, cfg,
        epochs=step2_epochs, lr=step2_lr, model=model,
    )

    torch.save(model.state_dict(), save_path)
    print(f"\nFinal model saved → {save_path}")
    return model, hist1, hist2


# ── Condition verification (Step 5) ──────────────────────────────────────────

def verify_conditions(clean_path, full_path):
    """Print dataset sizes and class distributions for all three conditions.

    Parameters
    ----------
    clean_path : str — path to train_clean.csv
    full_path  : str — path to train_test_split.csv
    """
    clean_train = load_clean_data(clean_path, split='trainval')
    clean_test  = load_clean_data(clean_path, split='test')
    full_train  = load_full_data(full_path,   split='trainval')

    print("=" * 60)
    print("Condition verification")
    print("=" * 60)

    print("\n[Condition 1 / 2]  Train set (clean, n_votes≥10)")
    _print_phase_stats("train", clean_train)

    print("\n[Condition 3]  Step 1 train set (full, unfiltered)")
    _print_phase_stats("train", full_train)

    print("\n[Condition 3]  Step 2 train set (clean)")
    _print_phase_stats("train", clean_train)

    print("\n[All conditions]  Val / Test set (clean test split)")
    _print_phase_stats("test", clean_test)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Cond 1/2  train : {len(clean_train):>6,}  rows")
    print(f"  Cond 3 Step1    : {len(full_train):>6,}  rows")
    print(f"  Cond 3 Step2    : {len(clean_train):>6,}  rows")
    print(f"  Shared test     : {len(clean_test):>6,}  rows")


def _print_phase_stats(label, df):
    """Print row count and per-class counts for a DataFrame."""
    print(f"  {label}: {len(df):,} rows")
    if 'label' in df.columns:
        counts = df['label'].value_counts().sort_index()
        for k, v in counts.items():
            print(f"    {_CLASS_NAMES.get(int(k), k):<10}: {v:,}")
    elif 'expert_consensus' in df.columns:
        counts = df['expert_consensus'].value_counts()
        for cls, v in counts.items():
            print(f"    {cls:<10}: {v:,}")
