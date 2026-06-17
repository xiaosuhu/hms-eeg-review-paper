# train.py (root)
import os, json, argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR

from project.src.utils import set_seed, get_lr_lambda
from project.src.data import EEGDataset
from project.src.models.classifier import build_model
from project.src.losses import build_loss
from project.src.evaluation import validate

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_used_cfg(cfg, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "used_config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

def make_dataloaders(cfg):
    train_ds = EEGDataset(cfg["data"]["train_meta"], cfg["data"]["eeg_dir"], split="train", **cfg["data"].get("ds_kwargs", {}))
    val_ds   = EEGDataset(cfg["data"]["val_meta"],   cfg["data"]["eeg_dir"], split="val",   **cfg["data"].get("ds_kwargs", {}))

    sampler = None
    if cfg["train"].get("use_weighted_sampler", False):
        class_counts = np.bincount(np.asarray(train_ds.targets))
        class_weights = 1.0 / np.clip(class_counts, 1, None)
        sample_weights = [class_weights[t] for t in train_ds.targets]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=(sampler is None),
                              sampler=sampler, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["train"]["batch_size"], shuffle=False,
                              num_workers=cfg["train"]["num_workers"], pin_memory=True)
    return train_loader, val_loader

def train(cfg, use_wandb: bool = False, run_name: str | None = None):
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["train"].get("out_dir", "runs"))
    exp_name = run_name or cfg["train"].get("run_name", "exp")
    run_dir = out_dir / exp_name
    save_used_cfg(cfg, run_dir)

    if use_wandb or cfg["log"].get("use_wandb", False):
        import wandb
        wandb.init(project=cfg["log"]["project"], name=exp_name, config=cfg)

    train_loader, val_loader = make_dataloaders(cfg)

    model = build_model(cfg["model"]).to(device)
    loss_fn = build_loss(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"].get("weight_decay", 1e-2))

    # Epoch-wise LambdaLR using your get_lr_lambda
    lr_lambda = get_lr_lambda(batch_size=cfg["train"]["batch_size"],
                              mode=cfg["train"].get("lr_mode", "cos"),
                              epochs=cfg["train"]["epochs"],
                              plot=False)
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=cfg["train"].get("amp", True))

    best_score = -1e9
    key = cfg["train"].get("early_stop_key", "macro_f1")
    patience = cfg["train"].get("early_stop_patience", 10)
    wait = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0

        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"].get("amp", True)):
                logits = model(x)
                if cfg["loss"].get("name", "ce").lower() == "kl":
                    if y.ndim == 1:
                        # defer one-hot to evaluation if you prefer; done inline here
                        num_classes = cfg["model"]["num_classes"]
                        y_probs = torch.zeros((y.size(0), num_classes), device=y.device)
                        y_probs.scatter_(1, y.unsqueeze(1).long(), 1.0)
                        loss = loss_fn(logits, y_probs)
                    else:
                        loss = loss_fn(logits, y)
                else:
                    loss = loss_fn(logits, y.long())

            scaler.scale(loss).backward()
            if cfg["train"].get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer); scaler.update()

            running += loss.item() * x.size(0)

        scheduler.step()

        val_loss, val_metrics, _, _ = validate(model, val_loader, loss_fn, cfg, device)
        train_loss = running / len(train_loader.dataset)
        score = val_metrics.get(key, -val_loss)

        log = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        print(json.dumps(log))
        if use_wandb or cfg["log"].get("use_wandb", False):
            import wandb; wandb.log(log)

        # save
        torch.save({"model": model.state_dict(), "cfg": cfg}, run_dir / "last.pt")

        if score > best_score:
            best_score, wait = score, 0
            torch.save({"model": model.state_dict(), "cfg": cfg}, run_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best {key}: {best_score:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    train(cfg, use_wandb=args.wandb, run_name=args.run_name)
