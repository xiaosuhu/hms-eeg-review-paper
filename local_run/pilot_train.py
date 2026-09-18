import argparse, yaml, torch, numpy as np
from torch import nn
from torch.utils.data import Subset
from pathlib import Path
from .src.utils import set_seed
from .src.models.classifier import EEGClassifier
from .src.data import load_split_frames

set_seed(42)  # Set a fixed seed for reproducibility

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    tot_loss, correct, total = 0.0, 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        tot_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return tot_loss/total, correct/max(total,1)

def maybe_subset(df, frac):
    if frac >= 1.0: return df
    n = max(1, int(len(df) * frac))
    return df.sample(n, random_state=0).reset_index(drop=True)

def main(cfg):
    set_seed(cfg["seed"])
    device = torch.device(cfg["device"] if torch.cuda.is_available() or cfg["device"]=="cpu" else "cpu")

    tr, va, _ = load_split_frames(cfg["meta_path"], cfg["fold"], cfg["split_key"], cfg["fold_key"])
    tr = maybe_subset(tr, cfg["sample_frac"])
    va = maybe_subset(va, min(1.0, cfg["sample_frac"]))  # keep val small for speed if you want

    train_loader = make_loader(tr, cfg["eeg_key"], cfg["label_key"], cfg["batch_size"], cfg["num_workers"], True)
    val_loader   = make_loader(va, cfg["eeg_key"], cfg["label_key"], cfg["batch_size"], cfg["num_workers"], False)

    model = EEGClassifier(in_ch=cfg["input_channels"], num_classes=cfg["num_classes"]).to(device)

    if cfg["num_classes"] > 1:
        loss_fn = nn.CrossEntropyLoss()
    else:
        # Binary—use BCEWithLogits; wrap target to float and adjust evaluate if needed
        loss_fn = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    best_val = float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0; seen = 0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y if cfg["num_classes"]>1 else y.float().unsqueeze(1))
            loss.backward(); opt.step()
            running += loss.item() * x.size(0); seen += y.size(0)
        train_loss = running/max(seen,1)

        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device, cfg["num_classes"])
        print(f"Epoch {epoch+1}/{cfg['epochs']} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({"model": model.state_dict(), "cfg": cfg}, "checkpoints/best_fold{}.pt".format(cfg["fold"]))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="config.yaml")
    args = ap.parse_args()
    with open(args.cfg, "r") as f: cfg = yaml.safe_load(f)
    main(cfg)
