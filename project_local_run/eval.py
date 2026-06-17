# eval.py (root)
import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import yaml

from project_local_run.src.data import EEGDataset
from project_local_run.src.models.classifier import build_model
from project_local_run.src.losses import build_loss
from project_local_run.src.evaluation import validate

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g., runs/exp/best.pt)")
    ap.add_argument("--split", type=str, default="val", choices=["val","train"], help="Which split to evaluate")
    ap.add_argument("--out_csv", type=str, default=None, help="Optional path to save per-sample predictions CSV")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta_key = "val_meta" if args.split == "val" else "train_meta"
    ds = EEGDataset(cfg["data"][meta_key], cfg["data"]["eeg_dir"], split=args.split, **cfg["data"].get("ds_kwargs", {}))
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                        num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # Build model & load weights
    model = build_model(cfg["model"]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])

    loss_fn = build_loss(cfg).to(device)

    val_loss, metrics, probs, y_true = validate(model, loader, loss_fn, cfg, device)
    print(json.dumps({"split": args.split, "val_loss": val_loss, **metrics}, indent=2))

    # Optional: save per-sample predictions (use ds to also dump IDs if your dataset exposes them)
    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame(probs.numpy(), columns=[f"prob_{i}" for i in range(cfg["model"]["num_classes"])])
        # If your dataset exposes IDs, attach them here:
        # df.insert(0, "eeg_id", ds.ids)  # example if you have ds.ids
        df["y_true"] = y_true.numpy()
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved predictions to {args.out_csv}")
