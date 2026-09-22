# Results

All rows below are scored with the **shared** `validate()` from
`kaggle_run/src/evaluation.py`, called as:

```python
validate(model, val_loader, val_loss_fn, eval_cfg, DEVICE, soft_key="soft_y")
```

- val KL divides by `len(loader.dataset)`, not `len(loader)` — earlier inline
  implementations averaged over batches, which biased the number whenever the
  sample count did not divide evenly by the batch size.
- `soft_key="soft_y"` scores KL against the true vote distribution, matching how
  the XGBoost notebooks compute KL against `y_val_soft`. Without it, KL would be
  measured against a one-hot of the hard label.
- `val_loss_fn` must come from `build_loss({"loss": {"name": "kl"}})` — that
  `KLLoss` applies `log_softmax` internally, so `validate()` passes raw logits.

Numbers produced before this change are **not** comparable to the rows below.

Each notebook prints its row in this format in its final cell:
`<notebook> | val_kl=… | val_f1=…`

| notebook | git commit | val KL | val F1 | date | notes |
|---|---|---|---|---|---|
| cnn-1d-v7-clean-data.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | Step 2a baseline |
| cnn-1d-v7b-weighted-sampler.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | Step 2b reweighting |
| cnn-1d-v8-two-step.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | Step 2c two-step |

## Not yet migrated

These still use their own inline validate loop (batch-averaged KL against
one-hot hard labels) and are **not** comparable to the table above:
`cnn-1d-v9-wavenet.ipynb`, `cnn-2d-v1-baseline.ipynb`,
`cnn-2d-v3-efficientnet.ipynb`, `cnn-2d-v4-two-step.ipynb`.
