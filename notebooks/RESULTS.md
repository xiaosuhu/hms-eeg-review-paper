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
| cnn-1d-v8c-weighted-two-step.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | Step 2d: two-step + Step1 reweighting |
| cnn-2d-v1-baseline.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | 2D baseline, migrated to shared validate() |
| cnn-2d-v3-efficientnet.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | 2D EfficientNet-B0, migrated to shared validate() |
| cnn-2d-v4-two-step.ipynb | (TBD) | (TBD) | (TBD) | (TBD) | 2D two-step, migrated to shared validate() |

Note: for the 2D notebooks, `validate()` is called with
`soft_key="soft_label", x_key="image", y_key="label"` — `SpectrogramDataset`
uses different batch-dict key names than `EEGDatasetV2` (which uses
`"x"`/`"y"`/`"soft_y"`), and `validate()` was extended with optional
`x_key`/`y_key` params (default `"x"`/`"y"`, backward compatible) to support this.

## Not yet migrated

Still uses its own inline validate loop (batch-averaged KL against
one-hot hard labels) and is **not** comparable to the table above:
`cnn-1d-v9-wavenet.ipynb`.

## Footnotes

**Step2 LR stress test (cnn-1d-v8b-step2-lr-x10.ipynb), tested and ruled out:**
v8b reran v8's Step2 (clean-data finetune) with `STEP2_LR=1e-4` instead of
v8's `1e-5` (a 10x increase, matching the Stage1/Stage2 LR ratio reported in
the HMS 2nd-place write-up), everything else held fixed. Result: val KL barely
moved (Step2 epochs ranged ~1.45–1.54, landing at 1.4925 on the epoch the old
F1-based checkpoint criterion picked), while macro F1 degraded over the course
of Step2 (peaked early at epoch 4, 0.2671, then trended down to ~0.24 by early
stopping at epoch 19) rather than improving. Conclusion: raising Step2 LR by
10x is not a promising direction — do not re-test this without new evidence.
(v8b predates the checkpoint-selection bugfix in this doc's header, so its own
printed "best f1" epoch is not directly comparable to the val_kl-selected rows
above; the KL/F1 trend across its epochs is still informative.)
