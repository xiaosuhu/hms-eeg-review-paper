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


## 1D CNN (EEGNet-style) — Step 2a/b/c/d

Reporting a single "best val_kl" number is misleading here: from-scratch
training on the 4,800-row clean subset (v7, v7b) shows val_kl bottom out
within the first 1-2 epochs and then rise almost monotonically for the rest
of training — there is no later, better-trained checkpoint to select instead.
Two-step training (v8, v8c) shows the opposite pattern: val_kl in Step2
improves gradually and continuously across all 30 epochs with no divergence.
This asymmetry is itself the mechanistic finding — see note below the table.

Each row reports two anchors: the epoch with lowest val_kl, and the epoch
with highest macro_f1 (often different epochs, since KL and F1 diverge).

| condition | notebook | best-KL epoch (kl / f1) | best-F1 epoch (kl / f1) | curve shape |
|---|---|---|---|---|
| Step 2a (clean baseline) | cnn-1d-v7-clean-data.ipynb | 1 (0.9215 / 0.1293) | 10 (1.3680 / 0.1963) | val_kl rises ~monotonically after ep1; early-stopped ep11 |
| Step 2b (reweighting) | cnn-1d-v7b-weighted-sampler.ipynb | 2 (0.9234 / 0.2144) | 10 (1.2483 / 0.2489) | same rising pattern after ep2; early-stopped ep12 |
| Step 2c (two-step) | cnn-1d-v8-two-step.ipynb | 30 (1.0826 / 0.2216) | 1 (1.2017 / 0.2417) | val_kl improves gradually across all 30 Step2 epochs, no divergence |
| Step 2d (two-step + Step1 reweight) | cnn-1d-v8c-weighted-two-step.ipynb | 24 (1.1426 / 0.2028) | 1 (1.1768 / 0.2366) | same gradual-improvement pattern as 2c; final numbers slightly worse than 2c |

**Interpretation**: v7/v7b's low best-KL numbers (0.92) come from checkpoints
epochs into training that are still largely undertrained (macro_f1 0.13-0.21) —
not representative of a usable model. v8/v8c never hit this failure mode: their
Step2 val_kl keeps improving for the full 30-epoch budget. This is the paper's
central mechanistic claim in action — separating noisy-but-large pretraining
from clean-but-small finetuning changes the *optimization dynamics* of the
finetune phase, not just its data distribution. Step1 reweighting (2d vs 2c)
does not add benefit on top of this — 2c's Step2 outperforms 2d's Step2 on
both final metrics, a negative result worth reporting as-is: the benefit of
two-step training comes from the clean/noisy split, not from reweighting.

Absolute KL values here (~0.92-1.6) are well above top Kaggle solutions
(Bhatti et al. report KL ≈ 0.24, using an ensembled/pretrained/heavily-tuned
pipeline). This is expected and by design: this ablation isolates one variable
at a time under a fixed, un-tuned architecture — it is not attempting to match
leaderboard performance (see paper's stated scope).