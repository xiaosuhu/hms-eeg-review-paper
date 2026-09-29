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
| cnn-1d-v7-clean-data.ipynb | 5b17af5 | 0.9215 | 0.1293 | (prior session — see thread-summary-1d-cnn-ablation-complete.md) | Step 2a baseline, best-KL anchor |
| cnn-1d-v7b-weighted-sampler.ipynb | 5b17af5 | 0.9234 | 0.2144 | (prior session) | Step 2b reweighting, best-KL anchor |
| cnn-1d-v8-two-step.ipynb | 5b17af5 | 1.0826 | 0.2216 | (prior session) | Step 2c two-step, best-KL anchor (Step2, epoch 30) |
| cnn-1d-v8c-weighted-two-step.ipynb | 5b17af5 | 1.1426 | 0.2028 | (prior session) | Step 2d, best-KL anchor (Step2, epoch 24) |
| cnn-2d-v1-baseline.ipynb | 9c090b3 | 0.6993 | 0.4766 | 2026-09-29 | 2D baseline, best-KL anchor |
| cnn-2d-v3-efficientnet.ipynb | 9c090b3 | 0.6402 | 0.4787 | 2026-09-29 | 2D EfficientNet-B0, best-KL anchor |
| cnn-2d-v4-two-step.ipynb | 5b17af5 | 0.4879 | 0.5317 | 2026-09-29 | 2D two-step, Step2 best-KL anchor |
| xgboost-v3-clean-data.ipynb | 5b17af5 | 0.8409 | 0.2002 | 2026-09-29 | XGBoost clean baseline |
| xgboost-v4-sample-weight.ipynb | 5b17af5 | 0.8938 | 0.2676 | 2026-09-29 | XGBoost + sample_weight |

- cnn-2d-v1-baseline.ipynb and cnn-2d-v3-efficientnet.ipynb were originally
  scored against a different, non-comparable validation set (an inner fold
  of the full noisy `train_test_split.csv`, not the shared clean test
  split) until commit `9c090b3` fixed this. **Any v1/v3 numbers recorded
  before `9c090b3` are invalid and must not be reused.**

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
(Bhatti et al., arXiv:2412.07878, report KL ≈ 0.24 for Multi-Stage TinyViT
and ≈ 0.26-0.34 for Multi-Stage EfficientNet). This is expected and by
design: this ablation isolates one variable at a time under a fixed,
un-tuned architecture — it is not attempting to match leaderboard
performance (see paper's stated scope).

**Important correction — Bhatti et al.'s two-stage order is the reverse
of ours.** Verified directly from the paper's Section III.D: their
Stage 1 trains on the *clean*, high-confidence (≥10-vote) data first (5
epochs), then Stage 2 fine-tunes on the *full* dataset including
low-vote samples, with progressively-reduced weight for lower-vote data
(15 more epochs). This is the opposite order from v8/v8c (1D) and v4
(2D) here, which pretrain on the full noisy data first and finetune on
the clean subset second. Do not describe this ablation as replicating
Bhatti et al.'s method — it independently tests a different-order
staged curriculum (the more standard "pretrain on large/noisy, finetune
on small/clean" transfer-learning pattern) and finds a similar
directional benefit. Also confirmed: Bhatti et al. do not report a
controlled single-step + reweighting-only ablation against their
two-stage approach for any one fixed architecture, and offer no
mechanistic explanation for why two-stage training works (their paper
states only that "the two-stage training approach ... proved to be the
most effective strategy" as an empirical observation) — both remain
genuine differentiators of this paper's contribution. Whether a
reverse-order (clean-first) staged run would show the same
divergence-reduction pattern here is an open question, deliberately
left for Section 3's structured 11-solution review to inform (i.e.,
whether Bhatti et al.'s stage order is common among top solutions, not
an isolated choice) before deciding whether it's worth a dedicated
follow-up experiment.

## 2D CNN (EfficientNet-B0) — baseline / same-architecture check / two-step

All three notebooks (v1, v3, v4) are scored against the same shared
clean 1,139-row test split used everywhere else in this ablation
(`load_clean_data(CLEAN_PATH, split='test')`) as of commit `9c090b3`.
v1 and v3 both train on the full ~83K-row noisy trainval split in a
single step; v4 trains in two steps (Step1: full noisy data, Step2:
finetune on the clean 4,800-row subset). v1 and v3 use the identical
underlying architecture (`EfficientNetEEG`, backbone=`efficientnet_b0`)
and near-identical training code — v3 exists as an independent
cross-check of v1, written separately; the two notebooks' numbers
should be read as a consistency check on each other, not as two
different conditions.

| condition | notebook | best-KL epoch (kl / f1) | best-F1 epoch (kl / f1) | curve shape |
|---|---|---|---|---|
| Baseline (single-step, full data) | cnn-2d-v1-baseline.ipynb | 1 (0.6993 / 0.4766) | 8 (0.9631 / 0.4889) | val_kl rises with noise after ep1 (+68% by ep11); early-stopped ep11 |
| Same architecture, independent run | cnn-2d-v3-efficientnet.ipynb | 1 (0.6402 / 0.4787) | 1 (same as best-KL) | rises with noise after ep1 (+92% by ep11); early-stopped ep11 |
| Two-step — Step1 (full-data pretrain) | cnn-2d-v4-two-step.ipynb | 2 (0.6868 / 0.5085) | 2 (same as best-KL) | rises with noise after ep2 (+79% by ep12); early-stopped ep12 |
| Two-step — Step2 (clean-data finetune) | cnn-2d-v4-two-step.ipynb | 2 (0.4879 / 0.5317) | 6 (0.5148 / 0.5386) | rises gently after ep2 (+13% by ep12) — much milder than the other three rows; early-stopped ep12 |

**Interpretation**: the single-step baseline (v1), its independent
cross-check (v3), and even Step1 of the two-step run (v4, which is
architecturally identical single-step training on the same full data)
all show the same failure mode as 1D's Step 2a: an early best-KL epoch
followed by a noisy but net-rising val_kl. Two-step's Step2 (v4) is the
best performer on both metrics of the four rows (lowest val_kl, highest
best-F1-anchor macro_f1), and its post-optimum rise is far gentler
(+13%) than the other three (+68% to +92%) — but unlike 1D's Step 2c,
it does **not** fully eliminate the divergence: the best-KL epoch is 2
of 12, not the last epoch. The correct claim for 2D is therefore
"two-step substantially reduces, but does not eliminate, the
early-divergence failure mode" — a weaker but still directionally
consistent version of the 1D finding. One candidate (untested)
explanation for the weaker effect: v1/v3/v4-Step1 all fine-tune the
full pretrained EfficientNet-B0 at a constant `lr=1e-3` with a cosine
schedule that has barely decayed by the early-stopping point (~epoch
11-12 of a 50-epoch `T_max`), whereas 1D's v8/v8c use a markedly
smaller Step2 LR (`1e-5`) than Step1 (`1e-3`). This was not re-tested
for 2D — flagged here as a candidate explanation / future work, not a
verified cause (see the v8b LR stress-test footnote above, which found
raising Step2 LR did *not* help in 1D — the direction of any 2D LR
effect is not assumed to be analogous).

## XGBoost — clean baseline / sample-weight

Both notebooks score against the same shared clean test split
(`load_clean_data(CLEAN_PATH, split='test')`, 1,139 rows) used
throughout this ablation — confirmed already consistent, no fix
needed. Unlike the CNN notebooks, XGBoost has no two-step variant in
this ablation; only single-step (v3) vs. single-step + class-balanced
`sample_weight` (v4) are compared.

| condition | notebook | val KL | val F1 | best round (of 2000) |
|---|---|---|---|---|
| Clean baseline | xgboost-v3-clean-data.ipynb | 0.8409 | 0.2002 | 49 |
| + sample_weight (class-balanced) | xgboost-v4-sample-weight.ipynb | 0.8938 | 0.2676 | 232 |

Per-class F1:

| class | v3 | v4 |
|---|---|---|
| seizure | 0.1951 | 0.2913 |
| lpd | 0.0581 | 0.1214 |
| gpd | 0.2105 | 0.3741 |
| lrda | 0.0000 | 0.0000 |
| grda | 0.0611 | 0.1422 |
| other | 0.6762 | 0.6768 |

**Interpretation**: `sample_weight` raises macro F1 substantially
(+34% relative) with gains concentrated in minority classes (lpd, gpd,
grda all roughly double), but val KL gets *worse* (+6% relative), not
better. This is a genuine trade-off, not a free improvement: a third,
architecturally unrelated model family (gradient-boosted trees, vs.
EEGNet and EfficientNet-B0 for the CNNs) independently confirms that
reweighting alone improves hard-classification accuracy at a real cost
to soft-label calibration — consistent with, and sharper than, the 1D
CNN's Step 2a→2b comparison (where KL was roughly flat rather than
measurably worse). `lrda` scores exactly 0.0000 F1 in both conditions —
reweighting could not recover this class at all, suggesting it may be
unlearnable from this feature set at this data volume regardless of
class weighting, worth a one-line mention as a limitation tied to the
quality-filtering-causes-imbalance thesis.

Note on curve shape: the `mlogloss` training curves for both XGBoost
runs are smooth and well-behaved (monotonic-ish train/val convergence
to a plateau), unlike the CNN notebooks' noisy val_kl curves. This is
expected, not a sign either pipeline is wrong: XGBoost's plotted metric
(`mlogloss`) is computed against hard labels with strong shrinkage
(`learning_rate=0.05`) and subsampling regularization on the *full*
dataset each round, while the CNN notebooks' `val_kl` targets a noisy
*soft* multi-rater label distribution using mini-batch SGD/AdamW —
both the loss target and the optimizer's stochasticity differ, and
both plausibly contribute to the smoother XGBoost curves.
