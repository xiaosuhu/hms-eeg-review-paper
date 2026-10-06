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
| cnn-2d-v3-efficientnet.ipynb | 9c090b3 | 0.6402 | 0.4787 | 2026-09-29 | 2D EfficientNet-B0, best-KL anchor |
| cnn-2d-v4-two-step.ipynb | 5b17af5 | 0.4879 | 0.5317 | 2026-09-29 | 2D two-step, Step2 best-KL anchor |
| xgboost-v3-clean-data.ipynb | 5b17af5 | 0.8409 | 0.2002 | 2026-09-29 | XGBoost clean baseline |
| xgboost-v4-sample-weight.ipynb | 5b17af5 | 0.8938 | 0.2676 | 2026-09-29 | XGBoost + sample_weight |
| cnn-2d-v5-clean-data.ipynb | d4bfcd6 | 0.4794 | 0.5197 | 2026-10-05 | 2D clean-only, single-step (epoch 6) |
| cnn-2d-v5b-weighted-sampler.ipynb | d4bfcd6 | 0.5131 | 0.4657 | 2026-10-05 | 2D clean + class-balanced sampler (epoch 12) |
| cnn-1d-v7c-full-noisy.ipynb | d4bfcd6 | 1.0938 | 0.2266 | (not recorded) | 1D full noisy, single-step (epoch 1) |
| cnn-1d-v7d-clean-lr1e-5.ipynb | d4bfcd6 | 0.9016 | 0.1151 | (not recorded) | 1D control: from scratch, clean-only, lr 1e-5 (epoch 29) |
| xgboost-v5-full-noisy.ipynb | d4bfcd6 | 0.9964 | 0.2782 | (not recorded) | XGBoost full noisy, best round 132 |

All rows are seed 42 (single run); multi-seed reruns (43, 44) are pending.

- cnn-2d-v3-efficientnet.ipynb was originally scored against a different,
  non-comparable validation set (an inner fold of the full noisy
  `train_test_split.csv`, not the shared clean test split) until commit
  `9c090b3` fixed this. (cnn-2d-v1-baseline.ipynb, which had the same
  problem, was retired and moved to `old_ver/`.) **Any v3 numbers recorded
  before `9c090b3` are invalid and must not be reused.**

## Reference baselines (no model)

A constant prediction equal to the mean soft label of the trainval rows,
scored with the same KL (probabilities clipped at 1e-7).

| test set | clean-train prior | full-train prior | uniform |
|---|---|---|---|
| clean test (n=1,139) KL | 0.8748 | 1.0882 | 1.136 |
| clean test macro-F1 | 0.1112 (equals always predicting "other") | n/a | n/a |
| non-clean test (n=14,863) KL | 2.0521 | 1.5336 | 1.5621 |

clean-train prior = mean soft label of the 4,800 clean trainval rows;
full-train prior = mean over the 83,430 trainval rows.

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
10x is not a promising direction — do not re-test this without new evidence. Caveat added after the v7d control: 1D KL is at prior level (see the 1D Interpretation), so this stress test carries little information about KL; only its F1 trend is informative.
(v8b predates the checkpoint-selection bugfix in this doc's header, so its own
printed "best f1" epoch is not directly comparable to the val_kl-selected rows
above; the KL/F1 trend across its epochs is still informative.)


## 1D CNN (EEGNet-style) — clean-only, reweighting, two-step, full-noisy, LR control

Reporting a single "best val_kl" number is misleading here: from-scratch
training on the 4,800-row clean subset (v7, v7b) shows val_kl bottom out
within the first 1-2 epochs and then rise almost monotonically for the rest
of training — there is no later, better-trained checkpoint to select instead.
Two-step training (v8, v8c) shows the opposite pattern: val_kl in Step2
improves gradually and continuously across all 30 epochs with no divergence. The v7d control below reproduces this no-divergence pattern without any pretraining, so it is attributable to the small learning rate.

Each row reports two anchors: the epoch with lowest val_kl, and the epoch
with highest macro_f1 (often different epochs, since KL and F1 diverge).

| condition | notebook | best-KL epoch (kl / f1) | best-F1 epoch (kl / f1) | curve shape |
|---|---|---|---|---|
| Step 2a (clean baseline) | cnn-1d-v7-clean-data.ipynb | 1 (0.9215 / 0.1293) | 10 (1.3680 / 0.1963) | val_kl rises ~monotonically after ep1; early-stopped ep11 |
| Step 2b (reweighting) | cnn-1d-v7b-weighted-sampler.ipynb | 2 (0.9234 / 0.2144) | 10 (1.2483 / 0.2489) | same rising pattern after ep2; early-stopped ep12 |
| Step 2c (two-step) | cnn-1d-v8-two-step.ipynb | 30 (1.0826 / 0.2216) | 1 (1.2017 / 0.2417) | val_kl improves gradually across all 30 Step2 epochs, no divergence |
| Step 2d (two-step + Step1 reweight) | cnn-1d-v8c-weighted-two-step.ipynb | 24 (1.1426 / 0.2028) | 1 (1.1768 / 0.2366) | same gradual-improvement pattern as 2c; final numbers slightly worse than 2c |
| Full noisy, single-step | cnn-1d-v7c-full-noisy.ipynb | 1 (1.0938 / 0.2266) | 5 (1.3108 / 0.2448) | val_kl rises after epoch 1, peaks at epoch 6 (1.6074, +47% over best), ends epoch 11 at 1.2494 / 0.2281 (+14.2%); early-stopped ep11 |
| Control: from scratch, clean-only, lr 1e-5, 30 epochs | cnn-1d-v7d-clean-lr1e-5.ipynb | 29 (0.9016 / 0.1151) | 1 (1.0504 / 0.1289) | slow monotone-ish decrease, ends epoch 30 at 0.9028 / 0.1150 (+0.1% over best); F1 stays 0.112-0.129 |

**Interpretation (revised after the v7c and v7d runs; seed 42, single run each).**
(1) KL is at prior level. On the clean test split, a constant prediction of the clean-train mean soft label scores KL 0.8748 (see Reference baselines). Every 1D row (0.90-1.14) is at or above that value, so KL differences between 1D conditions do not indicate differences in learned skill. Macro-F1 is the more informative 1D metric: 0.115-0.13 for from-scratch clean-only runs (v7, v7d), 0.21 with reweighting (v7b), and 0.20-0.25 for runs that saw the full noisy data (v7c, v8, v8c).
(2) Absence of divergence is a learning-rate effect, not a pretraining effect. v7d (no pretraining, clean-only, lr 1e-5, 30 epochs) shows no divergence (final epoch +0.1% over best), as v8/v8c do. The earlier reading that two-step training changes the finetune-phase optimization dynamics is therefore withdrawn: the same curve shape is reproduced by the small learning rate alone.
(3) What pretraining supplies is F1, not KL: at the same Step 2 setting, macro-F1 goes from 0.115 (v7d) to 0.22 (v8), while best KL goes from 0.9016 to 1.0826.
(4) Full-noisy single-step training (v7c) shows the same early-best-then-drift shape as the 2D full-noisy baseline.
(5) Step 1 reweighting (2d vs 2c) adds no benefit: 2c's Step 2 outperforms 2d's on both final metrics.

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
on small/clean" transfer-learning pattern) and, at seed 42, finds no KL benefit over clean-only training on the clean test split (2D: 0.4879 vs 0.4794; 1D: 1.0826 vs 0.9215, both at or above the 0.8748 prior baseline), although two-step is better on the non-clean test split in 2D (see the 2D held-out non-clean test subsection). Also confirmed: Bhatti et al. do not report a
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

## 2D CNN (EfficientNet-B0) — full-noisy, clean-only, reweighting, two-step

Notebooks v3, v5, v5b and v4 are all scored on the same shared clean
1,139-row test split used everywhere else in this ablation
(`load_clean_data(CLEAN_PATH, split='test')`) as of commit `9c090b3`.
v3 trains on the full noisy trainval split (~83K rows, single step;
`EfficientNetEEG`, backbone=`efficientnet_b0`, `grad_clip=1.0`,
consistent with v4); v5 and v5b train on the 4,800-row clean trainval
split; v4 trains on the full noisy data first (Step1), then finetunes on
the clean subset (Step2).

| condition | notebook | best-KL epoch (kl / f1) | best-F1 epoch (kl / f1) | final epoch (kl / f1) | best->last KL drift |
|---|---|---|---|---|---|
| Full noisy, single-step | cnn-2d-v3-efficientnet.ipynb | 1 (0.6402 / 0.4787) | 1 (same) | ep11 (1.2302 / 0.4554) | +92% |
| Clean-only, single-step | cnn-2d-v5-clean-data.ipynb | 6 (0.4794 / 0.5197) | 6 (same) | ep16 (0.5685 / 0.4245) | +18.6% |
| Clean + class-balanced sampler | cnn-2d-v5b-weighted-sampler.ipynb | 12 (0.5131 / 0.4657) | 5 (0.5793 / 0.5005) | ep22 (0.5884 / 0.4207) | +14.7% |
| Two-step, Step 1 (full noisy) | cnn-2d-v4-two-step.ipynb | 2 (0.6868 / 0.5085) | 2 (same) | ep12 (1.2279 / 0.4943) | +79% |
| Two-step, Step 2 (clean finetune) | cnn-2d-v4-two-step.ipynb | 2 (0.4879 / 0.5317) | 6 (0.5148 / 0.5386) | ep12 (0.5514 / 0.5360) | +13.0% |

### 2D held-out non-clean test

Definition: the rows of the full test split with fewer than 10 votes
(n=14,863; median 3 votes; patient-disjoint from all training data; may
share eeg_ids with the clean test rows). Evaluated with the best-val_kl
checkpoint of each run:

| condition | notebook | clean test KL / F1 | non-clean test KL / F1 |
|---|---|---|---|
| Full noisy | cnn-2d-v3 | 0.6402 / 0.4787 | 0.9763 / 0.5933 |
| Clean-only | cnn-2d-v5 | 0.4794 / 0.5197 | 1.3325 / 0.5200 |
| Clean + sampler | cnn-2d-v5b | 0.5131 / 0.4657 | 1.4594 / 0.4437 |
| Two-step | cnn-2d-v4 | 0.4879 / 0.5317 | 1.0968 / 0.5230 |

Reference KL on this set: clean-train prior 2.0521, full-train prior 1.5336, uniform 1.5621.

Notes: non-clean labels are noisy, so absolute KL is not comparable with the clean-test column; F1 depends on class prevalence and must not be compared across the two test sets; every checkpoint was selected on the clean test, which does not optimize the non-clean number for any condition; single seed.

**Interpretation**:
(1) The ranking of conditions depends on the test set. Clean test: clean-only (0.4794) ~ two-step (0.4879) < clean+sampler (0.5131) < full noisy (0.6402). Non-clean test: full noisy (0.9763) < two-step (1.0968) < clean-only (1.3325) < clean+sampler (1.4594). The KL benefit of clean-only training on the clean test is therefore at least partly train/test distribution alignment (clean test is drawn from the clean subset; seizure share is 19.8% in full train, 4.1% in clean train, 5.3% in the clean test, 28.6% in the non-clean test).
(2) Two-step is the only condition in the top two on both test sets. Versus clean-only: tie on the clean test (difference 0.0085, below the ~0.04 epoch-to-epoch SD of val_kl), better on the non-clean test (difference 0.236). Single seed, preliminary.
(3) Reweighting does not improve KL on either test set (clean+sampler is worse than clean-only on both), and lowers macro-F1 in 2D.
(4) The earlier statement that two-step "substantially reduces early divergence" in 2D is withdrawn. The like-for-like comparison for the finetune phase is clean-only (+18.6% best->last) versus two-step Step 2 (+13.0%), a small difference; the +79% to +92% figures belong to full-noisy training.
(5) Candidate explanation about the Step 2 learning rate (1e-4 for 2D, a 10x reduction from Step 1) stays as untested.

## XGBoost — clean-only, sample-weight, full-noisy

All three notebooks score against the same shared clean test split
(`load_clean_data(CLEAN_PATH, split='test')`, 1,139 rows) used
throughout this ablation — confirmed already consistent, no fix
needed. Unlike the CNN notebooks, XGBoost has no two-step variant in
this ablation; only single-step (v3) vs. single-step + class-balanced
`sample_weight` (v4) are compared.

| condition | notebook | val KL | val F1 | best round (of 2000) |
|---|---|---|---|---|
| Clean baseline | xgboost-v3-clean-data.ipynb | 0.8409 | 0.2002 | 49 |
| + sample_weight (class-balanced) | xgboost-v4-sample-weight.ipynb | 0.8938 | 0.2676 | 232 |
| Full noisy | xgboost-v5-full-noisy.ipynb | 0.9964 | 0.2782 | 132 |

Per-class F1:

| class | v3 | v4 | v5 |
|---|---|---|---|
| seizure | 0.1951 | 0.2913 | 0.2768 |
| lpd | 0.0581 | 0.1214 | 0.2420 |
| gpd | 0.2105 | 0.3741 | 0.2314 |
| lrda | 0.0000 | 0.0000 | 0.0964 |
| grda | 0.0611 | 0.1422 | 0.2029 |
| other | 0.6762 | 0.6768 | 0.6197 |

**Interpretation**: `sample_weight` raises macro F1 substantially
(+34% relative) with gains concentrated in minority classes (lpd, gpd,
grda all roughly double), but val KL gets *worse* (+6% relative), not
better. This is a genuine trade-off, not a free improvement: a third,
architecturally unrelated model family (gradient-boosted trees, vs.
EEGNet and EfficientNet-B0 for the CNNs) independently confirms that
reweighting alone improves hard-classification accuracy at a real cost
to soft-label calibration — consistent with, and sharper than, the 1D
CNN's Step 2a→2b comparison (where KL was roughly flat rather than
measurably worse). lrda scores 0.0000 F1 in both clean-only conditions but 0.0964 when training on the full noisy data (v5), so the failure is data-limited (4,800 rows) rather than a hard limit of the feature set.

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

## Multi-seed status

Seeds 43 and 44 are pending for all rows above; results will be reported as mean +/- SD with per-seed values.
