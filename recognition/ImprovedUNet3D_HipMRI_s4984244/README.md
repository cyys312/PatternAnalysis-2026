# 3D Improved U-Net for Prostate MRI Segmentation (HipMRI Study)

**Author:** Wenyu Cai (s4984244) · COMP3710 Pattern Recognition, 2026 · Difficulty: Hard

> Work in progress. Sections below are filled in as the project develops.

## Problem and Algorithm

## Feasibility Review

*One-page review for the midpoint check-off. Draft of 5 Oct 2026; items marked ⏳ are
filled in from the first full training runs.*

**1. User need, scope and acceptance criteria.** The user is a radiotherapy planner who
wants draft contours of the prostate and nearby organs at risk (bladder, rectum) plus body
and bone on pelvic MR, to review and correct rather than draw from scratch. The prototype must
show *where* a 3D model is safe to trust, not only its average Dice. It is accepted if:

| # | Criterion (held-out test set: 6 patients, 34 scans) |
|---|---|
| A1 | 3D Improved U-Net reaches Dice ≥ 0.70 on **each** of the 5 foreground classes. |
| A2 | Dilemma: on identical voxels, 3D context counts as a real gain only if prostate Dice in the apex and base thirds of the gland improves on the 2D U-Net by ≥ 0.05, or prostate HD95 drops (paired Wilcoxon over patients). |
| A3 | Prostate leaking into rectum and bladder (mL) and onto slices beyond the true apex/base is reported for both models; the recommendation does not rest on global Dice. |
| A4 | 3D training fits in ≤ 16 GB VRAM and ≤ 3 h on one A100; inference ≤ 1 s per volume; the 3D/2D cost ratio is reported. |
| A5 | No patient appears in two subsets; the split (`splits.json`) and seed are committed. |

**2. Model choice and course concepts.** 3D Improved U-Net (Isensee et al., 2018; Hard):
an encoder–decoder with skip connections, pre-activation residual context modules (dropout 0.3),
instance norm (batch size is only 2) and three summed deep-supervision heads; 9.47 M
parameters. The baseline is a 2D U-Net (Ronneberger et al., 2015; 7.76 M parameters)
trained on axial slices of the *same* volumes and restacked for evaluation, so the only
difference is whether the convolutions see neighbouring slices. Loss is cross-entropy +
foreground soft Dice for the severe class imbalance. Course concepts used: receptive fields
of 2D vs 3D convolutions, skip connections, residual learning, normalisation, loss design for
imbalance, and leakage-free evaluation.

**3. Preliminary evidence.**
*Data audit (Rangpur):* 211 scans from 38 patients (1–8 each); 256×256×128 voxels at
≈1.68×1.68×1.56 mm; prostate is 0.10 % and rectum 0.14 % of voxels; the prostate spans
17–41 axial slices (median 25). Volumes are resampled to 128×128×128, keeping every acquired
slice so the apex/base slices survive. Split stratified by scans per patient: 26/6/6 patients
(143/34/34 scans).
*Smoke tests:* the full pipeline (train → test → evaluation figures) runs end to end on
synthetic NIfTI data; on real data, 3 epochs on an A100 already give test Dice body 0.97,
bone 0.74, bladder 0.77, prostate 0.51, rectum 0.15.
*Measured cost:* 3D at 128³, batch 2: **4.0 GB** peak VRAM, **14 s/epoch** on an A100
(≈ 70 min for 300 epochs). On an RTX 4060, the 3D model needed 4.0 GB and 0.43 s per step, and
the 2D model 0.9 GB and 0.07 s per step (batch 32) — 4.4× less memory.
⏳ Full-run Dice of both models and 2D wall time.

**4. Risks, budget and fallback.**

- *Tiny structures* (rectum, prostate) may miss 0.70 at half in-plane resolution → train at
  the native 128×256×256 (≈ 16 GB, fits an A100) and/or weight the cross-entropy.
- *Few patients* (26 for training; weekly scans are near-duplicates) → augmentation,
  dropout, model selection on validation patients only.
- *Low statistical power* (6 test patients) → report effect sizes and per-scan tests as well.
- *Shared-cluster queues* → short time limits, cache built on a CPU node, 20-min `a100-test`
  partition for smoke tests.

*Budget:* ≈ 1.2 A100-hours per 3D run; at most ~10 runs (≈ 15 GPU-hours) including tuning.
*Next experiment:* full 300-epoch 3D and 50-epoch 2D runs (queued), then the zone-wise
evaluation in `predict.py`. *Fallback:* if the 3D model still misses A1 by 22 Oct, switch to a
2D Improved U-Net (Normal) on the same pipeline; the 2D-vs-3D question can still be answered.

## Dataset and Pre-processing

## Usage

## Results

## Open Research Dilemma: Spatial Context vs. Clinical Boundary Utility

## Artificial Intelligence Usage Disclosure

## References
