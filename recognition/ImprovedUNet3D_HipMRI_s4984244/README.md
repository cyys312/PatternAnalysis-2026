# 3D Improved U-Net for Prostate MRI Segmentation (HipMRI Study)

**Author:** Wenyu Cai (s4984244) · COMP3710 Pattern Recognition, 2026 · Difficulty: Hard

> Work in progress: sections marked ⏳ are filled in once the full training runs finish.

## Problem and Algorithm

**The problem.** External-beam radiotherapy for prostate cancer needs the prostate and the
organs at risk around it delineated on every planning scan, so the dose can cover the gland while
sparing the bladder and rectum. Drawing these contours by hand is slow, and the HipMRI Study
re-scanned most patients every week of treatment. This project segments pelvic MR volumes into
six classes — background, body, bone, bladder, rectum and prostate — so that a planner only has
to review and correct a draft. It is hard for three reasons: the prostate occupies just 0.10 % of
the voxels; its soft-tissue contrast with the bladder neck (base) and pelvic floor (apex) is
weak; and it sits a few millimetres from the rectal wall, where over-contouring causes toxicity.

**How the model works.** The 3D Improved U-Net (Isensee et al. [1]) is an encoder–decoder
that labels every voxel of the whole 128 × 128 × 128 volume at once (figure below). The encoder
halves the resolution four times with stride-2 3 × 3 × 3 convolutions; each level adds a
residual *context module* (two convolutions with dropout, added back to the input), so deep
levels see the whole pelvis while training stays stable. The decoder upsamples step by step,
concatenates the encoder features of the same resolution through skip connections to recover
precise boundaries, and merges them in *localisation modules*. Segmentation maps taken from
three decoder levels are upscaled and summed (*deep supervision*), which injects coarse,
context-rich predictions into the final output. Instance normalisation [4] is used because the
batch holds only two volumes, and the loss is cross-entropy plus soft Dice over the foreground
classes [3], which keeps the tiny prostate and rectum from being swamped by the background.
Because every 3 × 3 × 3 kernel also sees the slices above and below, the model can use
through-plane continuity that a slice-by-slice 2D U-Net [2] — the baseline here — cannot. Whether
that context actually fixes the clinically dangerous apex and base slices is the question this
project investigates.

![3D Improved U-Net architecture](images/architecture.png)

## Feasibility Review

*One-page review for the midpoint check-off, updated 6 Oct 2026 with the first full training
runs.*

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
synthetic NIfTI data and on 3 epochs of real data.
*First full runs (test set, one A100):* the 3D model reaches Dice body 0.99, bone 0.92,
bladder 0.95, rectum 0.88, prostate 0.87, so **A1 is met**; the 2D baseline gets 0.99, 0.92,
0.94, 0.85, 0.87. Cost: 3D 1.1 h training (300 epochs) and 4.0 GB peak VRAM vs 2D 0.23 h and
0.9 GB, i.e. ≈ 5× the compute; inference takes 0.025 s vs 0.022 s per volume (A4 met).
*Early answer to the dilemma:* both models fall from 0.91 Dice in the mid-gland to 0.76–0.78
at the apex and base, and 3D context does not change that (apex −0.02, base +0.02, HD95
−0.1 mm; none significant), so **A2 is not met**. Its one consistent gain is the rectum
(+0.03 Dice, better in all 6 test patients, p = 0.03).

**4. Risks, budget and fallback.**

- *Apex and base stay weak for both models* — the limit may be the 3.4 mm in-plane voxels or
  ambiguous ground truth rather than missing context → next experiment below.
- *Few patients* (26 for training; weekly scans are near-duplicates) → augmentation,
  dropout, model selection on validation patients only.
- *Low statistical power* (6 test patients) → report effect sizes and per-scan tests as well.
- *Shared-cluster queues* → short time limits, cache built on a CPU node, 20-min `a100-test`
  partition for smoke tests.

*Budget:* 1.1 A100-hours per 3D run at 128³ (best epoch ≈ 100 of 300, so runs can be
shortened); at most ≈ 15 GPU-hours in total. *Next experiment:* train both models at the
native 128 × 256 × 256 resolution (an estimated 16 GB for 3D) to test whether resolution, not context,
limits the apex and base; then autopsy the worst apex/base cases. *Fallback:* A1 is already
met at 128³, so the current models are the fallback if the high-resolution runs do not fit
the budget or the queue.

## Dataset and Pre-processing

**Data.** HipMRI Study [7, 8], on Rangpur at `/home/groups/comp3710/HipMRI_Study_open`: pelvic MR
volumes in `semantic_MRs/<patient>_Week<n>_LFOV.nii.gz` with labels in
`semantic_labels_only/<patient>_Week<n>_SEMANTIC.nii.gz`. `python dataset.py` audits it:

| Property | Value |
|---|---|
| Scans / patients | 211 / 38 — 22 patients with 8 weekly scans, 11 with 1, five with 2, 4, 5, 6 and 7 |
| Size (x, y, z) | 256 × 256 × 128 (one scan has 144 slices) |
| Voxel size | 1.41–1.88 mm in-plane (mostly 1.68), 1.56 mm slices; all LPS orientation |
| Labels | 0 background 60.35 %, 1 body 35.46 %, 2 bone 3.37 %, 3 bladder 0.57 %, 4 rectum 0.14 %, 5 prostate 0.10 % of voxels; no out-of-range values; every scan contains prostate |
| Prostate extent | 17–41 axial slices (median 25) |

**Pre-processing** (`dataset.py`), run once and cached as a single tensor file:

1. *Intensity:* each volume is z-score normalised. MR intensities have no physical unit and vary
   with scanner gain, so only relative contrast within a scan is meaningful.
2. *Resampling to 128 × 128 × 128 (D × H × W):* the 128 acquired slices are kept and the in-plane
   grid is halved (voxels ≈ 3.4 mm in-plane, 1.56 mm between slices). With a median of 25 prostate slices, halving the
   slices as well would leave about four slices each for the apex and base — the very slices
   the research question is about. At this size the 3D model trains in 4 GB of GPU memory.
3. *Labels* are resampled by averaging their one-hot encoding over each output voxel and taking
   the arg-max, so a thin structure such as the rectal wall keeps the voxels it dominates;
   nearest-neighbour sampling would drop or keep it depending on where the grid points fall.

**Split.** Patients, not scans, are divided into train / validation / test (70 / 15 / 15 %,
seed 3710). A patient's weekly scans are nearly identical, so a scan-level split would put the
same anatomy in training and test and inflate the scores. Because scan counts per patient range
from 1 to 8, single-scan and multi-scan patients are split separately; a plain random split
gave test sets of 11 to 48 scans depending on the seed. The split is committed as
[`splits.json`](splits.json) so every run and every reviewer uses the same test patients:

| Subset | Patients | Scans |
|---|---|---|
| Train | 26 | 143 |
| Validation | 6 | 34 |
| Test | 6 (C032, M013, R016, R024, S035, T009) | 34 |

**Augmentation** (training only, applied identically to image and label): a left–right flip
(p = 0.5); a rotation of up to ±10° and a scaling by 0.9–1.1, both within the axial plane (p = 0.5); an
intensity gain of 0.9–1.1 and offset of ±0.1; and Gaussian noise with σ = 0.05 (p = 0.2). The
pelvis is roughly left–right symmetric, but anterior–posterior or superior–inferior flips would
put the rectum in front of the prostate or the bladder below it, so they are not used.

**2D baseline data.** The 2D U-Net is trained on the axial slices of the same cached volumes
(not the separate `keras_slices_data` set), and its slice predictions are stacked back into
volumes for evaluation. Both models are therefore scored on exactly the same patients and
voxels.

## Usage

**Files.**

| File | Contents |
|---|---|
| `modules.py` | `ImprovedUNet3D`, the `UNet2D` baseline and `DiceCELoss` (pure PyTorch) |
| `dataset.py` | file pairing, patient split, pre-processing and cache, augmentation, datasets, data audit |
| `utils.py` | model factory, checkpoint loading, whole-volume inference for both models, Dice / IoU |
| `train.py` | training, per-epoch validation, testing of the best checkpoint, loss and Dice curves |
| `predict.py` | evaluation of trained models on the test patients and all figures in this README |
| `slurm/*.sh` | Rangpur batch jobs: `prepare_data.sh` (CPU), `train.sh` and `predict.sh` (one A100) |
| `splits.json` | the committed patient split |

**Dependencies.** Versions used for all reported results (conda environment on Rangpur,
NVIDIA A100-PCIE-40GB):

| Package | Version |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.13.0 (CUDA 13.0 build) |
| nibabel | 5.4.2 |
| NumPy | 2.4.6 |
| SciPy | 1.17.1 |
| Matplotlib | 3.11.1 |

`pip install torch nibabel scipy matplotlib` is enough. The model code (`modules.py`) uses only
PyTorch; NumPy and SciPy are used in `predict.py` for evaluation and plotting.

**Running on Rangpur.** From this folder (Slurm does not create the log folder itself):

```bash
mkdir -p logs
sbatch slurm/prepare_data.sh                     # audit, splits.json and cache on a CPU node
sbatch -J unet2d slurm/train.sh --model unet2d   # 2D U-Net baseline
sbatch -J unet3d slurm/train.sh --model unet3d   # 3D Improved U-Net
sbatch slurm/predict.sh --checkpoints runs/unet3d/best.pt runs/unet2d/best.pt
```

The scripts forward their arguments to the Python files, which can also be run directly, e.g.
`python train.py --model unet3d --root <HipMRI_Study_open>`. The batch scripts set
`--account=comp3710` (required by the `comp3710` partition) and deliberately request no
`--mem`, which Rangpur's nodes can never satisfy.

| `train.py` option | Default | Meaning |
|---|---|---|
| `--model` | `unet3d` | `unet3d` or `unet2d` |
| `--epochs`, `--batch-size`, `--lr` | 300, 2, 5e-4 (3D); 50, 32, 1e-3 (2D) | optimisation |
| `--shape` | `128x128x128` | volume size D × H × W after resampling (multiples of 16) |
| `--root`, `--cache`, `--out` | Rangpur data path, `cache`, `runs` | input, pre-processed cache, outputs |
| `--weight-decay`, `--seed`, `--workers` | 1e-5, 3710, 4 | |
| `--no-amp` | off | disable mixed precision |
| `--limit N` | – | use only N scans per subset, for smoke tests |

Training uses AdamW [5] with cosine learning-rate decay [6] and mixed precision. After every epoch the
model segments each whole validation volume, and the checkpoint with the highest mean foreground
Dice is kept as `best.pt`. `train.py` writes `best.pt`, `last.pt`, `history.json`, `curves.png`
(loss and per-class validation Dice) and `test_metrics.json` to `runs/<model>/`. `predict.py`
writes the figures and `results.json` to `images/`; with two checkpoints, the first is treated as
the main model.

**Reproducibility.** The split is fixed by `splits.json`, the seed by `--seed`, and the cache
is rebuilt automatically if the shape or the number of scans changes. GPU convolutions and mixed
precision are not bit-for-bit deterministic, so repeated runs differ slightly. A quick check
of the whole pipeline takes a few minutes:
`python train.py --model unet3d --epochs 1 --limit 2`.

## Results

⏳ Filled in from the full training runs: training curves of both models, per-class test Dice
and IoU, example inputs and outputs (axial and sagittal), and the resource table (peak VRAM,
parameters, inference latency, training time).

## Open Research Dilemma: Spatial Context vs. Clinical Boundary Utility

⏳ Prostate Dice from apex to base, leakage into the rectum and bladder, HD95, paired Wilcoxon
tests, autopsies of 3–5 failure cases, and the recommendation to the project manager.

## Artificial Intelligence Usage Disclosure

*Draft — the author completes and confirms this section before submission.*

- **Tool:** Anthropic Claude (Claude Code, model Claude Opus 5.5). Commits it contributed to
  carry a `Co-Authored-By` trailer.
- **Tasks it assisted with:** drafting the Python modules and Slurm scripts, the data audit,
  a synthetic-data generator for smoke tests, diagnosing failed Rangpur jobs, the architecture
  figure, and drafting this README.
- **How its output was checked:**
  - the whole pipeline was first run end to end on synthetic NIfTI volumes with known anatomy;
  - the audit of the real data contradicted an assumption about file names (`Case_XXX_WeekY`),
    and the loader was corrected before any training;
  - the patient split was recomputed independently on a second machine and matched exactly;
  - the architecture figure and parameter counts were checked against `modules.py`, and the
    dataset references were checked against their DOIs;
  - ⏳ *[author: own line-by-line review of each file, what was changed as a result, and further
    checks]*.

## Acknowledgements

Data were provided by the HipMRI Study [7], acquired at Calvary Mater Newcastle Hospital in a
retrospective MRI-alone radiation therapy study [8], and used under its data use agreement
(non-commercial; no attempt to identify participants).

## References

1. F. Isensee, P. Kickingereder, W. Wick, M. Bendszus and K. H. Maier-Hein, "Brain tumor
   segmentation and radiomics survival prediction: Contribution to the BRATS 2017 challenge," in
   *Brainlesion: Glioma, Multiple Sclerosis, Stroke and Traumatic Brain Injuries*, 2018,
   pp. 287–297.
2. O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional networks for biomedical image
   segmentation," in *Proc. MICCAI*, 2015, pp. 234–241.
3. F. Milletari, N. Navab and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for
   volumetric medical image segmentation," in *Proc. 3DV*, 2016, pp. 565–571.
4. D. Ulyanov, A. Vedaldi and V. Lempitsky, "Instance normalization: The missing ingredient for
   fast stylization," arXiv:1607.08022, 2016.
5. I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. ICLR*, 2019.
6. I. Loshchilov and F. Hutter, "SGDR: Stochastic gradient descent with warm restarts," in
   *Proc. ICLR*, 2017.
7. J. Dowling and P. Greer, "Labelled weekly MR images of the male pelvis," v1, CSIRO Data
   Collection, 2021. https://doi.org/10.25919/45t8-p065
8. J. A. Dowling et al., "Automatic substitute computed tomography generation and contouring
   for magnetic resonance imaging (MRI)-alone external beam radiation therapy from standard MRI
   sequences," *Int. J. Radiat. Oncol. Biol. Phys.*, vol. 93, no. 5, pp. 1144–1153, 2015.
   https://doi.org/10.1016/j.ijrobp.2015.08.045
