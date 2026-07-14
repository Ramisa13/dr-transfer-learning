# Diabetic Retinopathy Detection via Transfer Learning

Transfer learning pipeline for diabetic retinopathy (DR) grading on the
DeepDRiD dataset. Covers pretrained-model fine-tuning, two-stage training,
a self-attention augmented ResNet, model/ensemble comparison, and
Grad-CAM based explainability.

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data](#data)
- [Usage](#usage)
  - [Task A — Baseline fine-tuning](#task-a--baseline-fine-tuning)
  - [Task B — Two-stage training](#task-b--two-stage-training)
  - [Task C — Self-attention](#task-c--self-attention)
  - [Task D — Model comparison / ensembling](#task-d--model-comparison--ensembling)
  - [Task E — Visualization & Grad-CAM](#task-e--visualization--grad-cam)
- [Results](#results)
- [Repository Notes](#repository-notes)

## Overview
Large labeled medical imaging datasets are expensive to obtain, so this
project fine-tunes ImageNet-pretrained CNNs (ResNet-18/50) on the DeepDRiD
dataset for 5-class DR severity grading (0–4), then improves on that
baseline with:

1. **Two-stage training** — pretrain on an auxiliary DR dataset, then
   fine-tune on DeepDRiD.
2. **Self-attention** — a lightweight self-attention block inserted after
   the CNN backbone's feature map, so the model can relate spatially
   distant lesion regions to each other.
3. **Ensembling** — combining several task-B checkpoints (different
   hyperparameters / preprocessing) via max-voting, weighted averaging,
   and stacking.
4. **Explainability** — Grad-CAM heatmaps overlaid on the fundus images to
   verify the model attends to clinically relevant regions.

Models are compared using accuracy, precision, recall, F1, and Cohen's
quadratic-weighted kappa (the standard DR-grading metric, since class
imbalance makes accuracy misleading).

## Project Structure
```
dr-transfer-learning/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml            # all hyperparameters in one place
├── src/
│   ├── dataset.py              # RetinopathyDataset + transforms
│   ├── preprocessing.py        # Ben Graham, CLAHE, circular crop, etc.
│   ├── models.py                # ResNet baseline, SelfAttention block, dual-input model
│   ├── train.py                  # training loop, used for tasks A/B/C
│   ├── evaluate.py              # metrics + test-set prediction CSV writer
│   ├── ensemble.py               # max-voting, weighted average, stacking
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   └── utils.py                  # seeding, checkpointing, logging helpers
├── scripts/
│   ├── run_task_a.sh
│   ├── run_task_b.sh
│   ├── run_task_c.sh
│   ├── run_task_d.sh
│   └── run_task_e.sh
├── notebooks/
│   └── exploration.ipynb        # optional EDA / quick visual checks
└── assets/
    └── (sample Grad-CAM outputs, plots — gitignored by default)
```

## Setup
```bash
git clone https://github.com/<your-username>/dr-transfer-learning.git
cd dr-transfer-learning
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x, CUDA 11.8 (also runs on CPU/Colab
free-tier GPU, just slower).

## Data
1. Download the DeepDRiD dataset from the course Kaggle competition page
   and unzip it into `data/` (this folder is gitignored — do not commit
   patient images).
2. Expected structure:
   ```
   data/
   ├── train.csv          # image_id, patient_id, img_path, label
   ├── val.csv
   ├── test.csv
   └── images/
   ```
3. Set the `data_root` field in `configs/config.yaml` to point at this
   folder.
4. **Do not commit any dataset files or trained weights** — see
   `.gitignore`. Share checkpoints via a cloud link instead (see below).

## Usage
Every task can be run either via the shell scripts in `scripts/` or by
calling `src/train.py` / `src/evaluate.py` directly with flags.

### Task A — Baseline fine-tuning
Fine-tunes ImageNet-pretrained ResNet-18 on DeepDRiD, freezing the backbone
and training only the classification head first.
```bash
bash scripts/run_task_a.sh
# equivalent:
python src/train.py --config configs/config.yaml --task a --freeze_backbone
```

### Task B — Two-stage training
Stage 1 trains on an auxiliary DR dataset (e.g. APTOS 2019); stage 2
fine-tunes the resulting weights on DeepDRiD, optionally unfreezing all
layers.
```bash
bash scripts/run_task_b.sh
python src/train.py --config configs/config.yaml --task b \
    --stage1_data aptos --stage2_data deepdrid --unfreeze_all
```

### Task C — Self-attention
Inserts the `SelfAttention` block from `src/models.py` after the ResNet
feature extractor before the classification head.
```bash
bash scripts/run_task_c.sh
python src/train.py --config configs/config.yaml --task c --use_attention
```

### Task D — Model comparison / ensembling
Loads three task-B checkpoints trained with different hyperparameters and
combines their predictions.
```bash
bash scripts/run_task_d.sh
python src/ensemble.py --checkpoints ckpts/model_b1.pth ckpts/model_b2.pth ckpts/model_b3.pth \
    --method max_voting --preprocess clahe
```
Supported `--method` values: `max_voting`, `weighted_average`, `stacking`,
`bagging`, `boosting`.

### Task E — Visualization & Grad-CAM
Plots training/validation loss and accuracy curves, and generates Grad-CAM
overlays for a sample of test images.
```bash
bash scripts/run_task_e.sh
python src/gradcam.py --checkpoint ckpts/best_model.pth --num_samples 10 --out_dir assets/gradcam
```

## Results
Kappa on the DeepDRiD test set (ResNet-18 backbone unless noted):

| Model / Strategy                              | Kappa  |
|------------------------------------------------|--------|
| Baseline (ImageNet fine-tune, head only)        | 0.7995 |
| Two-stage (fine-tuned on DR-resized)            | 0.8178 |
| Two-stage, all layers unfrozen                  | 0.8631 |
| Self-attention                                  | —      |
| Max-voting ensemble + CLAHE                     | 0.8354 |

*(Fill in blanks with your latest run numbers before publishing — these
are carried over from the original course submission and should be
re-verified against `test_predictions_*.csv`.)*

## Repository Notes
- Trained checkpoints are **not** committed to this repo (large binary
  files, and often subject to dataset redistribution restrictions).
  Instead, upload `.pth` files to Google Drive / institutional storage and
  link them here.
- `configs/config.yaml` is the single source of truth for hyperparameters
  — prefer editing it over hardcoding values in scripts.
- Contributions: see commit history / `CONTRIBUTORS.md` for individual
  task ownership if working as a group.

## License
MIT — see `LICENSE`. Adjust if your course or dataset license requires
otherwise (DeepDRiD has its own terms of use; keep the dataset itself out
of the repo regardless of code license).
