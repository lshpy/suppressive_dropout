> **Archived early prototype.** Kept for history only. The maintained Suppressive Dropout (SDrop) code is at https://github.com/lshpy/sdrop.

# Suppressive Dropout (CNN, one-spot)

First prototype of Suppressive Dropout: a small CNN on CIFAR-10 with a channel-wise Suppressive Dropout layer inserted at **one point only**, after the middle stage.

## What it does

- Score per channel (x_j = spatial mean activation of channel j):
  S_j ∝ (Σ_{k≠j} x_k) * x_j^2 / (1 + b Σ_i x_i^2)^{c+1}
- During training, the `drop_ratio` fraction of channels with the largest S_j is zeroed per sample (identity at eval time).
- 3-stage CNN (`model/cnn.py`), Adam + StepLR, 80/20 train/validation split of the CIFAR-10 training set.
- Logs accuracy, macro-F1, AUC and ECE per epoch on validation, then reports the test set.

## How to run

```bash
pip install -r requirements.txt
python main.py --use_sdrop --drop_ratio 0.2 --b 1.0 --c 1.0 --epochs 30
# baseline
python main.py --epochs 30
```

Outputs: per-epoch CSV and test summary JSON under `results/cnn_sdrop/`, model weights under `checkpoints/`.

## Layout

```
main.py                            entry point (argparse, data loaders, training loop)
train.py / evaluate.py             one training epoch / evaluation
experiments/suppressive_dropout.py SuppressiveDropout layer
model/cnn.py                       3-stage CNN with one SDrop insertion point
utils/                             metrics (acc, F1, AUC, ECE) and logging
```

## Status

Early prototype; ideas noted for later were spatial (HxW) suppression instead of channels and multiple insertion points / ablation switches. No results are included in this repository.

More projects: https://github.com/lshpy
