# E2 128-to-25 Fusion Variant

Date: 2026-06-08

This is an exploratory E2 classical fusion variant. It is not canonical unless
manually promoted later.

## Run

```text
folder: e2_canonical_128_25
run: e2_canonical_128_25
checkpoint: e2_canonical_128_25/checkpoints/best_e2_canonical_128_25.pt
metrics: e2_canonical_128_25/results/e2_canonical_128_25_metrics.json
```

Architecture:

```text
Linear(1344,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128,25)
```

This removes the 4D bottleneck used in canonical E2:

```text
Linear(1344,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128,4) -> ReLU -> Linear(4,25)
```

## Training

```text
seed: 42
device: cuda
batch size: 512
epochs requested: 60
early stopping patience: 8
best epoch: 15
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-4
loss: weighted cross entropy
scheduler: ReduceLROnPlateau on validation Macro F1
trainable params: 175,641
```

## Metrics

| Split | Accuracy | Macro F1 | Macro Recall | Early Recall | Mid Recall | Severe Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Val | 0.991527 | 0.981353 | 0.980022 | 0.987562 | 0.948454 | 0.885714 | 0.005478 |
| Test | 0.988035 | 0.973366 | 0.971059 | 0.981161 | 0.938636 | 0.785714 | 0.007645 |

## Comparison Notes

Against canonical E2:

```text
canonical E2 test Macro F1:      0.967926
e2_canonical_128_25 test Macro F1: 0.973366
canonical E2 test ECE:           0.006775
e2_canonical_128_25 test ECE:    0.007645
```

Against E2-matched-8:

```text
E2-matched-8 test Macro F1:      0.973356
e2_canonical_128_25 test Macro F1: 0.973366
E2-matched-8 test ECE:           0.248338
e2_canonical_128_25 test ECE:    0.007645
```

Interpretation:

- The 128-to-25 head is the strongest classical predictive E2 variant observed
  so far by a very small Macro F1 margin.
- It preserves calibration close to canonical E2 and far better than the
  parameter-matched bottleneck comparators.
- It is not a fair E3 comparator because it removes the low-dimensional
  bottleneck required for VQC comparison.
- It is a useful practical classical reference if the thesis discusses whether
  the 4D bottleneck in canonical E2 was necessary.
