# E1 IoT-Only Baseline Training Report

Last updated: 2026-06-01

## Purpose

E1 is the IoT-only baseline for the thesis. It measures how much crop-disease-stage signal exists in the Dacon environmental sensor streams without using images.

E1 is compared against:

```text
E0: image-only baseline
E2: classical image + IoT fusion
E3: hybrid quantum-classical multimodal model
```

## Environment

All E1 work used:

```text
G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe
```

Verified packages:

```text
pandas 2.3.3
scikit-learn 1.7.2
torch 2.11.0+cu128
CUDA available: true
GPU: NVIDIA GeForce RTX 5070
```

CUDA was used, but GPU utilization stayed low because E1 is a small tabular MLP.

## Dataset

Main dataset:

```text
G:\FOR THESIS\MAIN\Data
```

Original manifests:

```text
Data\train.csv: 51,906 samples
Data\test.csv:   5,767 samples
```

Label format:

```text
crop_disease_stage
example: 3_b7_1
```

Stage mapping:

```text
0 = healthy
1 = early
2 = mid
3 = severe
```

## Shared Split

For coherence with Abrar's E0 image-only baseline, E1 uses Abrar's fixed split:

```text
G:\FOR THESIS\MAIN\Data\train.csv
G:\FOR THESIS\MAIN\Data\val.csv
G:\FOR THESIS\MAIN\Data\test.csv
```

Split sizes:

```text
train: 46,713 rows
val:    5,193 rows
test:   5,767 rows
```

All split files use:

```text
image,label
```

Overlap checks:

```text
train-val overlap: 0
train-test overlap: 0
val-test overlap: 0
```

## Feature Extraction

Script:

```text
scripts\extract_iot_features.py
```

Input files:

```text
Data\train\{image_id}\{image_id}_final.csv
Data\test\{image_id}\{image_id}_final.csv
```

Numeric IoT columns:

```text
int_temp_1_avg
int_temp_1_max
int_temp_1_min
int_hum_1_avg
int_hum_1_max
int_hum_1_min
int_dew_point_avg
int_dew_point_max
int_dew_point_min
```

Statistics per column:

```text
mean
std
min
max
median
last
```

Total features:

```text
9 numeric columns x 6 statistics = 54 features
```

Metadata columns:

```text
image
label
crop
disease
stage
n_rows
```

Missing-value handling:

```text
forward fill
backward fill
column median fill
remaining nulls filled with 0
```

Generated feature tables:

```text
features\train_iot_features.csv: 51,906 rows x 60 columns
features\test_iot_features.csv:   5,767 rows x 60 columns
```

Validation:

```text
train NaNs: 0
test NaNs: 0
labels: 25
Abrar train coverage: 46,713 / 46,713
Abrar val coverage:   5,193 / 5,193
Abrar test coverage:  5,767 / 5,767
```

## Training Script

Script:

```text
scripts\train_e1_iot_mlp.py
```

Training behavior:

```text
loads full train/test IoT feature tables
selects train/val/test rows using Abrar split CSVs
checks labels match Abrar split order
fits LabelEncoder on Abrar train only
fits MinMaxScaler on Abrar train only
transforms validation and test with train-fitted scaler
uses weighted cross entropy
checkpoints by validation Macro F1
reports validation and test metrics
```

Checkpoint evaluation helper:

```text
scripts\evaluate_e1_checkpoint.py
```

This helper reloads an existing checkpoint and recomputes metrics without retraining. It was used to add ROC-AUC metrics to the best run after training.

## Best Model Config

Best run:

```text
e1_h128_64_nodrop_wd_bs512_sched
```

Config:

```text
seed: 42
batch size: 512
max epochs: 60
early stopping patience: 8
best epoch: 36
input dim: 54
hidden layers: 128, 64
dropout: 0.0, 0.0
output dim: 25
optimizer: Adam
learning rate: 1e-3
weight decay: 1e-4
loss: weighted cross entropy
scheduler: ReduceLROnPlateau
scheduler monitor: validation Macro F1
scheduler mode: max
scheduler patience: 3
scheduler factor: 0.5
scaler: MinMaxScaler
trainable parameters: 17,305
device: cuda
```

Command:

```powershell
cd "G:\FOR THESIS\MAIN\e1_iot_only"

& "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe" .\scripts\train_e1_iot_mlp.py `
  --run-name e1_h128_64_nodrop_wd_bs512_sched `
  --seed 42 `
  --epochs 60 `
  --patience 8 `
  --batch-size 512 `
  --lr 1e-3 `
  --hidden1 128 `
  --hidden2 64 `
  --dropout1 0.0 `
  --dropout2 0.0 `
  --weight-decay 1e-4 `
  --scheduler `
  --scheduler-patience 3 `
  --scheduler-factor 0.5
```

## Best Results

Validation metrics:

```text
accuracy: 0.943963
Macro F1: 0.887618
macro recall: 0.889753
macro ROC-AUC OVR: 0.997332
weighted ROC-AUC OVR: 0.998732
early-stage recall: 0.932007
ECE: 0.016533
```

Test metrics:

```text
accuracy: 0.943645
Macro F1: 0.883370
macro recall: 0.882642
macro ROC-AUC OVR: 0.997505
weighted ROC-AUC OVR: 0.998793
early-stage recall: 0.939714
ECE: 0.016543
```

Per-stage test recall:

```text
stage 0: 1.000000
stage 1: 0.939714
stage 2: 0.884091
stage 3: 0.482143
```

Interpretation:

```text
E1 is a strong IoT-only baseline on this split.
The severe-stage recall is lower, consistent with severe-stage class scarcity.
The result should be compared directly with Abrar's E0 using the same split.
```

## Optimization Runs

Run comparison file:

```text
results\e1_run_comparison.csv
```

Completed runs:

```text
e1_baseline_bs512_sched
e1_h128_64_bs512_sched
e1_h128_64_lowdrop_bs512_sched
e1_h128_64_nodrop_wd_bs512_sched
e1_h256_128_nodrop_wd_bs512_sched
e1_h128_64_nodrop_wd_bs512_lr5e4_sched
```

Summary ordered by validation Macro F1:

```text
run                                      val_macro_f1  test_macro_f1  test_acc
e1_h128_64_nodrop_wd_bs512_sched          0.887618      0.883370      0.943645
e1_h128_64_nodrop_wd_bs512_lr5e4_sched    0.884022      0.878773      0.939830
e1_h256_128_nodrop_wd_bs512_sched         0.878175      0.878309      0.939657
e1_h128_64_bs512_sched                    0.876318      0.879241      0.941044
e1_h128_64_lowdrop_bs512_sched            0.866303      0.867663      0.933241
e1_baseline_bs512_sched                   0.857443      0.853586      0.931160
```

## Output Artifacts

Best checkpoint:

```text
checkpoints\best_e1_h128_64_nodrop_wd_bs512_sched.pt
```

Best metrics:

```text
results\e1_h128_64_nodrop_wd_bs512_sched_metrics.json
```

Training history:

```text
results\e1_h128_64_nodrop_wd_bs512_sched_history.csv
```

Predictions:

```text
results\e1_h128_64_nodrop_wd_bs512_sched_val_predictions.csv
results\e1_h128_64_nodrop_wd_bs512_sched_test_predictions.csv
```

Confusion matrix:

```text
results\confusion_matrices\e1_h128_64_nodrop_wd_bs512_sched_test_confusion_matrix.csv
```

Evaluation helper:

```text
scripts\evaluate_e1_checkpoint.py
```

All-run comparison:

```text
results\e1_run_comparison.csv
```

## Notes For Thesis Writing

Suggested wording:

```text
The IoT-only baseline converts each environmental sensor time series into a fixed-length statistical summary vector. For each of the nine selected sensor channels, six descriptive statistics are computed, producing a 54-dimensional feature vector. A lightweight multilayer perceptron is trained using weighted cross entropy and a train-only MinMax scaler. The baseline is evaluated on the same fixed train/validation/test split used by the image-only baseline to preserve comparability.
```

Important caveat:

```text
High E1 performance may indicate strong environmental signal, but it may also reflect dataset-specific collection or class-condition correlations. Compare with E0 and later E2 to decide whether multimodal fusion adds value.
```

## Next Steps

1. Get Abrar's E0 metrics on the same split.
2. Compare E0 vs E1:

```text
accuracy
Macro F1
early-stage recall
per-stage recall
confusion matrix patterns
parameter count
```

3. Use E1 as the sensor-only floor for E2 multimodal fusion.
4. Keep this report updated if E1 is rerun.
