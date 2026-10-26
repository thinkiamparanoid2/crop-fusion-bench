# E1: IoT-Only Baseline Report

## E1

E1 is the IoT-only baseline.

This model uses only the environmental sensor CSV files and does not use leaf images.

The goal is to check how much disease/class signal exists in the IoT data alone before doing multimodal fusion.

---

## Dataset Used

Main dataset:

```text
G:\FOR THESIS\MAIN\Data
```

Original dataset manifests:

```text
Data\train.csv = 51,906 samples
Data\test.csv  = 5,767 samples
```

Abrar's E0 split was used for coherence:

```text
G:\FOR THESIS\MAIN\Data\train.csv
G:\FOR THESIS\MAIN\Data\val.csv
G:\FOR THESIS\MAIN\Data\test.csv
```

Split sizes:

```text
Train:      46,713 samples
Validation:  5,193 samples
Test:        5,767 samples
```

Why this split:

```text
E0 and E1 must use the same train/validation/test split so the image-only and IoT-only baselines are comparable.
```

---

## Label Format

Labels follow:

```text
crop_disease_stage
```

Example:

```text
3_b7_1
```

Meaning:

```text
crop = 3
disease = b7
stage = 1
```

Stage meaning:

```text
0 = healthy
1 = early
2 = mid
3 = severe
```

Total classes:

```text
25
```

---

## Feature Extraction

Script:

```text
G:\FOR THESIS\MAIN\e1_iot_only\scripts\extract_iot_features.py
```

Each sample has one `_final.csv` sensor file:

```text
Data\train\{image_id}\{image_id}_final.csv
Data\test\{image_id}\{image_id}_final.csv
```

The timestamp column was kept in the CSV files but was not used as a direct model feature.

Selected numeric features:

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

For each numeric column, these statistics were calculated:

```text
mean
standard deviation
minimum
maximum
median
last value
```

So the final feature vector size is:

```text
9 sensor columns x 6 statistics = 54 features
```

Metadata columns kept:

```text
image
label
crop
disease
stage
n_rows
```

Final extracted feature files:

```text
features\train_iot_features.csv
features\test_iot_features.csv
```

Feature extraction output:

```text
train_iot_features.csv = 51,906 rows x 60 columns
test_iot_features.csv  = 5,767 rows x 60 columns
```

Null check:

```text
Train NaN count = 0
Test NaN count  = 0
```

---

## Missing Value Handling

Known issue:

```text
Some dew-point columns contain null values.
```

Handling method:

```text
1. Forward fill
2. Backward fill
3. Fill remaining nulls with column median
4. Fill any still-remaining nulls with 0
```

This ensured no missing values remained in the extracted feature table.

---

## Normalization

Normalization used:

```text
MinMaxScaler
```

Important rule:

```text
The scaler was fitted only on Abrar's E0 train split.
```

Then the same scaler was applied to:

```text
validation split
test split
```

Why MinMax normalization:

```text
The thesis pipeline later uses quantum angle encoding, where bounded feature values are useful.
MinMax normalization also matches the normalization logic described in the thesis report.
```

No validation or test information was used while fitting the scaler.

---

## Model Architecture

Best E1 model:

```text
Input dimension: 54
Hidden layer 1: 128
Hidden layer 2: 64
Output dimension: 25
```

Architecture:

```text
Linear(54, 128)
BatchNorm1d(128)
ReLU
Dropout(0.0)

Linear(128, 64)
BatchNorm1d(64)
ReLU
Dropout(0.0)

Linear(64, 25)
```

Trainable parameters:

```text
17,305
```

---

## Training Configuration

Best run name:

```text
e1_h128_64_nodrop_wd_bs512_sched
```

Training config:

```text
Seed: 42
Batch size: 512
Maximum epochs: 60
Early stopping patience: 8
Best epoch: 36
Optimizer: Adam
Learning rate: 1e-3
Weight decay: 1e-4
Loss: Weighted Cross Entropy
Scheduler: ReduceLROnPlateau
Scheduler patience: 3
Scheduler factor: 0.5
Checkpoint metric: Validation Macro F1
Device: CUDA
```

Why weighted cross entropy:

```text
The dataset is imbalanced, especially across disease-stage labels.
Weighted cross entropy reduces domination by majority classes.
```

Why validation Macro F1 for checkpointing:

```text
Macro F1 is the primary thesis metric and is more appropriate than accuracy for imbalanced classification.
```

---

## Things Tracked

During training:

```text
Training loss
Validation loss
Validation accuracy
Validation Macro F1
Validation macro recall
Validation early-stage recall
Validation ECE
Learning rate
```

After training:

```text
Test accuracy
Test Macro F1
Test macro recall
Test ROC-AUC
Test per-stage recall
Test early-stage recall
Test ECE
Confusion matrix
Prediction CSV
Trainable parameter count
Best epoch
Model checkpoint
```

---

## Final Validation Results

Best validation result:

```text
Validation Accuracy:              0.943963
Validation Macro F1:              0.887618
Validation Macro Recall:          0.889753
Validation Macro ROC-AUC OVR:     0.997332
Validation Weighted ROC-AUC OVR:  0.998732
Validation Early-Stage Recall:    0.932007
Validation ECE:                   0.016533
```

---

## Final Test Results

Final test result:

```text
Test Accuracy:              0.943645
Test Macro F1:              0.883370
Test Macro Recall:          0.882642
Test Macro ROC-AUC OVR:     0.997505
Test Weighted ROC-AUC OVR:  0.998793
Test Early-Stage Recall:    0.939714
Test ECE:                   0.016543
```

Per-stage test recall:

```text
Stage 0 healthy: 1.000000
Stage 1 early:   0.939714
Stage 2 mid:     0.884091
Stage 3 severe:  0.482143
```

Observation:

```text
The model performs strongly on healthy, early, and mid-stage samples.
Severe-stage recall is much lower, which is expected because severe-stage samples are very sparse.
```

---

## Optimization Runs

Several configurations were tested with batch size 512.

Run comparison:

```text
results\e1_run_comparison.csv
```

Summary:

```text
Run                                      Val Macro F1   Test Macro F1   Test Accuracy
e1_h128_64_nodrop_wd_bs512_sched          0.887618       0.883370        0.943645
e1_h128_64_nodrop_wd_bs512_lr5e4_sched    0.884022       0.878773        0.939830
e1_h256_128_nodrop_wd_bs512_sched         0.878175       0.878309        0.939657
e1_h128_64_bs512_sched                    0.876318       0.879241        0.941044
e1_h128_64_lowdrop_bs512_sched            0.866303       0.867663        0.933241
e1_baseline_bs512_sched                   0.857443       0.853586        0.931160
```

Best selected model:

```text
e1_h128_64_nodrop_wd_bs512_sched
```

Reason:

```text
It achieved the highest validation Macro F1 and the strongest overall test Macro F1 among the tested configurations.
```

---

## Output Files

Best checkpoint:

```text
G:\FOR THESIS\MAIN\e1_iot_only\checkpoints\best_e1_h128_64_nodrop_wd_bs512_sched.pt
```

Best metrics:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\e1_h128_64_nodrop_wd_bs512_sched_metrics.json
```

Training history:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\e1_h128_64_nodrop_wd_bs512_sched_history.csv
```

Validation predictions:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\e1_h128_64_nodrop_wd_bs512_sched_val_predictions.csv
```

Test predictions:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\e1_h128_64_nodrop_wd_bs512_sched_test_predictions.csv
```

Test confusion matrix:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\confusion_matrices\e1_h128_64_nodrop_wd_bs512_sched_test_confusion_matrix.csv
```

All-run comparison:

```text
G:\FOR THESIS\MAIN\e1_iot_only\results\e1_run_comparison.csv
```

---

## Interpretation

E1 answers this question:

```text
How much predictive signal exists in the IoT/environmental data alone?
```

Result:

```text
The IoT-only baseline is strong on the current split.
```

Meaning:

```text
Environmental sensor summaries contain meaningful signal for the crop-disease-stage labels.
However, this could also reflect dataset-specific correlations between class labels and collection conditions.
```

Important next comparison:

```text
Compare E1 directly with Abrar's E0 image-only baseline using the same split.
```

If E2 beats both E0 and E1:

```text
Multimodal fusion is justified.
```

If E2 does not beat E0 or E1:

```text
Fusion may not be adding useful information, or one modality may dominate.
```

---

## Suggested Thesis Wording

```text
The IoT-only baseline evaluates whether environmental sensor measurements alone contain discriminative information for crop-disease-stage classification. Each time-series sensor file is converted into a 54-dimensional statistical summary vector by computing six descriptive statistics over nine selected environmental channels. A lightweight multilayer perceptron is trained using weighted cross entropy, train-only MinMax scaling, and validation Macro F1 checkpointing. The same fixed train/validation/test split used by the image-only baseline is used to ensure a fair comparison.
```

---

## Next Steps

1. Collect Abrar's E0 result on the same split.
2. Compare E0 vs E1:

```text
Accuracy
Macro F1
Macro ROC-AUC
Early-stage recall
Per-stage recall
Confusion matrix
Parameter count
```

3. Use E1 as the sensor-only floor for E2 multimodal fusion.
4. Keep this file updated if E1 is rerun.
