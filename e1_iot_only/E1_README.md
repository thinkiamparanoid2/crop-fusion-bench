# E1: IoT-Only Baseline

## Current Status Note

This README preserves the original E1 design notes. The active canonical E1 run
is `e1_bs512_mlp128x64`, copied from `e1_h128_64_nodrop_wd_bs512_sched` into
`e1_iot_only/canonical/`. For thesis comparisons, use
`docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md` and
`docs/by_stage/03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md`.

Key current differences from the early plan below:

- E1 uses the fixed shared root split: `Data/train.csv`, `Data/val.csv`,
  `Data/test.csv`.
- The selected run uses seed 42, batch size 512, hidden layers 128/64, no dropout,
  and validation Macro F1 checkpointing.
- Three-seed E1 reporting is deferred unless E0/E1/E2/E3 are all rerun on the
  same three split definitions.

## Purpose

E1 is the IoT-only baseline for the thesis.

It uses only the environmental sensor CSV files from the Dacon dataset and does not use crop images. The goal is to measure how much predictive signal exists in the IoT stream alone.

E1 is compared against:

- E0: image-only baseline
- E2: classical multimodal baseline
- E3: hybrid quantum-classical multimodal model

E1 is not expected to be the strongest model. Its purpose is to provide a clean sensor-only floor.

## Dataset Inputs

Main dataset location:

```text
G:\FOR THESIS\MAIN\Data
```

Manifest files:

```text
Data\train.csv
Data\test.csv
```

Sample folders:

```text
Data\train\{image_id}\{image_id}_final.csv
Data\test\{image_id}\{image_id}_final.csv
```

Each `_final.csv` contains IoT time-series data.

Expected columns:

```text
timestamp
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

The `timestamp` column is kept for reference but is not used directly as a model feature in the first E1 implementation.

## Label Format

Labels follow this format:

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

The E1 model predicts the full combined label, not only the stage.

## Dataset Audit Summary

From the heavy scan:

```text
Train samples: 51,906
Test samples: 5,767
Classes: 25
Missing _final.csv files: 0
Missing JPG files: 0
Schema variants: 1
```

Stage distribution:

```text
Train:
stage 0: 35,645
stage 1: 12,050
stage 2: 3,869
stage 3: 342

Test:
stage 0: 3,944
stage 1: 1,327
stage 2: 440
stage 3: 56
```

Known data issues:

```text
Some dew-point columns contain null values.
Some files contain duplicate timestamps.
```

E1 handles nulls during feature extraction.

## E1 Folder Structure

```text
e1_iot_only/
|-- E1_README.md
|-- scripts/
|   |-- extract_iot_features.py
|   `-- train_e1_iot_mlp.py
|-- features/
|   |-- train_iot_features.csv
|   `-- test_iot_features.csv
|-- results/
|   |-- seed_13_metrics.json
|   |-- seed_37_metrics.json
|   |-- seed_73_metrics.json
|   |-- e1_summary.csv
|   `-- confusion_matrices/
`-- checkpoints/
    |-- best_e1_seed_13.pt
    |-- best_e1_seed_37.pt
    `-- best_e1_seed_73.pt
```

## Feature Extraction Strategy

Each `_final.csv` is a time-series file. The MLP needs one fixed-size vector per sample.

Use these 9 numeric columns:

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

For each numeric column, compute:

```text
mean
std
min
max
median
last
```

Total feature count:

```text
9 columns x 6 statistics = 54 features
```

Metadata columns to keep:

```text
image
label
crop
disease
stage
n_rows
```

Output files:

```text
features\train_iot_features.csv
features\test_iot_features.csv
```

## Pseudocode: Feature Extraction

```python
SET base_dir = PROJECT_ROOT / "Data"
SET output_dir = E1_ROOT / "features"

DEFINE numeric_cols:
    int_temp_1_avg
    int_temp_1_max
    int_temp_1_min
    int_hum_1_avg
    int_hum_1_max
    int_hum_1_min
    int_dew_point_avg
    int_dew_point_max
    int_dew_point_min

DEFINE stats:
    mean
    std
    min
    max
    median
    last

FOR split IN ["train", "test"]:

    READ manifest file:
        Data/{split}.csv

    CREATE empty feature_rows list

    FOR each row in manifest:

        image_id = row["image"]
        label = row["label"]

        PARSE label:
            crop = first part before "_"
            disease = second part
            stage = third part

        csv_path = Data/{split}/{image_id}/{image_id}_final.csv

        READ csv_path into dataframe df

        SELECT numeric_cols

        CONVERT numeric columns to numeric values

        HANDLE missing values:
            forward fill
            backward fill
            fill remaining missing values with file-level median
            if still missing, fill with 0

        CREATE feature_row:
            image = image_id
            label = label
            crop = crop
            disease = disease
            stage = stage
            n_rows = number of rows in df

        FOR each col in numeric_cols:
            feature_row[col + "_mean"] = mean(df[col])
            feature_row[col + "_std"] = std(df[col])
            feature_row[col + "_min"] = min(df[col])
            feature_row[col + "_max"] = max(df[col])
            feature_row[col + "_median"] = median(df[col])
            feature_row[col + "_last"] = last value of df[col]

        APPEND feature_row to feature_rows

    CONVERT feature_rows to dataframe

    CHECK:
        no missing feature values
        row count matches manifest row count

    SAVE dataframe to:
        features/{split}_iot_features.csv
```

## Training Strategy

Use `train_iot_features.csv` for the shared train/validation split and
`test_iot_features.csv` for final evaluation.

```text
Data/train.csv: 46,713 rows
Data/val.csv:    5,193 rows
Data/test.csv:   5,767 rows
```

The canonical run selects rows by image ID from the fixed split files instead of
creating a new random E1-only split.

Current canonical seed:

```text
42
```

## Scaling

Use `MinMaxScaler`.

Important rule:

```text
Fit scaler on training split only.
```

Then transform:

```text
train split
validation split
test set
```

Do not fit the scaler on validation or test data.

## Model Architecture

Input dimension:

```text
54
```

Canonical MLP:

```text
Linear(54, 128)
BatchNorm1d(128)
ReLU
Dropout(0.0)

Linear(128, 64)
BatchNorm1d(64)
ReLU
Dropout(0.0)

Linear(64, num_classes)
```

Output dimension:

```text
25 classes
```

## Pseudocode: E1 Training

```python
SET seed = 42

READ train_iot_features.csv
READ test_iot_features.csv

IDENTIFY feature columns:
    all columns except image, label, crop, disease, stage, n_rows

ENCODE labels:
    fit LabelEncoder on train labels
    transform train labels
    transform test labels

FOR the canonical seed:

    SET random seed for:
        python
        numpy
        torch
        cuda

    SELECT train_split and val_split by Data/train.csv and Data/val.csv image IDs

    FIT MinMaxScaler on train_split feature columns only

    TRANSFORM:
        train_split features
        val_split features
        test features

    COMPUTE class weights from train_split labels

    CREATE PyTorch Dataset objects:
        train_dataset
        val_dataset
        test_dataset

    CREATE DataLoaders:
        train_loader with shuffle=True
        val_loader with shuffle=False
        test_loader with shuffle=False

    INITIALIZE E1 MLP model

    DEFINE loss:
        weighted cross entropy

    DEFINE optimizer:
        Adam, lr = 1e-3

    SET:
        max_epochs = 40
        early_stopping_patience = 7
        best_val_macro_f1 = -infinity
        patience_counter = 0

    FOR epoch in range(max_epochs):

        TRAIN for one epoch:
            model.train()
            for batch in train_loader:
                x, y = batch
                logits = model(x)
                loss = cross_entropy(logits, y)
                backpropagate
                optimizer step

        VALIDATE:
            model.eval()
            collect validation predictions
            calculate validation macro_f1

        IF validation macro_f1 improves:
            save model checkpoint
            reset patience_counter

        ELSE:
            increment patience_counter

        IF patience_counter >= early_stopping_patience:
            stop training

    LOAD best checkpoint

    EVALUATE on test set:
        accuracy
        macro_f1
        macro_recall
        per-stage recall
        early-stage recall
        confusion matrix

    SAVE:
        metrics json
        test predictions csv
        confusion matrix image
        checkpoint
```

## Metrics to Report

For each seed:

```text
accuracy
macro_f1
macro_recall
per-stage recall
early-stage recall
best epoch
trainable parameter count
```

Final E1 result should report:

```text
mean +/- std over seeds 13, 37, 73
```

Primary metric:

```text
Macro F1
```

Thesis-critical metric:

```text
Early-stage recall
```

## Interpretation

E1 answers this question:

```text
How much class and stage signal exists in IoT/environmental data alone?
```

Possible outcomes:

```text
If E1 is weak:
    This is acceptable. It means IoT alone is insufficient.

If E1 is strong:
    This means environmental conditions contain useful disease/stage signal.

If E2 beats E0 and E1:
    Multimodal fusion is justified.

If E2 does not beat E0:
    The IoT branch may be adding noise or leakage needs to be checked.
```

## Notes for Thesis Writing

E1 should be described as a sensor-only baseline.

Suggested wording:

```text
The IoT-only baseline evaluates whether environmental sensor measurements alone contain sufficient discriminatory information for crop-disease-stage classification. Each time-series sensor file is converted into a fixed-length statistical summary vector and classified using a lightweight multilayer perceptron. This baseline establishes the standalone contribution of the environmental modality before multimodal fusion.
```
