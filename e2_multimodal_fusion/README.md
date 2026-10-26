# E2 Multimodal Fusion

Frozen-encoder classical image + IoT fusion baseline for the thesis experiment sequence.

> Canonical run: `e2_bs512_fusionmlp`, a frozen snapshot in `e2_multimodal_fusion/canonical/`
> (see `docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md`). Training still writes regular experiment outputs, but
> downstream E2/E3 extraction reads the canonical E0/E1 snapshots by default.

## Selected Encoders

| Encoder | Run | Checkpoint | Embedding |
|---|---|---|---:|
| E0 image | `e0_bs768_mobilenetv3large` | `e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth` | 1280 |
| E1 IoT | `e1_bs512_mlp128x64` | `e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt` | 64 |

Fusion input dimension: `1280 + 64 = 1344`.

## Data Split

```text
Data/train.csv: 46,713 rows
Data/val.csv:    5,193 rows
Data/test.csv:   5,767 rows
classes: 25 sorted labels, identical to E0/E1
```

## Workflow

From the repository root:

```powershell
python e2_multimodal_fusion\scripts\extract_e2_embeddings.py --batch-size 512 --num-workers 0
python e2_multimodal_fusion\scripts\train_e2_fusion.py --batch-size 512 --epochs 60 --patience 8 --lr 0.001 --weight-decay 0.0001
```

The extraction step reads canonical E0/E1 checkpoints. Retraining E0/E1 does not
change E2/E3 inputs until the new run is explicitly promoted into the
corresponding `canonical/` folder.

The extraction script validates:

```text
E0/E1 class-list equality
train/val/test row counts
split label order
NaN-free embeddings
```

## Fusion Head

```text
Linear(1344, 128)
BatchNorm1d(128)
ReLU
Dropout(0.1)
Linear(128, 4)
ReLU
Linear(4, 25)
```

Training uses weighted cross entropy, Adam, ReduceLROnPlateau on validation Macro F1, and validation Macro F1 checkpointing.

## Final Run

```text
canonical run name: e2_bs512_fusionmlp
original source run: e2_frozen_e0_manual_e1_h128_64_fusion
best epoch: 38
trainable params: 173,057
device: cuda
```

Validation:

```text
accuracy: 0.988446
Macro F1: 0.970913
macro recall: 0.976274
early-stage recall: 0.985904
severe-stage recall: 0.914286
ECE: 0.005393
```

Test:

```text
accuracy: 0.987342
Macro F1: 0.967926
macro recall: 0.970493
macro ROC-AUC OVR: 0.998345
weighted ROC-AUC OVR: 0.999274
early-stage recall: 0.986436
severe-stage recall: 0.821429
ECE: 0.006775
```

## Interpretation

E2 does not beat selected E0 on validation Macro F1:

```text
E0 val Macro F1: 0.973325
E2 val Macro F1: 0.970913
drop: 0.002412
```

E2 still satisfies the planned secondary-gain criterion because validation ECE improves by about 12.7% relative and validation severe-stage recall improves while the validation Macro F1 drop remains below 0.005.

On test, E2 improves over E0 on accuracy, Macro F1, macro recall, early-stage recall, severe-stage recall, and ECE.

## Canonical Artifacts

```text
canonical/best_e2_bs512_fusionmlp.pt
canonical/e2_bs512_fusionmlp_metrics.json
canonical/e2_bs512_fusionmlp_history.csv
canonical/e2_bs512_fusionmlp_val_predictions.csv
canonical/e2_bs512_fusionmlp_test_predictions.csv
canonical/e2_bs512_fusionmlp_test_confusion_matrix.csv
```

## Working Pipeline Artifacts

These remain under the original training run name and are kept for regeneration/history:

```text
features/e2_embedding_manifest.json
results/e2_frozen_e0_manual_e1_h128_64_fusion_metrics.json
results/e2_frozen_e0_manual_e1_h128_64_fusion_history.csv
results/e2_frozen_e0_manual_e1_h128_64_fusion_val_predictions.csv
results/e2_frozen_e0_manual_e1_h128_64_fusion_test_predictions.csv
results/confusion_matrices/e2_frozen_e0_manual_e1_h128_64_fusion_test_confusion_matrix.csv
checkpoints/best_e2_frozen_e0_manual_e1_h128_64_fusion.pt
```

Large generated NPZ embedding caches and checkpoints are ignored by git.
