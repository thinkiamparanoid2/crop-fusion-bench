# E0/E1/E2 Results Comparison

Date: 2026-06-02

Canonical run names are locked in [CANONICAL_RUNS.md](CANONICAL_RUNS.md); the runs below are those canonical runs.

## Selected Runs

| Experiment | Run | Checkpoint | Trainable params |
|---|---|---:|---:|
| E0 image-only | `e0_bs768_mobilenetv3large` | `e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth` | 2,214,985 |
| E1 IoT-only | `e1_bs512_mlp128x64` | `e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt` | 17,305 |
| E2 classical fusion | `e2_bs512_fusionmlp` | `e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt` | 173,057 |

All runs use the same fixed split and 25-class sorted label space:

```text
Data/train.csv: 46,713 rows
Data/val.csv:    5,193 rows
Data/test.csv:   5,767 rows
```

## Validation Metrics

| Experiment | Accuracy | Macro F1 | Macro Recall | Early Recall | Severe Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|
| E0 image-only | 0.985750 | 0.973325 | 0.973945 | 0.985904 | 0.885714 | 0.006182 |
| E1 IoT-only | 0.943963 | 0.887618 | 0.889753 | 0.932007 | 0.542857 | 0.016533 |
| E2 classical fusion | 0.988446 | 0.970913 | 0.976274 | 0.985904 | 0.914286 | 0.005393 |

## Test Metrics

| Experiment | Accuracy | Macro F1 | Macro Recall | Macro ROC-AUC OVR | Weighted ROC-AUC OVR | Early Recall | Severe Recall | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 image-only | 0.982140 | 0.963896 | 0.962667 | 0.999021 | 0.999596 | 0.980407 | 0.803571 | 0.007805 |
| E1 IoT-only | 0.943645 | 0.883370 | 0.882642 | 0.997505 | 0.998793 | 0.939714 | 0.482143 | 0.016543 |
| E2 classical fusion | 0.987342 | 0.967926 | 0.970493 | 0.998345 | 0.999274 | 0.986436 | 0.821429 | 0.006775 |

All metrics are computed with the shared `common/thesis_eval.py` logic, so E0/E1/E2 (and the upcoming E3) are scored identically.

## Per-Stage Test Recall

| Experiment | Stage 0 Healthy | Stage 1 Early | Stage 2 Mid | Stage 3 Severe |
|---|---:|---:|---:|---:|
| E0 image-only | 0.996450 | 0.980407 | 0.915909 | 0.803571 |
| E1 IoT-only | 1.000000 | 0.939714 | 0.884091 | 0.482143 |
| E2 classical fusion | 0.999746 | 0.986436 | 0.913636 | 0.821429 |

The headline multimodal effect is severe-stage (Stage 3) recovery: fusion lifts test severe recall from E1's 0.482 to 0.821, also edging past E0's 0.804, while improving early-stage (Stage 1) recall to 0.986.

## Stage-Level Confusion Matrices (test)

Rows are true stages, columns are predicted stages.

### E0 image-only

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 3930 | 13 | 1 | 0 |
| 1 | 3 | 1301 | 22 | 1 |
| 2 | 4 | 31 | 403 | 2 |
| 3 | 0 | 6 | 5 | 45 |

### E1 IoT-only

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 3944 | 0 | 0 | 0 |
| 1 | 0 | 1247 | 61 | 19 |
| 2 | 0 | 29 | 389 | 22 |
| 3 | 0 | 8 | 21 | 27 |

### E2 classical fusion

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 3943 | 1 | 0 | 0 |
| 1 | 1 | 1309 | 14 | 3 |
| 2 | 0 | 35 | 402 | 3 |
| 3 | 0 | 6 | 4 | 46 |

## Interpretation

E2 does not beat selected E0 on validation Macro F1:

```text
E0 val Macro F1: 0.973325
E2 val Macro F1: 0.970913
drop: 0.002412
```

However, E2 satisfies the planned secondary-gain rule:

```text
Validation ECE improves from 0.006182 to 0.005393, about 12.7% relative.
Validation severe-stage recall improves from 0.885714 to 0.914286.
The validation Macro F1 drop is less than 0.005.
```

On the held-out test set, E2 improves over E0 on accuracy, Macro F1, macro recall, early-stage recall, severe-stage recall, and ECE. This supports keeping E2 as a valid multimodal baseline and as the controlled classical interface for E3.

## E2 Artifacts

Primary canonical artifacts:

```text
e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt
e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_metrics.json
e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_history.csv
e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_val_predictions.csv
e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_test_predictions.csv
e2_multimodal_fusion/canonical/e2_bs512_fusionmlp_test_confusion_matrix.csv
```

Working pipeline outputs remain under the original training run name:

```text
e2_multimodal_fusion/scripts/extract_e2_embeddings.py
e2_multimodal_fusion/scripts/train_e2_fusion.py
e2_multimodal_fusion/features/e2_embedding_manifest.json
e2_multimodal_fusion/results/e2_frozen_e0_manual_e1_h128_64_fusion_metrics.json
e2_multimodal_fusion/results/e2_frozen_e0_manual_e1_h128_64_fusion_history.csv
e2_multimodal_fusion/results/e2_frozen_e0_manual_e1_h128_64_fusion_val_predictions.csv
e2_multimodal_fusion/results/e2_frozen_e0_manual_e1_h128_64_fusion_test_predictions.csv
e2_multimodal_fusion/results/confusion_matrices/e2_frozen_e0_manual_e1_h128_64_fusion_test_confusion_matrix.csv
```

## Toward E3

E2 is the controlled classical interface for the quantum experiment. The exact
trainable-parameter accounting and the parameter-matched comparator (E2-matched
vs E3-4) are specified in
[../../e2_multimodal_fusion/PARAM_MATCH_SPEC.md](../../e2_multimodal_fusion/PARAM_MATCH_SPEC.md).
E0 robustness under image corruptions is in
[E0_OOD_CORRUPTION_RESULTS.md](E0_OOD_CORRUPTION_RESULTS.md).
