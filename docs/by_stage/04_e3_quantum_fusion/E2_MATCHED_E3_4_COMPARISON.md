# E2-Matched vs E3-4 Comparison

Date: 2026-06-04

This report compares the parameter-matched classical comparator against the first
4-qubit hybrid quantum model. Both use the same frozen E2 embedding interface:

```text
E0 image embedding: 1280 dims
E1 IoT embedding:     64 dims
Fused input:        1344 dims
Split: Data/train.csv / Data/val.csv / Data/test.csv
Classes: 25 sorted crop-disease-stage labels
```

## Runs

| Model | Run | Device | Simulator | Best epoch | Total params | Post-bottleneck params |
|---|---|---|---|---:|---:|---:|
| E2-matched | `e2_matched_4` | cuda | n/a | 44 | 173,073 | 16 |
| E3-4 | `e3_4_vqc` | cpu | `lightning.qubit` | 53 | 173,073 | 16 |

E3-4 used CPU because the installed PennyLane stack has `default.qubit` and
`lightning.qubit`, but not `lightning.gpu`.

## Architecture

```text
Shared projection:
Linear(1344,128) -> BatchNorm1d(128) -> ReLU -> Dropout(0.1) -> Linear(128,4)

E2-matched middle:
tanh -> Linear(4,4,bias=False) -> tanh

E3-4 middle:
tanh*pi -> 4-qubit VQC -> 4 Pauli-Z expectations

Shared readout:
Linear(4,25)
```

## Validation Metrics

| Metric | E2-matched | E3-4 | E3-4 minus E2-matched |
|---|---:|---:|---:|
| Accuracy | 0.930483 | 0.905642 | -0.024841 |
| Macro F1 | 0.680223 | 0.723280 | +0.043057 |
| Macro recall | 0.700216 | 0.741393 | +0.041177 |
| Early-stage recall | 0.974295 | 0.870647 | -0.103648 |
| ECE | 0.341999 | 0.270870 | -0.071129 |

## Test Metrics

| Metric | E2-matched | E3-4 | E3-4 minus E2-matched |
|---|---:|---:|---:|
| Accuracy | 0.924224 | 0.896133 | -0.028091 |
| Macro F1 | 0.661438 | 0.719661 | +0.058223 |
| Macro recall | 0.691956 | 0.741060 | +0.049104 |
| Early-stage recall | 0.971364 | 0.868124 | -0.103240 |
| ECE | 0.336519 | 0.263169 | -0.073350 |

## Per-Stage Test Recall

| Stage | E2-matched | E3-4 | E3-4 minus E2-matched |
|---|---:|---:|---:|
| Stage 0 healthy | 0.981998 | 0.999493 | +0.017495 |
| Stage 1 early | 0.971364 | 0.868124 | -0.103240 |
| Stage 2 mid | 0.743182 | 0.947727 | +0.204545 |
| Stage 3 severe | 0.250000 | 0.000000 | -0.250000 |

## Interpretation

E3-4 beats the parameter-matched classical comparator on test Macro F1, macro
recall, and calibration:

```text
Macro F1:     +0.058223
Macro recall: +0.049104
ECE:          -0.073350
```

However, the gain is not uniformly thesis-positive. E3-4 loses about 10.3 points
of early-stage recall and predicts no severe-stage test samples correctly. The
Macro F1 improvement appears to come mainly from stronger mid-stage performance
and healthier calibration, not from better early/severe detection.

This means E3-4 is a useful controlled quantum result, but it should not be
reported as a clean early-detection advantage. The honest claim is:

```text
Under the matched 4-dimensional bottleneck, the 4-qubit VQC improves Macro F1,
macro recall, and ECE over the matched classical 4->4 comparator, but it reduces
early-stage recall and fails on severe-stage recall.
```

## Artifacts

```text
e3_quantum_fusion/results/e2_matched_4_metrics.json
e3_quantum_fusion/results/e2_matched_4_history.csv
e3_quantum_fusion/results/e2_matched_4_val_predictions.csv
e3_quantum_fusion/results/e2_matched_4_test_predictions.csv
e3_quantum_fusion/results/confusion_matrices/e2_matched_4_test_confusion_matrix.csv
e3_quantum_fusion/checkpoints/best_e2_matched_4.pt

e3_quantum_fusion/results/e3_4_vqc_metrics.json
e3_quantum_fusion/results/e3_4_vqc_history.csv
e3_quantum_fusion/results/e3_4_vqc_val_predictions.csv
e3_quantum_fusion/results/e3_4_vqc_test_predictions.csv
e3_quantum_fusion/results/confusion_matrices/e3_4_vqc_test_confusion_matrix.csv
e3_quantum_fusion/checkpoints/best_e3_4_vqc.pt
```

