# E0 vs E1 Results

Historical note: this comparison predates the selected E0 rerun and completed E2 fusion baseline. The E0 column here is the non-canonical batch-size-256 run `e0_mobilenetv3_large_gpu_bs256_unfreeze2_amp`, not the canonical bs768 run. For the current thesis comparison and the locked run names, use:

```text
docs/by_stage/03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md
docs/by_stage/03_baselines_e0_e1_e2/CANONICAL_RUNS.md
```

## Test Metrics

| Model | Modality | Accuracy | Macro F1 | Macro Recall | Early Recall | ECE |
|---|---|---:|---:|---:|---:|---:|
| E0 | Image-only | 0.983180 | 0.961853 | 0.960249 | 0.978900 | 0.009742 |
| E1 | IoT-only | 0.943645 | 0.883370 | 0.882642 | 0.939714 | 0.016543 |

## Per-Stage Test Recall

| Model | Stage 0 Healthy | Stage 1 Early | Stage 2 Mid | Stage 3 Severe |
|---|---:|---:|---:|---:|
| E0 | 0.998732 | 0.978900 | 0.920455 | 0.785714 |
| E1 | 1.000000 | 0.939714 | 0.884091 | 0.482143 |

## Confusion Matrix: Stage Level

Rows are true stages. Columns are predicted stages.

### E0

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 3939 | 5 | 0 | 0 |
| 1 | 8 | 1299 | 17 | 3 |
| 2 | 0 | 32 | 405 | 3 |
| 3 | 0 | 3 | 9 | 44 |

### E1

| True \ Pred | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| 0 | 3944 | 0 | 0 | 0 |
| 1 | 0 | 1247 | 61 | 19 |
| 2 | 0 | 29 | 389 | 22 |
| 3 | 0 | 8 | 21 | 27 |

## Top Class Confusions

### E0

| Count | True | Predicted |
|---:|---|---|
| 30 | 3_a9_2 | 3_a9_1 |
| 15 | 3_a9_1 | 3_a9_2 |
| 9 | 3_a9_3 | 3_a9_2 |
| 6 | 3_b6_1 | 3_b8_1 |

### E1

| Count | True | Predicted |
|---:|---|---|
| 61 | 3_a9_1 | 3_a9_2 |
| 36 | 3_b6_1 | 3_b8_1 |
| 31 | 5_b8_1 | 3_b8_1 |
| 29 | 3_a9_2 | 3_a9_1 |
| 28 | 5_b6_1 | 3_b3_1 |
| 27 | 3_b8_1 | 3_b6_1 |

## E0 OOD Corruption Accuracy

| Condition | Accuracy | Drop vs Clean |
|---|---:|---:|
| Clean | 0.983180 | 0.000000 |
| Brightness Low | 0.977458 | -0.005722 |
| Brightness High | 0.980406 | -0.002774 |
| Contrast Low | 0.975897 | -0.007283 |
| Gaussian Noise | 0.696376 | -0.286804 |
| Blur | 0.754638 | -0.228542 |
| Combined | 0.827987 | -0.155193 |
