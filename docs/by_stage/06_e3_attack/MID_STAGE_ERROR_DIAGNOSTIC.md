# E3 Mid-Stage Error Diagnostic

Source predictions:

- `e3_bottleneck_sweep/results/e2_matched_8_test_predictions.csv`
- `e3_bottleneck_sweep/results/e3_8_vqc_test_predictions.csv`

## Stage Summary

| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | healthy | 3944 | 3942 | 3943 | 1 | 0.999493 | 0.999746 |
| 1 | early | 1327 | 1296 | 1274 | 40 | 0.976639 | 0.960060 |
| 2 | mid | 440 | 412 | 353 | 64 | 0.936364 | 0.802273 |
| 3 | severe | 56 | 45 | 45 | 1 | 0.803571 | 0.803571 |

## E3 Mid-Stage Error Direction

| E3 predicted stage | Name | Count |
|---|---|---:|
| 1 | early | 85 |
| 3 | severe | 2 |

## Top E3 Wrong Confusion Pairs

| True stage | True label | E3 pred label | E3 pred stage | Count |
|---|---|---|---|---:|
| 2 | 3_a9_2 | 3_a9_1 | 1 | 85 |
| 1 | 3_b3_1 | 5_b6_1 | 1 | 25 |
| 1 | 3_a9_1 | 3_a9_2 | 2 | 13 |
| 3 | 3_a9_3 | 3_a9_1 | 1 | 9 |
| 1 | 3_b8_1 | 3_b6_1 | 1 | 5 |
| 1 | 3_b6_1 | 5_b6_1 | 1 | 2 |
| 2 | 3_a9_2 | 3_a9_3 | 3 | 2 |
| 3 | 3_a9_3 | 3_a9_2 | 2 | 2 |
| 0 | 3_00_0 | 6_a12_2 | 2 | 1 |
| 1 | 3_a9_1 | 2_00_0 | 0 | 1 |
| 1 | 3_a9_1 | 3_a9_3 | 3 | 1 |
| 1 | 3_b3_1 | 3_b6_1 | 1 | 1 |
| 1 | 3_b6_1 | 3_b8_1 | 1 | 1 |
| 1 | 3_b6_1 | 4_00_0 | 0 | 1 |
| 1 | 3_b7_1 | 1_00_0 | 0 | 1 |
| 1 | 5_b8_1 | 3_b6_1 | 1 | 1 |
| 1 | 5_b8_1 | 5_b6_1 | 1 | 1 |

## Files

- `stage_error_summary.csv`
- `mid_stage_e2_correct_e3_wrong.csv`
- `e3_wrong_confusion_pairs.csv`
- `e3_mid_stage_error_direction.csv`
