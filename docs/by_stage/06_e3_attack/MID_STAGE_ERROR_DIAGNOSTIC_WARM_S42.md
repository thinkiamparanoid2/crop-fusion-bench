# E3 Mid-Stage Error Diagnostic

Source predictions:

- `e3_bottleneck_sweep\results\e2_matched_8_test_predictions.csv`
- `e3_beat_e2_attack\results\e3_8_warm_s42_best_macro_test_predictions.csv`

## Stage Summary

| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | healthy | 3944 | 3942 | 3939 | 5 | 0.999493 | 0.998732 |
| 1 | early | 1327 | 1296 | 1281 | 16 | 0.976639 | 0.965335 |
| 2 | mid | 440 | 412 | 396 | 18 | 0.936364 | 0.900000 |
| 3 | severe | 56 | 45 | 44 | 3 | 0.803571 | 0.785714 |

## E3 Mid-Stage Error Direction

| E3 predicted stage | Name | Count |
|---|---|---:|
| 1 | early | 33 |
| 3 | severe | 6 |
| 0 | healthy | 4 |
| 2 | mid | 1 |

## Top E3 Wrong Confusion Pairs

| True stage | True label | E3 pred label | E3 pred stage | Count |
|---|---|---|---|---:|
| 2 | 3_a9_2 | 3_a9_1 | 1 | 23 |
| 1 | 3_a9_1 | 3_a9_2 | 2 | 14 |
| 2 | 3_a9_2 | 5_b8_1 | 1 | 9 |
| 2 | 3_a9_2 | 3_a9_3 | 3 | 6 |
| 1 | 3_a9_1 | 5_b8_1 | 1 | 5 |
| 1 | 3_a9_1 | 3_a9_3 | 3 | 4 |
| 2 | 3_a9_2 | 1_00_0 | 0 | 4 |
| 3 | 3_a9_3 | 3_a9_1 | 1 | 4 |
| 0 | 3_00_0 | 3_b3_1 | 1 | 3 |
| 1 | 3_a9_1 | 1_00_0 | 0 | 3 |
| 3 | 3_a9_3 | 3_a9_2 | 2 | 3 |
| 1 | 3_b8_1 | 3_00_0 | 0 | 2 |
| 1 | 3_b8_1 | 5_b8_1 | 1 | 2 |
| 1 | 3_b8_1 | 6_00_0 | 0 | 2 |
| 3 | 3_a9_3 | 5_b8_1 | 1 | 2 |
| 3 | 3_a9_3 | 6_b4_3 | 3 | 2 |
| 0 | 2_00_0 | 5_b8_1 | 1 | 1 |
| 0 | 6_00_0 | 4_00_0 | 0 | 1 |
| 1 | 3_a9_1 | 5_a7_2 | 2 | 1 |
| 1 | 3_b3_1 | 4_00_0 | 0 | 1 |

## Files

- `stage_error_summary.csv`
- `mid_stage_e2_correct_e3_wrong.csv`
- `e3_wrong_confusion_pairs.csv`
- `e3_mid_stage_error_direction.csv`
