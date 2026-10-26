# E3 Mid-Stage Error Diagnostic

Source predictions:

- `e3_beat_e2_attack\results\e2_matched_8_residual_bounded_diag_s42_best_macro_test_predictions.csv`
- `e3_beat_e2_attack\results\e3_8_warm_residual_bounded_diag_s42_best_macro_test_predictions.csv`

## Stage Summary

| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | healthy | 3944 | 3942 | 3934 | 8 | 0.999493 | 0.997465 |
| 1 | early | 1327 | 1302 | 1288 | 17 | 0.981161 | 0.970610 |
| 2 | mid | 440 | 406 | 409 | 0 | 0.922727 | 0.929545 |
| 3 | severe | 56 | 46 | 44 | 2 | 0.821429 | 0.785714 |

## E3 Mid-Stage Error Direction

| E3 predicted stage | Name | Count |
|---|---|---:|
| 1 | early | 29 |
| 3 | severe | 2 |

## Top E3 Wrong Confusion Pairs

| True stage | True label | E3 pred label | E3 pred stage | Count |
|---|---|---|---|---:|
| 2 | 3_a9_2 | 3_a9_1 | 1 | 29 |
| 1 | 3_a9_1 | 3_a9_2 | 2 | 22 |
| 3 | 3_a9_3 | 3_a9_1 | 1 | 5 |
| 3 | 3_a9_3 | 3_a9_2 | 2 | 5 |
| 1 | 3_b6_1 | 3_a9_2 | 2 | 3 |
| 0 | 3_00_0 | 3_a9_1 | 1 | 2 |
| 0 | 3_00_0 | 5_b6_1 | 1 | 2 |
| 0 | 6_00_0 | 4_00_0 | 0 | 2 |
| 1 | 3_b8_1 | 3_b6_1 | 1 | 2 |
| 1 | 5_b6_1 | 3_b3_1 | 1 | 2 |
| 1 | 5_b8_1 | 3_b8_1 | 1 | 2 |
| 2 | 3_a9_2 | 3_a9_3 | 3 | 2 |
| 3 | 3_a9_3 | 6_b4_1 | 1 | 2 |
| 0 | 3_00_0 | 3_b3_1 | 1 | 1 |
| 0 | 4_00_0 | 2_a5_2 | 2 | 1 |
| 0 | 4_00_0 | 3_b7_1 | 1 | 1 |
| 0 | 6_00_0 | 2_a5_2 | 2 | 1 |
| 1 | 3_a9_1 | 3_a9_3 | 3 | 1 |
| 1 | 3_b3_1 | 3_b6_1 | 1 | 1 |
| 1 | 3_b6_1 | 3_b8_1 | 1 | 1 |

## Files

- `stage_error_summary.csv`
- `mid_stage_e2_correct_e3_wrong.csv`
- `e3_wrong_confusion_pairs.csv`
- `e3_mid_stage_error_direction.csv`
