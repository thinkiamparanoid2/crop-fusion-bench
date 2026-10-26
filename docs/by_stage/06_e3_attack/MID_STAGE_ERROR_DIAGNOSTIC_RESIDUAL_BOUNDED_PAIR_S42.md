# E3 Mid-Stage Error Diagnostic

Source predictions:

- `e3_beat_e2_attack\results\e2_matched_8_residual_bounded_pair_diag_s42_best_macro_test_predictions.csv`
- `e3_beat_e2_attack\results\e3_8_warm_residual_bounded_pair_diag_s42_best_macro_test_predictions.csv`

## Stage Summary

| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | healthy | 3944 | 3943 | 3941 | 3 | 0.999746 | 0.999239 |
| 1 | early | 1327 | 1294 | 1284 | 12 | 0.975132 | 0.967596 |
| 2 | mid | 440 | 412 | 412 | 2 | 0.936364 | 0.936364 |
| 3 | severe | 56 | 45 | 45 | 0 | 0.803571 | 0.803571 |

## E3 Mid-Stage Error Direction

| E3 predicted stage | Name | Count |
|---|---|---:|
| 1 | early | 26 |
| 3 | severe | 2 |

## Top E3 Wrong Confusion Pairs

| True stage | True label | E3 pred label | E3 pred stage | Count |
|---|---|---|---|---:|
| 2 | 3_a9_2 | 3_a9_1 | 1 | 26 |
| 1 | 3_a9_1 | 3_a9_2 | 2 | 25 |
| 3 | 3_a9_3 | 3_a9_1 | 1 | 6 |
| 3 | 3_a9_3 | 3_a9_2 | 2 | 5 |
| 1 | 3_b8_1 | 3_b6_1 | 1 | 4 |
| 1 | 3_b3_1 | 3_b8_1 | 1 | 2 |
| 1 | 3_b6_1 | 3_00_0 | 0 | 2 |
| 1 | 5_b6_1 | 3_00_0 | 0 | 2 |
| 2 | 3_a9_2 | 3_a9_3 | 3 | 2 |
| 0 | 3_00_0 | 3_b3_1 | 1 | 1 |
| 0 | 3_00_0 | 5_b8_1 | 1 | 1 |
| 0 | 6_00_0 | 6_a12_2 | 2 | 1 |
| 1 | 3_a9_1 | 3_a9_3 | 3 | 1 |
| 1 | 3_b3_1 | 3_00_0 | 0 | 1 |
| 1 | 3_b3_1 | 3_a9_2 | 2 | 1 |
| 1 | 3_b3_1 | 3_b6_1 | 1 | 1 |
| 1 | 3_b7_1 | 3_a9_2 | 2 | 1 |
| 1 | 3_b7_1 | 5_b7_1 | 1 | 1 |
| 1 | 3_b7_1 | 6_a12_1 | 1 | 1 |
| 1 | 5_b8_1 | 3_b8_1 | 1 | 1 |

## Files

- `stage_error_summary.csv`
- `mid_stage_e2_correct_e3_wrong.csv`
- `e3_wrong_confusion_pairs.csv`
- `e3_mid_stage_error_direction.csv`
