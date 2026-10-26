# E3 Mid-Stage Error Diagnostic

Source predictions:

- `e3_beat_e2_attack\results\e2_matched_8_pair_s42_best_macro_test_predictions.csv`
- `e3_beat_e2_attack\results\e3_8_warm_pair_s42_best_macro_test_predictions.csv`

## Stage Summary

| Stage | Name | N | E2 correct | E3 correct | E2 correct / E3 wrong | E2 acc | E3 acc |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | healthy | 3944 | 3944 | 3937 | 7 | 1.000000 | 0.998225 |
| 1 | early | 1327 | 1297 | 1274 | 24 | 0.977393 | 0.960060 |
| 2 | mid | 440 | 411 | 412 | 2 | 0.934091 | 0.936364 |
| 3 | severe | 56 | 45 | 41 | 4 | 0.803571 | 0.732143 |

## E3 Mid-Stage Error Direction

| E3 predicted stage | Name | Count |
|---|---|---:|
| 1 | early | 25 |
| 0 | healthy | 1 |
| 2 | mid | 1 |
| 3 | severe | 1 |

## Top E3 Wrong Confusion Pairs

| True stage | True label | E3 pred label | E3 pred stage | Count |
|---|---|---|---|---:|
| 1 | 3_a9_1 | 3_a9_2 | 2 | 23 |
| 2 | 3_a9_2 | 3_a9_1 | 1 | 16 |
| 1 | 3_a9_1 | 3_b6_1 | 1 | 8 |
| 2 | 3_a9_2 | 3_b6_1 | 1 | 6 |
| 3 | 3_a9_3 | 3_a9_2 | 2 | 6 |
| 0 | 3_00_0 | 3_a9_2 | 2 | 4 |
| 1 | 3_b3_1 | 3_a9_2 | 2 | 4 |
| 1 | 3_b6_1 | 3_a9_3 | 3 | 3 |
| 2 | 3_a9_2 | 3_b3_1 | 1 | 3 |
| 3 | 3_a9_3 | 3_a9_1 | 1 | 3 |
| 1 | 3_b6_1 | 6_a12_1 | 1 | 2 |
| 1 | 3_b7_1 | 6_00_0 | 0 | 2 |
| 1 | 3_b8_1 | 5_00_0 | 0 | 2 |
| 3 | 3_a9_3 | 2_a5_2 | 2 | 2 |
| 3 | 3_a9_3 | 3_b6_1 | 1 | 2 |
| 0 | 1_00_0 | 6_a12_2 | 2 | 1 |
| 0 | 2_00_0 | 3_a9_2 | 2 | 1 |
| 0 | 3_00_0 | 4_00_0 | 0 | 1 |
| 1 | 3_a9_1 | 3_00_0 | 0 | 1 |
| 1 | 3_a9_1 | 3_a9_3 | 3 | 1 |

## Files

- `stage_error_summary.csv`
- `mid_stage_e2_correct_e3_wrong.csv`
- `e3_wrong_confusion_pairs.csv`
- `e3_mid_stage_error_direction.csv`
