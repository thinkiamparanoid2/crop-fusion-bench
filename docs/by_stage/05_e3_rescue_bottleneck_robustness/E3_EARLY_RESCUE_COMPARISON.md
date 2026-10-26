# E3 Early-Detection Rescue Comparison

This report includes the original CE baselines plus rescue runs generated from `e3_early_rescue/results/*_metrics.json`.

## Runs

| Group | Run | Model | Loss | Gamma | Checkpoint | Best epoch | Device | Test acc | Test Macro F1 | Test early recall | Test severe recall | Test ECE |
|---|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| baseline_ce | e2_matched_4 | e2_matched | weighted_cross_entropy | n/a | validation_macro_f1 | 44 | cuda | 0.924224 | 0.661438 | 0.971364 | 0.250000 | 0.336519 |
| baseline_ce | e3_4_vqc | e3_4 | weighted_cross_entropy | n/a | validation_macro_f1 | 53 | cpu | 0.896133 | 0.719661 | 0.868124 | 0.000000 | 0.263169 |
| early_rescue | e2_matched_focal_g2 | e2_matched | focal | 2.000000 | val_early_stage_recall | 5 | cuda | 0.754985 | 0.244287 | 0.778448 | 0.000000 | 0.448116 |
| early_rescue | e2_matched_weighted_focal_g2 | e2_matched | weighted_focal | 2.000000 | val_early_stage_recall | 6 | cuda | 0.478238 | 0.296063 | 0.984928 | 0.000000 | 0.286279 |
| early_rescue | e3_4_focal_g2 | e3_4 | focal | 2.000000 | val_early_stage_recall | 9 | cpu | 0.659788 | 0.298524 | 0.763376 | 0.000000 | 0.260376 |
| early_rescue | e3_4_weighted_focal_g2 | e3_4 | weighted_focal | 2.000000 | val_early_stage_recall | 5 | cpu | 0.332929 | 0.302279 | 0.990957 | 0.000000 | 0.191605 |

## Per-Stage Test Recall

| Run | Stage 0 healthy | Stage 1 early | Stage 2 mid | Stage 3 severe |
|---|---:|---:|---:|---:|
| e2_matched_4 | 0.981998 | 0.971364 | 0.743182 | 0.250000 |
| e3_4_vqc | 0.999493 | 0.868124 | 0.947727 | 0.000000 |
| e2_matched_focal_g2 | 0.982505 | 0.778448 | 0.000000 | 0.000000 |
| e2_matched_weighted_focal_g2 | 0.648327 | 0.984928 | 0.429545 | 0.000000 |
| e3_4_focal_g2 | 0.989097 | 0.763376 | 0.686364 | 0.000000 |
| e3_4_weighted_focal_g2 | 0.245943 | 0.990957 | 0.663636 | 0.000000 |

## Success Criterion

The rescue target is to keep E3-4 from losing about 10 points of early-stage recall versus the matched classical comparator.

## Interpretation

- Plain focal loss did not rescue early recall for E3-4; it lowered early recall and Macro F1.
- Weighted focal loss did rescue E3-4 early recall numerically, but at a major cost to accuracy and healthy-stage recall.
- None of the rescue runs recovered severe-stage recall; Stage 3 remains 0.0 for all focal variants.
- The weighted focal result should be treated as a tradeoff experiment, not a clean improved E3 model.

