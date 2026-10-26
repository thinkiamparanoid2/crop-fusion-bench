# E3-4 vs E3-8 Bottleneck Comparison

This report compares the completed 4D bottleneck E2/E3 runs against the new 8D bottleneck sweep.

## Metrics

| Model | Run | Bottleneck | Qubits | VQC layers | Best epoch | Device | Params | Post params | Accuracy | Macro F1 | Macro recall | Early recall | Mid recall | Severe recall | ECE |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E2-matched-4 | e2_matched_4 | 4 | n/a | n/a | 44 | cuda | 173073 | 16 | 0.924224 | 0.661438 | 0.691956 | 0.971364 | 0.743182 | 0.250000 | 0.336519 |
| E3-4 | e3_4_vqc | 4 | 4 | n/a | 53 | cpu | 173073 | 16 | 0.896133 | 0.719661 | 0.741060 | 0.868124 | 0.947727 | 0.000000 | 0.263169 |
| E2-matched-8 | e2_matched_8 | 8 | n/a | n/a | 11 | cuda | 173737 | 64 | 0.987515 | 0.973356 | 0.971295 | 0.981161 | 0.936364 | 0.803571 | 0.248338 |
| E3-8 | e3_8_vqc | 8 | 8 | 4 | 14 | cpu | 173737 | 64 | 0.973643 | 0.943716 | 0.944777 | 0.987189 | 0.802273 | 0.803571 | 0.118487 |

## Key Deltas

| Comparison | Accuracy | Macro F1 | Macro recall | Early recall | Mid recall | Severe recall | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2-matched-8 minus E2-matched-4 | 0.063291 | 0.311917 | 0.279338 | 0.009797 | 0.193182 | 0.553571 | -0.088182 |
| E3-8 minus E3-4 | 0.077510 | 0.224055 | 0.203717 | 0.119066 | -0.145455 | 0.803571 | -0.144682 |
| E3-8 minus E2-matched-8 | -0.013872 | -0.029639 | -0.026517 | 0.006029 | -0.134091 | 0.000000 | -0.129850 |

## Severe-Stage Confusion

See also `e3_bottleneck_sweep/results/severe_stage_confusion.csv`.

| Run | True severe count | Pred healthy | Pred early | Pred mid | Pred severe | Severe recall |
|---|---:|---:|---:|---:|---:|---:|
| e2_matched_4 | 56 | 14 | 28 | 0 | 14 | 0.250000 |
| e3_4_vqc | 56 | 0 | 4 | 52 | 0 | 0.000000 |
| e2_matched_8 | 56 | 0 | 5 | 6 | 45 | 0.803571 |
| e3_8_vqc | 56 | 0 | 9 | 2 | 45 | 0.803571 |

## Interpretation

- The 8D bottleneck recovered early recall for E3: E3-4 early recall was 0.868124, while E3-8 reached 0.987189.
- The 8D bottleneck recovered severe recall for E3: E3-4 severe recall was 0.000000, while E3-8 reached 0.803571.
- The classical E2-matched-8 comparator recovered severe recall to the same 0.803571 and achieved stronger Macro F1/accuracy than E3-8. This suggests the severe-stage recovery is primarily a bottleneck-capacity effect, not uniquely a quantum-layer effect.
- E3-8 has slightly higher early recall than E2-matched-8 and substantially lower ECE, but lower accuracy, Macro F1, macro recall, and mid-stage recall.
- The fair conclusion is that the 4D bottleneck was too compressed for severe-stage recognition. The 8D quantum layer is viable and better calibrated, but the strongest 8D model in this sweep is the classical matched comparator.

