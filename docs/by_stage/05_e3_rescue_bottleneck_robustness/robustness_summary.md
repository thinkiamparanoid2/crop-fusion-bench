# IoT Robustness Summary

Only the 64-dimensional IoT embedding slice was corrupted. Image embeddings, labels, splits, and weights were unchanged.

## Clean Baselines

| Model | Accuracy | Macro F1 | Early recall | Mid recall | Severe recall | ECE-10 | Avg confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| e2_matched_8 | 0.987515 | 0.973356 | 0.981161 | 0.936364 | 0.803571 | 0.247842 | 0.739729 |
| e3_8_vqc | 0.973643 | 0.943716 | 0.987189 | 0.802273 | 0.803571 | 0.118466 | 0.855178 |

## Worst Drops vs Clean

### e2_matched_8

- Worst Macro F1 condition: `shuffled_iot_20` (0.908241, drop -0.065114).
- Worst severe-recall condition: `shuffled_iot_20` (0.785714, drop -0.017857).
- Worst ECE condition: `missing_20` (0.249597, change 0.001756).

### e3_8_vqc

- Worst Macro F1 condition: `shuffled_iot_20` (0.728120, drop -0.215596).
- Worst severe-recall condition: `shuffled_iot_20` (0.678571, drop -0.125000).
- Worst ECE condition: `missing_20` (0.132104, change 0.013638).

## Interpretation Prompts

- If both models degrade similarly under IoT corruption, the 8D recovery is mostly robust to noisy sensor embeddings.
- If E3-8 keeps lower ECE under corruption, that supports a calibration-oriented quantum result.
- If severe recall collapses mainly under shuffled/contradictory IoT, then severe recognition depends on sensor-image consistency.
