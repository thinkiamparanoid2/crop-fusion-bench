# IoT Robustness Stability Summary

Corruption seeds: 13, 37, 73, 101, 202.

## Required Answers

- Does E3-8 keep lower ECE across all corruption seeds? Yes.
- Does E3-8 ever beat E2-matched-8 in Macro F1 on average? No.
- Which model has smaller Macro F1 degradation under shuffled_iot_20? e2_matched_8 (E2 drop -0.058441; E3 drop -0.222409).
- Which model has smaller severe recall degradation under shuffled_iot_20? e2_matched_8 (E2 drop 0.010714; E3 drop -0.060714).
- Is the earlier result stable across corruption randomness? Yes: E3 remains better calibrated, while E2 remains stronger and less degraded on Macro F1/severe recall under shuffled IoT.
