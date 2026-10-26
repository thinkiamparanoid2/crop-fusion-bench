# Final Claim Revision

## Status

The original P1 claim expected the hybrid quantum-classical model to improve accuracy, scalability, and computational efficiency compared with classical baselines. The completed E3 experiments do not support that claim as written.

This package is the thesis-safe fallback if the active `e3_beat_e2_attack` experiments do not beat `E2-matched-8`. It should not be read as a command to stop the currently reopened E3 attack branch. The latest active direction remains `docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md`.

## Revised Thesis Claim

The thesis claim should shift from improvement to rigorous evaluation.

Recommended revised claim:

> This thesis evaluates whether a hybrid quantum-classical variational quantum circuit can serve as a compact multimodal fusion head for crop disease stage classification under NISQ bottleneck constraints. The results show that the VQC did not improve predictive performance over the matched classical comparator, but it showed a consistent calibration advantage. The quantum hypothesis was partially supported only through calibration.

## What The Results Support

- The VQC can process compressed multimodal image-IoT features.
- The VQC did not improve predictive performance over the matched classical comparator.
- E2-matched-8 is the strongest predictive classifier.
- E3-8 is the best calibrated hybrid model.
- The severe-stage recovery was caused by increased bottleneck capacity.
- The quantum hypothesis was partially supported only through calibration.

## What Must Not Be Claimed

- Do not claim quantum advantage.
- Do not claim accuracy improvement from the VQC.
- Do not claim robustness improvement from the VQC.
- Do not claim severe-stage quantum advantage.
- Do not claim the 8D severe-stage recovery is uniquely quantum.

## Final Decision

Recommended next action:

- Continue the focused E3 attack branch only while it targets the known failure mode (`3_a9_2 -> 3_a9_1`).
- Use `E2-matched-8` as the strongest predictive model unless an attack run beats Macro F1 `0.973356`.
- Current best attack run is `e3_8_warm_s42` at Macro F1 `0.956587`; the active next run is `e3_8_warm_pair_s42`.
- Keep this claim-revision package as the safe final thesis framing if the attack branch fails.
- Move to thesis writing once the pair-targeted attack branch is exhausted or too expensive.
