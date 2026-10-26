# Thesis Results Narrative Draft

## 1. Classical vs Quantum 4D Comparison

The first controlled E3 comparison used a 4D bottleneck so that the classical comparator and the VQC model had matched trainable parameter counts. E2-matched-4 achieved test Accuracy 0.924224, Macro F1 0.661438, early-stage recall 0.971364, and severe-stage recall 0.250000. E3-4 achieved test Accuracy 0.896133, Macro F1 0.719661, early-stage recall 0.868124, and severe-stage recall 0.000000.

This result did not support an accuracy improvement from the VQC. E3-4 improved Macro F1 over E2-matched-4, but it lost early-stage recall and completely failed severe-stage recall. The 4D comparison showed that the VQC could operate on compressed multimodal features, but the compression was too restrictive for reliable disease-stage separation.

## 2. Bottleneck Sweep From 4D To 8D

The bottleneck sweep tested whether the 4D representation was too compressed. Both the classical matched model and the VQC model were expanded from 4D to 8D while keeping the post-bottleneck parameter count matched. E2-matched-8 and E3-8 each used 173,737 trainable parameters and 64 post-bottleneck decision parameters.

Increasing the bottleneck size substantially improved both models. E2-matched-8 reached test Accuracy 0.987515 and Macro F1 0.973356. E3-8 reached test Accuracy 0.973643 and Macro F1 0.943716. The improvement from 4D to 8D shows that bottleneck capacity was a major constraint in the earlier E3 result.

## 3. Severe-Stage Confusion Analysis

The severe-stage confusion analysis showed the clearest bottleneck effect. E3-4 had 56 true severe samples, but predicted 52 of them as mid-stage and 4 as early-stage, giving severe recall 0.000000. After increasing to 8D, E3-8 correctly predicted 45 of 56 severe samples, giving severe recall 0.803571.

However, E2-matched-8 also recovered severe recall to 0.803571. Because both the classical and quantum models recovered severe-stage recognition after the bottleneck was expanded, the severe-stage recovery was caused by increased bottleneck capacity. It should not be described as a severe-stage quantum advantage.

## 4. E2-Matched-8 vs E3-8 Final Comparison

The final matched comparison favors the classical model on predictive performance. E2-matched-8 achieved higher Accuracy, Macro F1, Macro Recall, and mid-stage recall than E3-8. E3-8 achieved slightly higher early-stage recall, matched severe-stage recall, and had much lower ECE.

The VQC did not improve predictive performance over the matched classical comparator. E2-matched-8 should be reported as the strongest predictive classifier. E3-8 should be reported as the best calibrated hybrid model.

## 5. IoT Robustness Stability

The IoT robustness evaluation corrupted only the IoT embedding slice at test time while keeping image embeddings, labels, splits, sample order, and weights unchanged. The multi-seed stability check used corruption seeds 13, 37, 73, 101, and 202.

Across the robustness conditions, E3-8 did not beat E2-matched-8 in average Macro F1. Under `shuffled_iot_20`, E2-matched-8 had a smaller Macro F1 degradation than E3-8: E2 dropped by -0.058441, while E3 dropped by -0.222409. E2-matched-8 also had smaller severe-recall degradation under `shuffled_iot_20`: E2 changed by +0.010714, while E3 dropped by -0.060714.

This means the VQC should not be described as more robust to IoT corruption. The earlier robustness result is stable across corruption randomness: E3 remains better calibrated, while E2 remains stronger and less degraded in predictive metrics under shuffled IoT.

## 6. Calibration Result

The strongest positive E3 finding is calibration. E3-8 had lower ECE than E2-matched-8 in the clean test condition and kept lower ECE across all IoT corruption seeds. This supports a limited calibration claim.

The allowed interpretation is: The VQC showed a consistent calibration advantage. This should not be extended into a claim of quantum advantage, accuracy improvement, robustness improvement, or severe-stage quantum advantage.

## 7. Final Interpretation

The final results partially support the quantum hypothesis only through calibration. They do not support the original expectation that the hybrid quantum-classical model would improve predictive accuracy over a strong matched classical comparator.

The thesis remains valid because it provides a controlled evaluation of a hybrid QML classifier under NISQ bottleneck constraints. The final thesis framing should shift from improvement to rigorous evaluation. The correct final claim is that the tested VQC is feasible and better calibrated, but not more accurate or more robust than the matched classical baseline.

## 8. Active Attack Branch Status

After the safe final claim package was created, E3 was reopened in `e3_beat_e2_attack/` to test whether the VQC gap can be closed without changing the frozen embedding interface or class order. The diagnostic found that E3-8's mid-stage failure is dominated by one progression collapse: `3_a9_2 -> 3_a9_1`.

The first full warm-start run, `e3_8_warm_s42`, copied the E2-matched-8 projection and readout into E3-8 and trained the VQC block. It improved Macro F1 from the original E3-8 value of 0.943716 to 0.956587, but it still did not beat the E2-matched-8 bar of 0.973356. The next targeted experiment is `e3_8_warm_pair_s42`, which adds progression-pair loss for `3_a9_2` versus `3_a9_1`. The attack plan also adds a residual bottleneck shortcut in Section 13.2.5 as the next fair architectural test if pair-loss does not close the gap.

## Final Decision

Recommended next action:

- Treat the claim-revision package as the safe fallback.
- Continue only the focused attack branch that targets the diagnosed pair collapse.
- Use `docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md` as the active direction; 07/08 remain fallback/framing docs.
- Do not claim quantum advantage unless an attack run beats the matched classical Macro F1 bar.
- Move to thesis writing when the targeted attack branch is exhausted or too expensive.
