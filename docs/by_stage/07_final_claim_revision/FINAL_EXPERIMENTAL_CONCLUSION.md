# Final Experimental Conclusion

The VQC did not outperform the parameter-matched classical comparator on the primary metric. E3-8 did not beat E2-matched-8 in Accuracy, Macro F1, Macro Recall, or robustness.

This conclusion is the current thesis-safe fallback. A separate `e3_beat_e2_attack` branch is active and is testing whether warm-start, progression-pair loss, and the residual bottleneck shortcut in Section 13.2.5 can close the gap. Until that branch beats the E2-matched-8 Macro F1 bar, the conclusion below remains the safe interpretation.

E2-matched-8 should be reported as the strongest predictive model. It achieved the best final predictive performance among the matched bottleneck models, with test Accuracy 0.987515, Macro F1 0.973356, and Macro Recall 0.971295.

E3-8 should be reported as the best calibrated hybrid model. It had lower ECE than E2-matched-8 in the clean test condition and kept lower ECE across all IoT corruption seeds in the robustness stability check.

The original accuracy-improvement expectation was not achieved. The VQC did not improve predictive performance over the matched classical comparator.

The 4D bottleneck was too compressed for severe-stage recognition. E3-4 had severe-stage recall 0.000000. Increasing the bottleneck to 8D recovered severe recall to 0.803571 for both E2-matched-8 and E3-8. Therefore, the severe-stage recovery was caused by increased bottleneck capacity, not by a uniquely quantum effect.

The thesis remains valid as a rigorous evaluation of hybrid QML under NISQ bottleneck constraints. The final conclusion should be framed as an empirical boundary result: the tested VQC was feasible and better calibrated, but it was not more accurate, more robust, or better at severe-stage recognition than the matched classical head.

## Final Decision

Recommended next action:

- Continue only targeted E3 attack runs that address the observed `3_a9_2 -> 3_a9_1` failure.
- Keep E2-matched-8 as the strongest predictive model unless an attack run exceeds Macro F1 `0.973356`.
- Treat `docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md` as the active direction.
- Keep E3-8 as the best calibrated hybrid model unless newer attack runs change calibration results.
- Move to thesis writing once the targeted attack branch is exhausted or no longer worth the runtime.
