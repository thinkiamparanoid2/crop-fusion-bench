# Supervisor Update Summary

## What We Expected

We expected the hybrid quantum-classical VQC model to improve accuracy, scalability, and computational efficiency compared with classical baselines.

## What We Found

E2-matched-8 is the best predictive classifier. E3-8 VQC does not beat E2-matched-8 in Accuracy, Macro F1, Macro Recall, or robustness.

E3-8 does keep consistently lower ECE across all IoT corruption seeds. The VQC showed a consistent calibration advantage.

After this safe conclusion was written, we reopened E3 in a separate attack branch. The first warm-start attack run improved E3 Macro F1 from 0.943716 to 0.956587, but it still did not beat E2-matched-8 at 0.973356. The latest active direction is the targeted attack plan in `docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md`.

## What Failed

The original accuracy-improvement expectation was not achieved. The VQC did not improve predictive performance over the matched classical comparator.

The 4D VQC bottleneck also failed severe-stage recognition. E3-4 had severe-stage recall 0.000000.

## What Still Worked

The VQC can process compressed multimodal image-IoT features.

The 8D bottleneck fixed severe-stage recall for E3-8, but the same recovery also happened for E2-matched-8. The severe-stage recovery was caused by increased bottleneck capacity.

E3-8 remained better calibrated than E2-matched-8 across all IoT corruption seeds.

## What Claim Must Be Revised

The thesis should not claim quantum advantage, accuracy improvement, robustness improvement, or severe-stage quantum advantage.

The thesis claim should shift from improvement to rigorous evaluation. The quantum hypothesis was partially supported only through calibration.

## Next Step

Recommended next action:

- Continue the focused pair-targeted attack run only.
- Keep E2-matched-8 as the strongest predictive model unless the attack branch beats Macro F1 0.973356.
- Next active experiment is `e3_8_warm_pair_s42`; Section 13.2.5 also defines a fair residual shortcut follow-up if pair-loss does not close the gap.
- Keep the claim-revision package as the safe fallback for thesis writing.
- Stop broader VQC tuning if the pair-targeted branch does not justify the runtime.
