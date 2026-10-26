# E3 Beat E2 Attack

This folder is an isolated continuation of the E3 work. It does not replace the
completed bottleneck sweep or the final claim-revision package.

Goal:

```text
Try to make E3-8 beat E2-matched-8 on test Macro F1.
Current bar: E2-matched-8 test Macro F1 = 0.973356.
```

Fairness constraints:

- Use the same frozen E2 embedding files.
- Use the same train/val/test split.
- Use the same 25-class order.
- Keep the 8D bottleneck and 64 post-bottleneck parameter budget for the first attack phase.
- Do not count calibration or early recall as a win unless test Macro F1 also wins.

Implemented first phase:

- `scripts/diagnose_mid_stage_errors.py`
- `scripts/train_warmstart_e3.py`
- `scripts/compare_attack_runs.py`

Current status:

- Diagnostic complete: E3's mid-stage failure is dominated by `3_a9_2 -> 3_a9_1`.
- Full warm-start seed 42 complete: best Macro F1 improved from `0.943716` to `0.956587`.
- E2-matched-8 is still ahead: Macro F1 bar remains `0.973356`.
- Next run is pair-targeted warm-start: `e3_8_warm_pair_s42`.
- If pair-loss does not close the gap, the next fair architecture test is the
  residual bottleneck shortcut from attack-plan Section 13.2.5:
  `e3_8_warm_residual_s42` vs `e2_matched_8_residual_s42`.

Outputs go under:

```text
e3_beat_e2_attack/results/
e3_beat_e2_attack/checkpoints/
```
