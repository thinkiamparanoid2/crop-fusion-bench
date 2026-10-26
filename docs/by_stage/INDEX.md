# Thesis Markdown Index By Stage

This folder is the centralized home for thesis/project Markdown documents. Model-local `README.md` files stay next to their experiment code.

## Current Active Direction

As of 2026-06-09, the active direction is:

```text
09_corruption_robustness/CORRUPTION_ROBUSTNESS_HARNESS_PLAN.md
```

Supervisor pivot (2026-06-09): P2 implements up to E2 only (E0, E1, E2-canonical,
E2-8). E3 is proposed as architecture but not implemented. Focus shifts to the claim
that **multimodal fusion degrades more gracefully than unimodal models on corrupted
test data** — corruption applied to TEST data only, with a train-on-corrupted arm
scaffolded but not run.

Prior direction (2026-06-06), now superseded but kept for reference:

```text
06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md
E2-matched-8 test Macro F1 bar: 0.973356
Best E3 attack so far: e3_8_warm_s42 test Macro F1 0.956587
```

Stages `07_final_claim_revision` and `08_reports_future_work` remain safe
fallback/framing docs.

## 00 Context And Audit

- `00_context_audit/FOR_THESIS_CONTEXT.md` — latest working context and checkpoint history.
- `00_context_audit/REPO_AUDIT.md` — repository audit.

## 01 Thesis Planning

- `01_thesis_planning/THESIS_MASTER_PLAN_v4.md`
- `01_thesis_planning/final_thesis_roadmap_v2.md`

## 02 Pre-E3 Consolidation

- `02_pre_e3_consolidation/PRE_E3_CONSOLIDATION_PLAN.md`
- `02_pre_e3_consolidation/NEXT_SESSION_E3_HANDOFF.md`
- `02_pre_e3_consolidation/pre-e3_consolidation_bb0f6d3c.plan.md`

## 03 Baselines E0/E1/E2

- `03_baselines_e0_e1_e2/CANONICAL_RUNS.md`
- `03_baselines_e0_e1_e2/E0_E1_RESULTS_COMPARISON.md`
- `03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md`
- `03_baselines_e0_e1_e2/E0_OOD_CORRUPTION_RESULTS.md`
- `03_baselines_e0_e1_e2/E1_TRAINING_REPORT.md`
- `03_baselines_e0_e1_e2/E1_THESIS_STYLE_REPORT.md`
- `03_baselines_e0_e1_e2/PARAM_MATCH_SPEC.md`

## 04 E3 Quantum Fusion

- `04_e3_quantum_fusion/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md`
- `04_e3_quantum_fusion/E3_VQC_CIRCUIT_AND_LAYOUT.md`
- `04_e3_quantum_fusion/E2_MATCHED_E3_4_COMPARISON.md`

## 05 E3 Rescue, Bottleneck, Robustness

- `05_e3_rescue_bottleneck_robustness/E3_EARLY_RESCUE_COMPARISON.md`
- `05_e3_rescue_bottleneck_robustness/E3_QUANTUM_FUSION_TO_BOTTLENECK_SWEEP_CHANGES.md`
- `05_e3_rescue_bottleneck_robustness/E3_4_vs_E3_8_BOTTLENECK_COMPARISON.md`
- `05_e3_rescue_bottleneck_robustness/severe_stage_confusion.md`
- `05_e3_rescue_bottleneck_robustness/robustness_metrics.md`
- `05_e3_rescue_bottleneck_robustness/robustness_summary.md`
- `05_e3_rescue_bottleneck_robustness/severe_confusion_by_condition.md`
- `05_e3_rescue_bottleneck_robustness/robustness_metrics_mean_std.md`
- `05_e3_rescue_bottleneck_robustness/robustness_stability_summary.md`

## 06 E3 Attack

- `06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md`
- `06_e3_attack/ATTACK_RUN_COMPARISON.md`
- `06_e3_attack/MID_STAGE_ERROR_DIAGNOSTIC.md`

## 07 Final Claim Revision

- `07_final_claim_revision/FINAL_CLAIM_REVISION.md`
- `07_final_claim_revision/FINAL_EXPERIMENTAL_CONCLUSION.md`
- `07_final_claim_revision/P1_CLAIM_VS_RESULTS_TABLE.md`
- `07_final_claim_revision/SUPERVISOR_UPDATE_SUMMARY.md`
- `07_final_claim_revision/THESIS_RESULTS_NARRATIVE_DRAFT.md`

## 08 Reports And Future Work

- `08_reports_future_work/AMPLITUDE_ENCODING_FUTURE_WORK.md`
- `08_reports_future_work/WHY_QUANTUM_DEFENSE_MEMO.md`

## 09 Corruption Robustness

- `09_corruption_robustness/CORRUPTION_ROBUSTNESS_HARNESS_PLAN.md` — P2 corruption-robustness build plan / Codex prompt (TEST-only corruption, E0/E1/E2 degradation comparison).

## 10 Quantum Advantage Pivot

- `10_quantum_advantage_pivot/QUANTUM_ADVANTAGE_PIVOT_STARTER.md` - starting note for searching quantum advantage outside the current hybrid crop-disease E3 path, including candidate domains and datasets.
