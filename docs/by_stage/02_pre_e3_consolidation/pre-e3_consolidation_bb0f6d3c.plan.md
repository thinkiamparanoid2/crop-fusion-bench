---
name: Pre-E3 Consolidation
overview: Consolidate and lock the E0/E1/E2 baselines, fix documentation drift, add reproducibility artifacts, and extract the shared evaluation/model infrastructure plus the E2-vs-E3 parameter-matching spec so E3 can be built cleanly on a stable foundation.
todos:
  - id: lock-runs
    content: Declare canonical E0 (e0_manual_bs768_w0_20260602), E1 (e1_h128_64_nodrop_wd_bs512_sched), E2 (e2_frozen_e0_manual_e1_h128_64_fusion) and reference them everywhere
    status: completed
  - id: shared-module
    content: Extract shared thesis_eval (metrics/ECE/stage-recall/ROC-AUC/confusion/predictions) and shared E1MLP; refactor E0/E1/E2 to import it and verify metrics reproduce
    status: completed
  - id: param-spec
    content: "Write the E2-vs-E3 parameter-matching spec: shared encoders, 4-dim bottleneck, trainable-param accounting rule, locked first-pass VQC config"
    status: completed
  - id: ood-rerun
    content: Re-run evaluate_ood_corruptions.py against canonical E0 (bs768) and regenerate the OOD results doc
    status: completed
  - id: unified-doc
    content: Create docs/by_stage/03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md with metrics, per-stage recall, stage confusion matrices, and narrative
    status: completed
  - id: repro
    content: Add requirements.txt (verified vs .venv-e1), confirm/regenerate E1 feature CSVs, add a regenerate-from-scratch command section
    status: completed
  - id: fix-docs
    content: Update README, FOR_THESIS_CONTEXT, RUN_COMMANDS, and run_e0_e1_new_results.ps1 defaults to reflect E2 and canonical runs
    status: completed
  - id: paths
    content: Make E2 scripts emit repo-relative checkpoint paths in manifest/metrics JSON instead of hardcoded G:\ paths
    status: completed
isProject: false
---

## Current Status Note (2026-06-06)

This plan is completed and historical. The current active direction is
`docs/by_stage/06_e3_attack/E3_BEAT_E2_ATTACK_PLAN.md`; the safe fallback docs
are `docs/by_stage/07_final_claim_revision/` and
`docs/by_stage/08_reports_future_work/`.

# Pre-E3 Consolidation Plan

Current status: completed on 2026-06-03. See
`e2_step_notes/STEP_14_PRE_E3_CONSOLIDATION.md`,
`e2_step_notes/STEP_15_CANONICAL_RENAME_RELOCATE.md`, and
`docs/REPO_AUDIT.md` for the final canonical state.

Goal: reach a clean, locked, reproducible state for E0/E1/E2 and stand up the shared infrastructure E3 will reuse, before any quantum code is written.

## Context confirmed during research

- E2 fusion head already bakes in the 4-dim bottleneck designed for a 4-qubit VQC: `Linear(1344,128)-BN-ReLU-Dropout(0.1)-Linear(128,4)-ReLU-Linear(4,25)` in [e2_multimodal_fusion/scripts/train_e2_fusion.py](e2_multimodal_fusion/scripts/train_e2_fusion.py).
- The master plan mandates E2-vs-E3 must share bottleneck dim, label space, and be parameter-matched within +/-10% ([docs/by_stage/01_thesis_planning/THESIS_MASTER_PLAN_v4.md](../01_thesis_planning/THESIS_MASTER_PLAN_v4.md) sec 12.4; [docs/by_stage/01_thesis_planning/final_thesis_roadmap_v2.md](../01_thesis_planning/final_thesis_roadmap_v2.md) Phase 4).
- `metrics_dict` / ECE / stage-recall / ROC-AUC logic is duplicated across E0/E1/E2; `E1MLP` is duplicated in [e1_iot_only/scripts/train_e1_iot_mlp.py](e1_iot_only/scripts/train_e1_iot_mlp.py) and [e2_multimodal_fusion/scripts/extract_e2_embeddings.py](e2_multimodal_fusion/scripts/extract_e2_embeddings.py).

## 1. Lock canonical runs

Declare one canonical run per experiment and use it everywhere downstream:

- E0: `e0_manual_bs768_w0_20260602` (already wired into E2 extraction; Macro F1 0.964).
- E1: `e1_h128_64_nodrop_wd_bs512_sched` (already the E2 IoT encoder).
- E2: `e2_frozen_e0_manual_e1_h128_64_fusion`.

Record this decision in a single "canonical runs" section that the docs and scripts reference.

Historical note: [docs/by_stage/03_baselines_e0_e1_e2/E0_E1_RESULTS_COMPARISON.md](../03_baselines_e0_e1_e2/E0_E1_RESULTS_COMPARISON.md) and the old OOD doc cited the `bs256` E0 run, but E2 used the `bs768` manual run.

## 2. Re-run OOD on the canonical E0 (re-runs allowed)

[e0_image_only/learning/evaluate_ood_corruptions.py](e0_image_only/learning/evaluate_ood_corruptions.py) defaults `--run-name` to the `bs256` run. Re-run it against `e0_manual_bs768_w0_20260602` so robustness numbers match the canonical E0 the thesis (and E2) actually use, then regenerate [docs/by_stage/03_baselines_e0_e1_e2/E0_OOD_CORRUPTION_RESULTS.md](../03_baselines_e0_e1_e2/E0_OOD_CORRUPTION_RESULTS.md).

## 3. Unified E0/E1/E2 results doc

Create `docs/by_stage/03_baselines_e0_e1_e2/E0_E1_E2_RESULTS_COMPARISON.md` with:

- Test Acc / Macro F1 / Macro Recall / Early Recall / ECE / trainable params for E0, E1, E2.
- Per-stage recall table (highlight severe-stage recovery: E1 ~0.48 -> E2 ~0.82).
- Stage-level confusion matrices for all three.
- Short narrative: fusion beats both unimodal floors; sets the classical ceiling E3 must beat.

Supersede or clearly re-point the older `E0_E1` doc.

## 4. Shared evaluation + model module (standard scope)

Add a shared module (e.g. `common/thesis_eval.py` and `common/models.py` at repo root, importable like `config`) containing:

- `expected_calibration_error`, `stage_recall`, `multiclass_roc_auc`, `metrics_dict`, `per_class_metrics`, `save_confusion_matrix`, `save_predictions`.
- A single `E1MLP` definition with an `.embedding()` accessor.

Refactor E0/E1/E2 scripts to import from it (behavior-preserving; verify metrics reproduce the existing JSONs). This guarantees E3 is scored with byte-identical logic, which is mandatory for a credible E2-vs-E3 claim.

## 5. E2-vs-E3 parameter-matching spec

Write `e2_multimodal_fusion/PARAM_MATCH_SPEC.md` (or a section in the unified doc) that defines, before E3 exists:

- Shared frozen encoders (E0 1280-dim image, E1 64-dim IoT) and shared 1344-dim fused input.
- Shared projection to the 4-dim bottleneck; the VQC replaces only the post-bottleneck head.
- The exact trainable-parameter accounting rule (what counts toward the +/-10% budget). Flag the open issue: E2's 173,057 trainable params are dominated by `Linear(1344,128)`, while a 4-qubit VQC has only tens of params, so the match must be defined on the comparable sub-network, not the whole head.
- Locked first-pass quantum config from the plan: 4 qubits, angle encoding, 2-layer RX/RY + CNOT-ring ansatz, Pauli-Z readout, freeze-then-fine-tune schedule.

## 6. Reproducibility artifacts

- Add `requirements.txt` pinning the documented stack (torch 2.11+cu128, torchvision, numpy, pandas 2.3, scikit-learn 1.7). Verify versions against the live `.venv-e1` before writing.
- Confirm whether `e1_iot_only/features/train_iot_features.csv` / `test_iot_features.csv` exist locally (only smoke subsets are visible in the tree). If missing, document the regeneration command via [e1_iot_only/scripts/extract_iot_features.py](e1_iot_only/scripts/extract_iot_features.py).
- Add a "Regenerate from scratch" section: Data -> E1 features -> E0/E1 train -> E2 extract -> E2 train, with exact commands.

## 7. Fix documentation drift

- [README.md](README.md): add `e2_multimodal_fusion/` to structure and results pointers.
- [FOR_THESIS_CONTEXT.md](../00_context_audit/FOR_THESIS_CONTEXT.md): add E2 completion checkpoint, update "next action" to E3, fix stale paths (`THESIS_MASTER_PLAN_v4.md` now under `docs/by_stage/01_thesis_planning/`, no `e1_iot_only/FOR_THESIS_CONTEXT.md`).
- [RUN_COMMANDS.txt](RUN_COMMANDS.txt): add the E2 extract + train commands and canonical run names.
- Align defaults in [scripts/run_e0_e1_new_results.ps1](scripts/run_e0_e1_new_results.ps1) with the locked canonical run names.
- `docs/by_stage/01_thesis_planning/THESIS_MASTER_PLAN_v4.md` has been relabeled as historical and now
  notes that the active implementation uses real Dacon IoT sensor data.

## 8. Hardcoded-path cleanup (low risk)

[e2_multimodal_fusion/features/e2_embedding_manifest.json](e2_multimodal_fusion/features/e2_embedding_manifest.json) and the E2 metrics JSON store absolute `G:\FOR THESIS\MAIN\...` checkpoint paths. Note these are produced by scripts; making the scripts emit repo-relative paths prevents the same issue recurring for E3.

## Out of scope (deferred to after E3)

- Multi-seed variance runs and McNemar/bootstrap harness (per ablation order E6-E9 come after E3).
- Any quantum/PennyLane code.

## Suggested order

1 (lock) -> 4 (shared module) -> 5 (param spec) -> 2 (OOD re-run) -> 3 (unified doc) -> 6 (repro) -> 7 (docs) -> 8 (paths).
