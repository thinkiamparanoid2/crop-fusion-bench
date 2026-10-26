# Step 14 - Pre-E3 Consolidation

Date: 2026-06-03

Completes the pre-E3 consolidation plan (all 8 items). Items 6, 7, 8 were done in
a prior session (reproducibility + doc-drift pass); items 1, 2, 3, 4, 5 were done
here.

## Item 1 - Lock canonical runs

- Verified all canonical artifacts exist on disk (E0 `.pth`, E1 `.pt`, E2 `.pt`, E0 metrics JSON).
- Created `docs/results/CANONICAL_RUNS.md` as the single source of truth (run names, repo-relative checkpoints, metrics JSON paths, trainable params, canonical val/test metrics, non-canonical bs256 caveat).
- Flagged `docs/results/E0_E1_RESULTS_COMPARISON.md` E0 column as the non-canonical bs256 run and linked the canonical doc.
- Pointed `docs/results/E0_E1_E2_RESULTS_COMPARISON.md`, `README.md`, and `RUN_COMMANDS.txt` at `CANONICAL_RUNS.md`.

Canonical runs:

```text
E0: e0_bs768_mobilenetv3large (2,214,985 params)
E1: e1_bs512_mlp128x64        (17,305 params)
E2: e2_bs512_fusionmlp        (173,057 params)
```

## Item 4 - Shared eval/model module

- Created repo-root `common/` package: `__init__.py`, `thesis_eval.py`, `models.py`.
- `common/thesis_eval.py`: count_trainable_params, label_to_stage, expected_calibration_error, stage_recall, multiclass_roc_auc, metrics_dict, per_class_metrics, save_confusion_matrix, save_predictions.
- `common/models.py`: single `E1MLP` with `.embedding()`.
- Refactored 6 consumers to import from `common`:
  - `e0_image_only/learning/train.py`
  - `e0_image_only/learning/evaluate_ood_corruptions.py`
  - `e1_iot_only/scripts/train_e1_iot_mlp.py`
  - `e1_iot_only/scripts/evaluate_e1_checkpoint.py`
  - `e2_multimodal_fusion/scripts/extract_e2_embeddings.py`
  - `e2_multimodal_fusion/scripts/train_e2_fusion.py`
- Standardized ECE on the E0/E2 left-closed bin convention (E1 previously used right-closed).
- Lint clean on all 8 files; all modules import cleanly with the new `common` wiring.
- Behavior-preserving verification: re-evaluated canonical E1 and E2 checkpoints through the shared logic; every scoring metric reproduced exactly (max abs diff 2.6e-9, in loss only). Temp verification script deleted after use.

## Item 5 - Parameter-matching spec

- Created `e2_multimodal_fusion/PARAM_MATCH_SPEC.md`.
- Verified counts after the later Spec B rename/relocation pass: E2-current = 173,057;
  E2-matched = E3-4 = 173,073 total; decision sub-network 16 vs 16
  (classical Linear(4,4,bias=False) vs 4-qubit/2-layer VQC).
- Locked first-pass VQC config (4 qubits, angle encoding, 2 layers, RX/RY, CNOT ring, PauliZ readout, 16 params, default.qubit) and the per-width qubit-sweep policy.

## Item 2 - OOD re-run on canonical E0

- Edited `evaluate_ood_corruptions.py`: default to canonical bs768 run + added `--seed` (reproducible noise corruptions).
- Ran OOD eval on canonical E0 `e0_bs768_mobilenetv3large` (7 conditions, 5,767 images).
- Regenerated `docs/results/E0_OOD_CORRUPTION_RESULTS.md` with canonical bs768 numbers (was bs256).
- Clean-row stage recalls match canonical E0 exactly; ~2e-4 accuracy gap is TF32-vs-FP32 (noted in doc).

## Item 3 - Unified results doc

- Added per-stage test recall table and stage-level confusion matrices (E0/E1/E2) to `docs/results/E0_E1_E2_RESULTS_COMPARISON.md`.
- Added shared-metrics note and a "Toward E3" section linking the param spec and OOD doc.
- Stage tables computed from canonical test-prediction CSVs and cross-checked against canonical JSONs. Temp script deleted after use.

## Bookkeeping

- Updated `FOR_THESIS_CONTEXT.md` with two checkpoint entries (items 1+4, then 2+3+5) and rewrote "Next Recommended Action" to point at E3 scaffolding.
- Downstream E2/E3 now reads canonical snapshots by default; training outputs remain separate and must be promoted before becoming canonical.

## Net file summary

- Created (5): `common/__init__.py`, `common/thesis_eval.py`, `common/models.py`, `docs/results/CANONICAL_RUNS.md`, `e2_multimodal_fusion/PARAM_MATCH_SPEC.md`.
- Generated (2): `e0_manual_bs768_w0_20260602_ood_corruptions.json` + `.csv`.
- Edited (11): 6 refactored scripts + `E0_OOD_CORRUPTION_RESULTS.md`, `E0_E1_E2_RESULTS_COMPARISON.md`, `E0_E1_RESULTS_COMPARISON.md`, `README.md`, `RUN_COMMANDS.txt`, `FOR_THESIS_CONTEXT.md`.
- Created then deleted (2): verification + stage-table temp scripts.

## Notes

- Did NOT retrain E0/E1/E2 end-to-end. Behavior preservation was verified by re-evaluating existing checkpoints (E1/E2 exactly; E0 via the OOD clean row). A full retrain of one experiment would additionally confirm the training path.

## Next step

- E3 scaffolding per `e2_multimodal_fusion/PARAM_MATCH_SPEC.md` and `docs/plans/E3_STRUCTURE_AND_IMPLEMENTATION_PLAN.md`: build E2-matched comparator and E3-4 (VQC), reuse the frozen E2 embedding interface, score with `common/thesis_eval.py`.
