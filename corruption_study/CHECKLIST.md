# Corruption Study Checklist

## Phase 1 - Audit

- [x] Identify existing E0/E1/E2/E2-8 evaluation paths.
- [x] Confirm selected checkpoint per model.
- [x] Confirm clean metrics per model.
- [x] Confirm E0 image geometry: resize 224x224, bilinear interpolation, no crop.
- [x] Confirm E1 feature-space columns and train-fitted scaler behavior.
- [x] Confirm E2 cached embedding layout: 1280 image + 64 IoT = 1344.
- [x] Confirm E2-8 checkpoint architecture and source script.
- [x] Write audit notes to `IMPLEMENTATION_REPORT.md`.

## Phase 2 - Materialization

- [x] Generate passthrough image set.
- [x] Generate MVP corrupted image sets for severities 2, 3, and 5.
  - [x] `gaussian_noise` for seeds 42, 123, 999.
  - [x] `brightness_increase` via old generated `brightness` folder and `brightness_increase` junction.
  - [x] `brightness_decrease`.
- [x] Generate MVP corrupted IoT feature CSVs for severities 2, 3, and 5.
- [x] Verify PNG output and sample count.
- [x] Verify IoT corruption touches only continuous features.
- [x] Regenerate clean TEST embeddings.
- [x] Compare regenerated clean TEST embeddings against existing clean E2/E2-8 metrics.
- [x] Regenerate MVP corrupted TEST embeddings for E2/E2-8.
  - [x] `image_corrupted_only/gaussian_noise` embeddings.
  - [x] `image_corrupted_only/brightness_increase` embeddings.
  - [x] `image_corrupted_only/brightness_decrease` embeddings.
  - [x] `iot_corrupted_only/sensor_noise` embeddings.
  - [x] `iot_corrupted_only/dropout` embeddings.

## Phase 3 - Smoke Test

- [x] Run severity-0 sanity check.
- [x] Run passthrough identity check.
- [x] Run one severity-2 corruption evaluation.
- [x] Confirm predictions CSV, confusion CSV, and metrics JSON are saved for passthrough and severity-0 clean checks.
- [x] Confirm no training is triggered.

## Phase 4 - Full Matrix

- [x] Expand image corruptions to all planned types for severities 2, 3, and 5.
- [x] Expand IoT corruptions to all planned types for severities 2, 3, and 5.
- [x] Regenerate all required corrupted E2/E2-8 embeddings.
- [x] Run MVP evaluation matrix.
- [x] Compute paired bootstrap CIs from saved predictions.
- [x] Regenerate `both_corrupted` E2/E2-8 embeddings for explicit representative pairings.
  - [x] Canonical: `gaussian_noise` + `sensor_noise` at severities 2, 3, and 5, with seeds 42, 123, 999.
  - [x] Data-driven worst-case distinct: `defocus_blur` + `drift` at severities 2, 3, and 5.
- [x] Run representative `both_corrupted` evaluations for E2-canonical and E2-8.
- [x] Run exploratory `both_corrupted` evaluations for e2_128_25.
- [x] Verify `both_corrupted` output names include both image and IoT corruption names.

## Phase 5 - Reporting

- [x] Generate `results/all_metrics.csv`.
- [x] Generate `results/robustness_summary.csv`.
- [x] Generate `results/robustness_summary.md`.
- [x] Generate `results/per_class_recall_by_condition.csv`.
- [x] Generate `results/severity0_sanity_check.md`.
- [x] Generate `results/passthrough_sanity_check.md`.
- [x] Add report artifact generator script.
- [x] Update `IMPLEMENTATION_REPORT.md`.

## Plan-Level Remaining Items

- [x] Apply clean-reference rule consistently:
  - [x] E0/E1 drops use severity-0 `clean`.
  - [x] E2-canonical/E2-8/e2_128_25 regenerated-embedding conditions use severity-0 `clean_regen`.
- [x] Add bootstrap CI fields to metrics/summaries:
  - [x] `accuracy_ci_low`, `accuracy_ci_high`.
  - [x] `macro_f1_ci_low`, `macro_f1_ci_high`.
  - [x] `accuracy_drop_ci_low`, `accuracy_drop_ci_high`.
  - [x] `macro_f1_drop_ci_low`, `macro_f1_drop_ci_high`.
- [x] Use paired bootstrap for drop CIs, aligned by `sample_id`.
- [x] Document bootstrap settings: 1000 resamples, 95% CI, fixed bootstrap seed, present-class macro-F1 handling.
- [x] Add missing final-table fields where practical:
  - [x] `macro_precision`.
  - [x] `weighted_f1`.
  - [x] `training_seed`.
  - [x] `checkpoint_path`.
  - [x] `config_path`.
  - [x] `timestamp`.
- [x] Generate final thesis-style robustness summary/report.
- [x] Update `README.md` to fully cover reproduction and interpretation requirements from the plan.
- [x] Update `IMPLEMENTATION_REPORT.md` after bootstrap and any `both_corrupted` runs.
- [x] Optional: generate degradation/drop plots from the summary CSVs.
- [x] Optional: bootstrap difference-of-drops between unimodal and multimodal models.
- [x] Optional: create secondary train-on-corrupted scaffold only if it does not delay primary results.

## E3 Corruption Extension - Supervisor Request

- [x] Repair `.venv-e1` so its base Python lives inside `.venv-e1`.
- [x] Add E3 model keys to `run_corruption_eval.py`.
- [x] Add serial E3 commands to `RUN_COMMANDS.md`.
- [x] Add parallel E3 runner for better CPU utilization.
- [x] Clean partial E3 generated outputs before rerun.
- [x] Run full E3 corruption matrix from a clean E3 output state.
- [x] Regenerate summaries, reports, bootstrap outputs, and plots after E3 runs complete.
- [x] Update thesis-style interpretation after E3 results are complete.
