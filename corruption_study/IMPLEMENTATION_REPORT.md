# Implementation Report

Created: 2026-06-09
Last updated: 2026-06-09

## Current Status

The MVP corruption-robustness harness is complete through report-artifact generation.

No model training was run.
No validation or training data was corrupted.
No E3/VQC/quantum code was added for the P2 corruption harness.

Completed MVP scope:

```text
image corruptions: gaussian_noise, brightness_increase, brightness_decrease
IoT corruptions:   sensor_noise, dropout
severities:        2, 3, 5
corruption seeds:  42, 123, 999 for stochastic corruptions
models:            E0, E1, E2-canonical, E2-8
optional model:    E2 128-to-25 exploratory variant
```

Current saved output coverage:

```text
metrics JSON files:    141
prediction CSV files:  141
confusion CSV files:   141
all_metrics.csv rows:  141
robustness_summary.csv rows: 60
per_class_recall_by_condition.csv rows: 3525
```

Severity-0 clean checks and the E0 passthrough image identity check pass within the
configured `0.005` absolute tolerance.

Remaining full-plan work, if needed later:

```text
both_corrupted pairings: not run
paired bootstrap confidence intervals: not yet computed
```

Primary severity scope was updated by user request:

```text
clean sanity severity: 0
corrupted severities: 2, 3, 5
deferred if time allows: 1
not currently defined: 6
```

Severity 6 is not in the current image/IoT schedules and should only be added after explicit
severity-6 parameters are defined for every corruption type.

## Files Created

```text
corruption_study/README.md
corruption_study/CHECKLIST.md
corruption_study/RUN_COMMANDS.md
corruption_study/IMPLEMENTATION_REPORT.md
corruption_study/gen/corrupt_images.py
corruption_study/gen/corrupt_iot.py
corruption_study/gen/regen_embeddings.py
corruption_study/eval/run_corruption_eval.py
corruption_study/eval/summarize_results.py
corruption_study/eval/generate_report_artifacts.py
corruption_study/results/.gitkeep
corruption_study/results/confusion/.gitkeep
corruption_study/results/predictions/.gitkeep
corruption_study/data/corrupted_images/.gitkeep
corruption_study/data/corrupted_iot/.gitkeep
corruption_study/data/corrupted_embeddings/.gitkeep
```

## Reuse Policy

The study reuses existing clean-trained artifacts:

```text
E0: e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth
E1: e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt
E2-canonical: e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt
E2-8: e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt
```

E2/E2-8 corrupted conditions require regenerated TEST embeddings from corrupted TEST inputs
through frozen E0/E1 encoders. This is inference only, not retraining.

## Existing Scripts And Evaluation Paths

```text
E0 training/evaluation source:
  e0_image_only/learning/train.py
  e0_image_only/learning/evaluate_ood_corruptions.py

E1 training/evaluation source:
  e1_iot_only/scripts/train_e1_iot_mlp.py
  e1_iot_only/scripts/evaluate_e1_checkpoint.py

E2-canonical training/evaluation source:
  e2_multimodal_fusion/scripts/extract_e2_embeddings.py
  e2_multimodal_fusion/scripts/train_e2_fusion.py

E2-8 source:
  e3_bottleneck_sweep/scripts/train_bottleneck.py
```

`corruption_study/eval/run_corruption_eval.py` reuses the same architecture definitions/patterns
for inference. It defines the E2-8 classical bottleneck locally to avoid importing quantum/VQC
code into the P2 corruption harness.

## Selected Checkpoints

```text
E0:
  e0_image_only/canonical/best_e0_bs768_mobilenetv3large.pth

E1:
  e1_iot_only/canonical/best_e1_bs512_mlp128x64.pt

E2-canonical:
  e2_multimodal_fusion/canonical/best_e2_bs512_fusionmlp.pt

E2-8:
  e3_bottleneck_sweep/checkpoints/best_e2_matched_8.pt
```

Only one selected canonical/checkpoint path is currently used per model in this harness.
Other historical result files exist, but the corruption study should not retrain or switch
checkpoints between conditions.

## Clean Test Metrics Found

```text
E0 canonical:
  test accuracy: 0.9821397607074736
  test Macro F1: 0.9638961355829906
  test macro recall: 0.9626670325082501
  test early-stage recall: 0.9804069329314242
  test ECE: 0.007804906629742942

E1 canonical:
  test accuracy: 0.9436448760187273
  test Macro F1: 0.8833699043245444
  test macro recall: 0.8826416953441898
  test early-stage recall: 0.9397136397889977
  test ECE: 0.0165426628375066

E2-canonical:
  test accuracy: 0.9873417721518988
  test Macro F1: 0.9679255699042882
  test macro recall: 0.9704933830323432
  test early-stage recall: 0.9864355689525245
  test ECE: 0.006775465358041582

E2-8:
  test accuracy: 0.9875151725333796
  test Macro F1: 0.9733555279688368
  test macro recall: 0.9712945410764199
  test early-stage recall: 0.9811605124340618
  test ECE: 0.24833769909230058
```

Severity-0 clean rerun metrics for all four models were completed by the user and verified.
They reproduce the audited clean accuracy and Macro F1 values exactly for E0, E1,
E2-canonical, and E2-8.

## Data And Split Counts

```text
Data/train.csv: 46,713 rows
Data/val.csv:    5,193 rows
Data/test.csv:   5,767 rows
```

All four audited model metric JSONs report 25 classes.

## Image Loader Geometry

E0 and E2 embedding extraction both use:

```text
read image as RGB tensor
resize to 224 x 224
interpolation: torchvision.transforms.InterpolationMode.BILINEAR
no center crop found
normalization mean: [0.485, 0.456, 0.406]
normalization std:  [0.229, 0.224, 0.225]
```

Relevant files:

```text
e0_image_only/learning/train.py
e2_multimodal_fusion/scripts/extract_e2_embeddings.py
```

## IoT Feature Space

E1 and E2 use extracted feature-space IoT summaries, not raw time-series files during
training/evaluation.

Metadata columns excluded from model features:

```text
image
label
crop
disease
stage
n_rows
```

Continuous IoT feature columns used by E1/E2: 54 columns, matching
`e2_multimodal_fusion/features/e2_embedding_manifest.json`.

Scaler behavior:

```text
MinMaxScaler is fit on aligned train split features only.
Validation and test features are transformed with the train-fitted scaler.
```

Relevant files:

```text
e1_iot_only/scripts/train_e1_iot_mlp.py
e1_iot_only/scripts/evaluate_e1_checkpoint.py
e2_multimodal_fusion/scripts/extract_e2_embeddings.py
```

## E2 Embedding Interface

```text
Cached clean embeddings:
  e2_multimodal_fusion/features/test_embeddings.npz

image_embedding_dim: 1280
iot_embedding_dim: 64
fusion_input_dim: 1344
```

E2 corrupted conditions must regenerate TEST embeddings from corrupted TEST inputs via frozen
E0/E1 encoders. Cached clean embeddings must not be used under corrupted conditions.

## Existing Robustness Utilities Found

```text
e0_image_only/learning/evaluate_ood_corruptions.py
e0_image_only/canonical/e0_bs768_mobilenetv3large_ood_corruptions.json
e0_image_only/canonical/e0_bs768_mobilenetv3large_ood_corruptions.csv
e3_bottleneck_sweep/scripts/evaluate_iot_robustness.py
e3_bottleneck_sweep/iot_robustness_stability/evaluate_iot_robustness_stability.py
e3_bottleneck_sweep/results/iot_robustness/
e3_bottleneck_sweep/results/iot_robustness_stability/
```

These are references only. The P2 corruption study should stay in `corruption_study/` and
should not implement or run E3/VQC code.

## Commands Run For Audit

Only file-inspection commands were run, including `Get-Content`, `Get-ChildItem`, `Import-Csv`,
and `rg`. No Python harness command was run.

## Phase 2 Passthrough Checkpoint

User reported passthrough generation complete. Verified by file count:

```text
corruption_study/data/corrupted_images/passthrough/*.png recursive count: 5,767
Data/test.csv rows: 5,767
```

This confirms the materialized passthrough image set exists and matches the test manifest row
count.

Passthrough E0 evaluation was then completed by the user and verified from saved outputs:

```text
metrics:
  corruption_study/results/e0__image_passthrough__sev0__metrics.json
predictions:
  corruption_study/results/predictions/e0__image_passthrough__sev0__predictions.csv
confusion:
  corruption_study/results/confusion/e0__image_passthrough__sev0__confusion.csv

E0 clean baseline:
  accuracy:  0.9821397607074736
  Macro F1: 0.9638961355829906

E0 passthrough:
  accuracy:  0.9824865614704352
  Macro F1: 0.9649755376138717

absolute deltas:
  accuracy:  0.0003468007629616
  Macro F1: 0.0010794020308811

passthrough tolerance:
  0.005

status:
  PASS
```

## Severity-0 Clean Sanity Check

User ran severity-0 clean evaluation for all four models. Verified outputs:

```text
metrics:
  corruption_study/results/e0__clean__sev0__metrics.json
  corruption_study/results/e1__clean__sev0__metrics.json
  corruption_study/results/e2_canonical__clean__sev0__metrics.json
  corruption_study/results/e2_8__clean__sev0__metrics.json

predictions:
  corruption_study/results/predictions/e0__clean__sev0__predictions.csv
  corruption_study/results/predictions/e1__clean__sev0__predictions.csv
  corruption_study/results/predictions/e2_canonical__clean__sev0__predictions.csv
  corruption_study/results/predictions/e2_8__clean__sev0__predictions.csv

confusion:
  corruption_study/results/confusion/e0__clean__sev0__confusion.csv
  corruption_study/results/confusion/e1__clean__sev0__confusion.csv
  corruption_study/results/confusion/e2_canonical__clean__sev0__confusion.csv
  corruption_study/results/confusion/e2_8__clean__sev0__confusion.csv
```

Metrics:

```text
E0:
  accuracy 0.9821397607074736
  Macro F1 0.9638961355829906
  status PASS

E1:
  accuracy 0.9436448760187273
  Macro F1 0.8833699043245444
  status PASS

E2-canonical:
  accuracy 0.9873417721518988
  Macro F1 0.9679255699042882
  status PASS

E2-8:
  accuracy 0.9875151725333796
  Macro F1 0.9733555279688368
  status PASS
```

## Next Step

## MVP Image Corruption Checkpoint

User ran the old MVP image command before brightness was split. Verified generated files:

```text
gaussian_noise:
  sev_2 cseed_42:  5,767 PNGs
  sev_2 cseed_123: 5,767 PNGs
  sev_2 cseed_999: 5,767 PNGs
  sev_3 cseed_42:  5,767 PNGs
  sev_3 cseed_123: 5,767 PNGs
  sev_3 cseed_999: 5,767 PNGs
  sev_5 cseed_42:  5,767 PNGs
  sev_5 cseed_123: 5,767 PNGs
  sev_5 cseed_999: 5,767 PNGs

old brightness folder:
  sev_2: 5,767 PNGs
  sev_3: 5,767 PNGs
  sev_5: 5,767 PNGs
```

The old `brightness` folder is the positive-delta corruption and is therefore equivalent to
`brightness_increase`. Direct folder rename was denied by Windows, so a directory junction was
created:

```text
corruption_study/data/corrupted_images/brightness_increase
  -> corruption_study/data/corrupted_images/brightness
```

Verified `brightness_increase` resolves to 5,767 PNGs for severities 2, 3, and 5.

Completed remaining MVP image corruption:

```text
brightness_decrease:
  sev_2: 5,767 PNGs
  sev_3: 5,767 PNGs
  sev_5: 5,767 PNGs
```

MVP image corruption generation is complete for:

```text
gaussian_noise
brightness_increase
brightness_decrease
```

Run MVP IoT corruption generation next:

```powershell
& $PY .\corruption_study\gen\corrupt_iot.py --train-features .\e1_iot_only\features\train_iot_features.csv --test-features .\e1_iot_only\features\test_iot_features.csv --train-manifest .\Data\train.csv --test-manifest .\Data\test.csv --output-root .\corruption_study\data\corrupted_iot --corruptions sensor_noise dropout --severities 2 3 5 --seeds 42 123 999
```

## MVP IoT And Embedding Checkpoint

User ran IoT generation and embedding-regeneration commands. Verified:

```text
IoT corrupted CSVs:
  sensor_noise severity 2, 3, 5 for seeds 42, 123, 999: all 5,767 rows
  dropout severity 2, 3, 5 for seeds 42, 123, 999: all 5,767 rows

corrupted_iot_columns.json:
  54 continuous IoT feature columns corrupted
  metadata columns left untouched: image, label, crop, disease, stage, n_rows

embeddings:
  clean_regen/test_embeddings.npz exists
  image_corrupted_only/gaussian_noise embeddings complete
  iot_corrupted_only/sensor_noise embeddings complete
  iot_corrupted_only/dropout embeddings complete
```

Missing:

```text
image_corrupted_only/brightness_increase embeddings
image_corrupted_only/brightness_decrease embeddings
```

Follow-up completed by user and verified:

```text
image_corrupted_only/brightness_increase:
  sev_2: test_embeddings.npz exists
  sev_3: test_embeddings.npz exists
  sev_5: test_embeddings.npz exists

image_corrupted_only/brightness_decrease:
  sev_2: test_embeddings.npz exists
  sev_3: test_embeddings.npz exists
  sev_5: test_embeddings.npz exists

total regenerated embedding files:
  34
  = 1 clean_regen + 33 MVP corrupted embedding sets
```

Image materialization count verification:

```text
all MVP image corruption folders contain 5,767 PNGs:
  gaussian_noise severity 2, 3, 5 for seeds 42, 123, 999
  brightness_increase severity 2, 3, 5
  brightness_decrease severity 2, 3, 5
```

Remaining Phase 2 check:

```text
Compare regenerated clean TEST embeddings against existing clean E2/E2-8 metrics.
```

## MVP Evaluation Matrix Checkpoint

User ran the clean-regenerated sanity check plus MVP image/IoT corruption evaluations.
Verified output coverage:

```text
metrics JSON files:    106
prediction CSV files:  106
confusion CSV files:   106
```

Expected count:

```text
4 severity-0 clean model metrics
1 E0 passthrough metric
2 clean_regen E2/E2-8 metrics
45 image-corruption metrics
  gaussian_noise: 3 models x 3 severities x 3 seeds = 27
  brightness_increase/decrease: 3 models x 3 severities x 2 corruptions = 18
54 IoT-corruption metrics
  sensor_noise/dropout: 3 models x 3 severities x 3 seeds x 2 corruptions = 54
total = 106
```

Clean regenerated embedding sanity check:

```text
E2-canonical clean:
  accuracy  0.9873417721518988
  Macro F1 0.9679255699042882

E2-canonical clean_regen:
  accuracy  0.9875151725333796
  Macro F1 0.96851101868545

E2-8 clean:
  accuracy  0.9875151725333796
  Macro F1 0.9733555279688368

E2-8 clean_regen:
  accuracy  0.9876885729148605
  Macro F1 0.9740187344481859

status:
  PASS within tolerance
```

Added summary helper:

```text
corruption_study/eval/summarize_results.py
```

Next command:

```powershell
& $PY .\corruption_study\eval\summarize_results.py --results-dir .\corruption_study\results
```

## Summary Files Checkpoint

User ran `summarize_results.py`. Verified outputs:

```text
corruption_study/results/all_metrics.csv
  rows: 106

corruption_study/results/robustness_summary.csv
  rows: 45

corruption_study/results/robustness_summary.md
```

Initial MVP pattern:

```text
Under image corruption:
  E2-8 has smaller Macro-F1 drops than E0 and E2-canonical for gaussian_noise,
  brightness_increase, and brightness_decrease in the generated MVP conditions.

Under IoT corruption:
  E2-8 has much smaller Macro-F1 drops than E1 and E2-canonical for sensor_noise
  and dropout in the generated MVP conditions.

E2-canonical:
  generally improves over E1 under IoT corruption, but is much weaker than E2-8
  in this MVP summary.
```

Interpretation remains provisional until final tables/plots and any desired bootstrap CIs are
generated.

## Optional E2 128-to-25 Variant Support

Added evaluator support for the exploratory `e2_canonical_128_25` model:

```text
model key:
  e2_128_25

checkpoint:
  e2_canonical_128_25/checkpoints/best_e2_canonical_128_25.pt

architecture:
  Linear(1344,128)-BatchNorm-ReLU-Dropout(0.1)-Linear(128,25)

clean test metrics from existing result JSON:
  accuracy 0.9880353736778221
  Macro F1 0.9733663251945521
```

This is an exploratory practical classical variant, not a canonical promotion and not a
parameter-matched E3 comparator. It can reuse every already generated `corruption_study`
embedding file.

Updated:

```text
corruption_study/eval/run_corruption_eval.py
corruption_study/RUN_COMMANDS.md
```

The optional variant has also been evaluated over the already generated MVP clean,
clean-regenerated, image-corrupted, and IoT-corrupted conditions. This increased the result
coverage from 106 to 141 metrics/prediction/confusion triples.

## Final MVP Report Artifact Checkpoint

Generated report artifacts from saved metrics/prediction files only:

```text
corruption_study/eval/generate_report_artifacts.py
corruption_study/results/per_class_recall_by_condition.csv
corruption_study/results/severity0_sanity_check.md
corruption_study/results/passthrough_sanity_check.md
```

Verification:

```text
per_class_recall_by_condition.csv rows: 3525
severity0_sanity_check.md: PASS for E0, E1, E2-canonical, E2-8
passthrough_sanity_check.md: PASS for E0 passthrough
```

Commands run:

```powershell
& "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe" .\corruption_study\eval\summarize_results.py --results-dir .\corruption_study\results
& "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe" .\corruption_study\eval\generate_report_artifacts.py --results-dir .\corruption_study\results
```

## Full Single-Modality Expansion Checkpoint

User generated/evaluated the remaining planned single-modality corruptions:

```text
image_corrupted_only:
  motion_blur severity 2, 3, 5
  defocus_blur severity 2, 3, 5
  contrast severity 2, 3, 5
  jpeg_compression severity 2, 3, 5

iot_corrupted_only:
  drift severity 2, 3, 5
```

Verified:

```text
all expanded image folders contain 5,767 PNGs
drift corrupted IoT CSVs exist for severities 2, 3, 5
expanded E2/E2-8 corrupted embeddings exist for every new condition
metrics JSON files:    201
prediction CSV files:  201
confusion CSV files:   201
all_metrics.csv rows:  201
robustness_summary.csv rows: 120
per_class_recall_by_condition.csv rows: 5025
```

Coverage now includes all planned single-modality corruption types at severities 2, 3, and 5.
Paired bootstrap CIs and optional both-corrupted pairings remain unfinished.

## Both-Corrupted Pairing Revision

The representative brightness+drift paired stress condition was revised from:

```text
brightness_increase + drift
```

to:

```text
brightness_decrease + drift
```

Rationale:

```text
brightness_decrease better represents underexposure, shade, and low-light crop-image
capture failures, which are plausible field/greenhouse acquisition stressors.
```

Current intended representative `both_corrupted` pairings:

```text
gaussian_noise + sensor_noise
brightness_decrease + drift
motion_blur + dropout
```

## Final Both-Corrupted Checkpoint

The final paired-stress design was revised to use:

```text
canonical paired stress:
  gaussian_noise + sensor_noise

data-driven worst-case distinct paired stress:
  defocus_blur + drift
```

Worst-case distinct selection from the completed single-modality matrix:

```text
E0 image-corruption mean Macro-F1 drop ranking selected:
  defocus_blur = 0.903679

E1 IoT-corruption mean Macro-F1 drop ranking selected:
  drift = 0.821596

The selected pair is distinct from the canonical gaussian_noise + sensor_noise pair,
so no fallback to the second-largest IoT corruption was needed.
```

Generated both-corrupted embeddings through frozen E0/E1 encoders, not by corrupting cached
embeddings:

```text
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_2\cseed_42
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_2\cseed_123
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_2\cseed_999
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_3\cseed_42
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_3\cseed_123
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_3\cseed_999
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_5\cseed_42
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_5\cseed_123
corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_5\cseed_999
corruption_study\data\corrupted_embeddings\both_corrupted\img-defocus_blur__iot-drift\sev_2
corruption_study\data\corrupted_embeddings\both_corrupted\img-defocus_blur__iot-drift\sev_3
corruption_study\data\corrupted_embeddings\both_corrupted\img-defocus_blur__iot-drift\sev_5
```

Evaluated models:

```text
core:
  e2_canonical
  e2_8

exploratory:
  e2_128_25
```

E0 and E1 were not rerun for `both_corrupted`; their relevant single-modality baselines are reused
because E0 has no IoT input and E1 has no image input.

## Bootstrap CI Checkpoint

Added offline bootstrap/enrichment script:

```text
corruption_study\eval\enrich_metrics_with_bootstrap.py
```

Settings:

```text
bootstrap resamples: 1000
confidence interval: 95%
bootstrap seed: 20260609
macro-F1 handling: present-class macro-F1; labels absent from a resample are skipped
drop CI method: paired bootstrap aligned by sample_id
```

Clean-reference rule applied consistently:

```text
E0/E1:
  severity-0 clean

E2-canonical/E2-8/e2_128_25 regenerated-embedding conditions:
  severity-0 clean_regen
```

Fields added to metrics JSONs and summaries:

```text
accuracy_ci_low
accuracy_ci_high
macro_f1_ci_low
macro_f1_ci_high
accuracy_drop_ci_low
accuracy_drop_ci_high
macro_f1_drop_ci_low
macro_f1_drop_ci_high
macro_precision
weighted_f1
training_seed
checkpoint_path
config_path
timestamp
```

Final output coverage after both-corrupted and bootstrap/report regeneration:

```text
metrics JSON files:                  237
prediction CSV files:                237
confusion CSV files:                 237
all_metrics.csv rows:                237
robustness_summary.csv rows:         138
per_class_recall_by_condition rows: 5925
degradation_curve_data.csv rows:     233
drop_summary_with_ci.csv rows:       138
```

Generated final report artifacts:

```text
corruption_study\results\bootstrap_settings.json
corruption_study\results\degradation_curve_data.csv
corruption_study\results\drop_summary_with_ci.csv
corruption_study\results\THESIS_STYLE_ROBUSTNESS_REPORT.md
```

## Optional Outputs Checkpoint

Completed optional non-training outputs:

```text
corruption_study\eval\generate_plots.py
corruption_study\eval\bootstrap_difference_of_drops.py
corruption_study\results\plots\plot_manifest.csv
corruption_study\results\difference_of_drops_bootstrap.csv
corruption_study\results\difference_of_drops_summary.csv
corruption_study\scaffold_train_corrupted\README.md
```

Plot outputs:

```text
24 PNG plots under corruption_study\results\plots
  image corruption Macro-F1 and Macro-F1-drop curves
  IoT corruption Macro-F1 and Macro-F1-drop curves
  representative both_corrupted Macro-F1 and Macro-F1-drop curves
```

Difference-of-drops bootstrap:

```text
rows: 144 detailed comparisons
summary rows: 30
bootstrap resamples: 1000
seed: 20260610
interpretation: positive difference-of-drops means the multimodal model degraded less
```

The thesis-style report was regenerated to include additional robustness metrics:

```text
Macro-F1
Macro-F1 drop
Macro-F1 drop CI
accuracy
accuracy drop
accuracy drop CI
weighted F1
macro precision
difference-of-drops summary
```

The secondary train-on-corrupted arm remains documentation-only. No corrupted-training data was
generated and no training was run.

## Supervisor Secondary Augmented-Training Plan

Supervisor requested on 2026-06-10:

```text
train the model with augmented images and clean images
test once with only clean data
test secondly with both clean and augmented data
compare the performance
```

Updated:

```text
docs\by_stage\09_corruption_robustness\CORRUPTION_ROBUSTNESS_HARNESS_PLAN.md
corruption_study\scaffold_train_corrupted\README.md
```

Plan summary:

```text
Implement as a separate secondary arm under augmentation_training_study/.
Start with E0 image-only because the request specifically concerns augmented images.
Train E0 on clean + augmented train images.
Evaluate on clean test.
Evaluate on corrupted/augmented test and combined clean+augmented/corrupted test.
Compare against the existing clean-trained E0 baseline.
Keep E2/E2-8 as optional extension after E0.
```

No training or data generation was run for this plan update.

## E3 Corruption Evaluation Extension Checkpoint

Supervisor-requested extension on 2026-06-26:

```text
Run E3 on the corrupted data and review whether there is any improvement.
```

Environment repair completed before E3 evaluation:

```text
.venv-e1 remains the main project environment:
  G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe

The venv base was moved inside .venv-e1:
  G:\FOR THESIS\MAIN\.venv-e1\.base-python

Updated .venv-e1\pyvenv.cfg:
  home = G:\FOR THESIS\MAIN\.venv-e1\.base-python
  include-system-site-packages = true
  version = 3.10.20
```

Verified stack after repair:

```text
Python 3.10.20
torch 2.11.0+cu128
torchvision 0.26.0+cu128
numpy 1.26.4
pandas 2.3.3
scikit-learn 1.7.2
scipy 1.15.3
pennylane 0.42.3
pennylane-lightning 0.42.0
CUDA visible to torch: true
lightning.qubit device: available
```

Evaluator support added:

```text
corruption_study\eval\run_corruption_eval.py
```

New model keys:

```text
e3_8_vqc:
  checkpoint: e3_bottleneck_sweep\checkpoints\best_e3_8_vqc.pt
  architecture: original 8-qubit VQC head from e3_bottleneck_sweep/scripts/train_bottleneck.py

e3_8_attack_best:
  checkpoint: e3_beat_e2_attack\checkpoints\best_e3_8_warm_residual_bounded_pair_diag_s42_val_macro_f1.pt
  architecture: bounded residual warm-start E3 attack model from e3_beat_e2_attack/scripts/train_warmstart_e3.py
```

Sanity checks completed:

```text
e3_8_vqc clean Macro F1:          0.943716
e3_8_attack_best clean Macro F1:  0.968655
e3_8_vqc clean_regen Macro F1:    0.943423
e3_8_attack_best clean_regen Macro F1: 0.969652
```

Partial E3 corrupted evaluation completed before the user requested all processes stop:

```text
Completed condition:
  image_corrupted_only / gaussian_noise
  severities: 2, 3, 5
  corruption seeds: 42, 123, 999
  models: e3_8_vqc, e3_8_attack_best

E3 metrics files currently present:
  22
  = 4 sanity metrics + 18 gaussian_noise corrupted metrics
```

The running Python processes were stopped by request. No training was run.

Run commands for the remaining E3 corruption matrix are now documented in:

```text
corruption_study\RUN_COMMANDS.md
```

Remaining before final E3 interpretation:

```text
Run remaining E3 image corruptions:
  brightness_increase, brightness_decrease, motion_blur, defocus_blur, contrast, jpeg_compression

Run E3 IoT corruptions:
  sensor_noise, dropout, drift

Run E3 representative both-corrupted conditions:
  gaussian_noise + sensor_noise
  defocus_blur + drift

Refresh summary/report artifacts after all E3 metrics are complete.
```

## E3 Partial Output Cleanup And Parallel Runner Update

User instruction on 2026-06-26:

```text
Clean all partial files generated by the interrupted E3 runs.
Keep documentation and run commands updated constantly.
Optimize E3 execution to use more CPU.
```

Cleanup completed:

```text
Removed all generated E3 partial outputs under corruption_study\results:
  e3_8_vqc* metrics/predictions/confusions
  e3_8_attack_best* metrics/predictions/confusions
  e3_corruption_eval_run.log
  e3_parallel_logs if present

Verified remaining E3 generated result files: 0
```

Added parallel execution helper:

```text
corruption_study\run_e3_corruption_parallel.ps1
```

Purpose:

```text
Run independent E3 corruption conditions concurrently as separate Python processes.
This better utilizes CPU hardware than a single serial VQC inference process.
```

Recommended command:

```powershell
cd "G:\FOR THESIS\MAIN"
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 4
```

Throttle guidance:

```text
Start with 4 concurrent jobs.
Increase to 6 only if CPU is still underused and memory remains stable.
Decrease to 2 if the machine becomes unresponsive.
```

Updated documentation:

```text
corruption_study\RUN_COMMANDS.md
corruption_study\README.md
corruption_study\CHECKLIST.md
corruption_study\IMPLEMENTATION_REPORT.md
docs\by_stage\00_context_audit\FOR_THESIS_CONTEXT.md
```

Current E3 status after cleanup:

```text
Code and commands are ready.
No partial E3 generated outputs remain.
The full E3 corruption matrix was rerun from the beginning; see the next checkpoint.
```

## Completed E3 Corruption Matrix Checkpoint

Completed on 2026-06-26 with the parallel E3 runner:

```powershell
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 4
```

Execution details:

```text
E3 tasks launched: 62
E3 model outputs per task: 2
E3 metrics generated: 124
E3 predictions generated: 124
E3 confusion matrices generated: 124
No E3 training or fine-tuning was run.
```

Refreshed report artifacts:

```text
all_metrics.csv rows: 361
robustness_summary.csv rows: 210
per_class_recall_by_condition.csv rows: 9025
drop_summary_with_ci.csv rows: 210
difference_of_drops_summary.csv rows: 30
plot PNG count: 24
```

Added result summary:

```text
corruption_study\results\E3_CORRUPTION_EXTENSION_SUMMARY.md
docs\by_stage\09_corruption_robustness\E3_CORRUPTION_EXTENSION_SUMMARY.md
```

Updated stage report copy:

```text
docs\by_stage\09_corruption_robustness\THESIS_STYLE_ROBUSTNESS_REPORT.md
```

Clean sanity metrics:

```text
e2_8 clean_regen Macro F1:             0.974019
e3_8_vqc clean_regen Macro F1:         0.943423
e3_8_attack_best clean_regen Macro F1: 0.969652
```

Main interpretation:

```text
E3 does not show a broad Macro-F1 robustness improvement over e2_8.
E3 variants generally have larger Macro-F1 drops than e2_8 under image_corrupted_only and iot_corrupted_only conditions.
Original e3_8_vqc has one narrow averaged drop advantage over e2_8 under the representative both_corrupted defocus_blur+drift condition, but this is not enough to claim overall E3 robustness superiority.
e3_8_attack_best improves clean Macro F1 over original E3 and is less degraded than original E3 under IoT corruptions, but it is generally less robust than e2_8.
```

## Runtime Log Cleanup After E3 Completion

After the completed E3 run and report refresh, runtime-only parallel execution logs were removed to avoid extra repository clutter:

```text
Removed:
  corruption_study\results\e3_parallel_stdout.log
  corruption_study\results\e3_parallel_stderr.log
  corruption_study\results\e3_parallel_logs\*

Retained durable artifacts:
  E3 metrics JSON files
  E3 prediction CSV files
  E3 confusion CSV files
  all summary CSV files
  thesis/report Markdown files
  plots
```

Verified remaining runtime-only E3 logs: 0
