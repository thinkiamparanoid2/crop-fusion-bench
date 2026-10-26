# Run Commands

Use the project venv explicitly:

```powershell
$PY = "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe"
cd "G:\FOR THESIS\MAIN"
```

These commands are for manual execution. They do not train models.

## Severity Scope

Primary study severities:

```text
clean severity: 0
corrupted severities: 2, 3, 5
```

Severity `1` can be added later if time allows. Severity `6` is not currently defined by
the corruption schedules and should only be added after explicit severity-6 parameters are
chosen for every corruption type.

## 1. MVP Image Corruption Generation

Passthrough image set:

```powershell
& $PY .\corruption_study\gen\corrupt_images.py --input-root .\Data\test --manifest .\Data\test.csv --output-root .\corruption_study\data\corrupted_images --passthrough
```

MVP image corruptions:

```powershell
& $PY .\corruption_study\gen\corrupt_images.py --input-root .\Data\test --manifest .\Data\test.csv --output-root .\corruption_study\data\corrupted_images --corruptions gaussian_noise brightness_increase brightness_decrease --severities 2 3 5 --seeds 42 123 999
```

## 2. MVP IoT Feature-Space Corruption Generation

```powershell
& $PY .\corruption_study\gen\corrupt_iot.py --train-features .\e1_iot_only\features\train_iot_features.csv --test-features .\e1_iot_only\features\test_iot_features.csv --train-manifest .\Data\train.csv --test-manifest .\Data\test.csv --output-root .\corruption_study\data\corrupted_iot --corruptions sensor_noise dropout --severities 2 3 5 --seeds 42 123 999
```

## 3. Regenerate Clean TEST Embeddings Sanity Check

This verifies the frozen-encoder regeneration path. It is not training.

```powershell
& $PY .\corruption_study\gen\regen_embeddings.py --condition clean_regen --image-root .\Data\test --iot-features .\e1_iot_only\features\test_iot_features.csv --output-dir .\corruption_study\data\corrupted_embeddings\clean_regen --batch-size 512
```

## 4. Regenerate MVP Corrupted E2/E2-8 TEST Embeddings

Image-corrupted only example:

```powershell
& $PY .\corruption_study\gen\regen_embeddings.py --condition image_corrupted_only --image-root .\corruption_study\data\corrupted_images\gaussian_noise\sev_2\cseed_42 --iot-features .\e1_iot_only\features\test_iot_features.csv --output-dir .\corruption_study\data\corrupted_embeddings\image_corrupted_only\gaussian_noise\sev_2\cseed_42 --batch-size 512
```

IoT-corrupted only example:

```powershell
& $PY .\corruption_study\gen\regen_embeddings.py --condition iot_corrupted_only --image-root .\Data\test --iot-features .\corruption_study\data\corrupted_iot\sensor_noise\sev_2\cseed_42\test_iot_features.csv --output-dir .\corruption_study\data\corrupted_embeddings\iot_corrupted_only\sensor_noise\sev_2\cseed_42 --batch-size 512
```

## 5. Evaluation Examples

Severity-0 clean checks:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e0 e1 e2_canonical e2_8 --condition clean --severity 0
```

Passthrough image check:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e0 --condition image_passthrough --image-root .\corruption_study\data\corrupted_images\passthrough --severity 0
```

One MVP image-corruption evaluation:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e0 e2_canonical e2_8 --condition image_corrupted_only --image-corruption gaussian_noise --severity 2 --corruption-seed 42 --image-root .\corruption_study\data\corrupted_images\gaussian_noise\sev_2\cseed_42 --embedding-dir .\corruption_study\data\corrupted_embeddings\image_corrupted_only\gaussian_noise\sev_2\cseed_42
```

One MVP IoT-corruption evaluation:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e1 e2_canonical e2_8 --condition iot_corrupted_only --iot-corruption sensor_noise --severity 2 --corruption-seed 42 --iot-features .\corruption_study\data\corrupted_iot\sensor_noise\sev_2\cseed_42\test_iot_features.csv --embedding-dir .\corruption_study\data\corrupted_embeddings\iot_corrupted_only\sensor_noise\sev_2\cseed_42
```

## 6. Full Matrix Later

Do not run the full matrix until the MVP slice passes severity-0, passthrough, and regenerated-clean-embedding checks.

## 7. Optional E2 128-to-25 Variant

This reuses the existing `e2_canonical_128_25` checkpoint and the already regenerated
corruption-study embeddings. It does not train.

Clean and clean-regenerated checks:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e2_128_25 --condition clean --severity 0
if ($LASTEXITCODE -ne 0) { throw "e2_128_25 clean check failed" }

& $PY .\corruption_study\eval\run_corruption_eval.py --models e2_128_25 --condition clean_regen --severity 0 --embedding-dir .\corruption_study\data\corrupted_embeddings\clean_regen
if ($LASTEXITCODE -ne 0) { throw "e2_128_25 clean_regen check failed" }
```

Image-corruption evaluations:

```powershell
foreach ($sev in 2,3,5) {
  foreach ($seed in 42,123,999) {
    & $PY .\corruption_study\eval\run_corruption_eval.py --models e2_128_25 --condition image_corrupted_only --image-corruption gaussian_noise --severity $sev --corruption-seed $seed --image-root ".\corruption_study\data\corrupted_images\gaussian_noise\sev_$sev\cseed_$seed" --embedding-dir ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\gaussian_noise\sev_$sev\cseed_$seed"
    if ($LASTEXITCODE -ne 0) { throw "e2_128_25 image eval failed: gaussian_noise sev_$sev seed_$seed" }
  }
}

foreach ($corr in "brightness_increase","brightness_decrease") {
  foreach ($sev in 2,3,5) {
    & $PY .\corruption_study\eval\run_corruption_eval.py --models e2_128_25 --condition image_corrupted_only --image-corruption $corr --severity $sev --image-root ".\corruption_study\data\corrupted_images\$corr\sev_$sev" --embedding-dir ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\$corr\sev_$sev"
    if ($LASTEXITCODE -ne 0) { throw "e2_128_25 image eval failed: $corr sev_$sev" }
  }
}
```

IoT-corruption evaluations:

```powershell
foreach ($corr in "sensor_noise","dropout") {
  foreach ($sev in 2,3,5) {
    foreach ($seed in 42,123,999) {
      & $PY .\corruption_study\eval\run_corruption_eval.py --models e2_128_25 --condition iot_corrupted_only --iot-corruption $corr --severity $sev --corruption-seed $seed --iot-features ".\corruption_study\data\corrupted_iot\$corr\sev_$sev\cseed_$seed\test_iot_features.csv" --embedding-dir ".\corruption_study\data\corrupted_embeddings\iot_corrupted_only\$corr\sev_$sev\cseed_$seed"
      if ($LASTEXITCODE -ne 0) { throw "e2_128_25 IoT eval failed: $corr sev_$sev seed_$seed" }
    }
  }
}
```

## 8. Supervisor-Requested E3 Corruption Evaluation

This is an inference-only extension requested by the supervisor. It evaluates E3 on the already materialized corrupted TEST inputs by reusing the existing corrupted embedding files. It does not train or fine-tune E3.

Models:

```text
e3_8_vqc          original E3-8 VQC checkpoint: e3_bottleneck_sweep/checkpoints/best_e3_8_vqc.pt
e3_8_attack_best  best attack checkpoint: e3_beat_e2_attack/checkpoints/best_e3_8_warm_residual_bounded_pair_diag_s42_val_macro_f1.pt
```

Use CPU for E3 VQC inference unless a working PennyLane GPU simulator is explicitly configured. Set `PYTHONDONTWRITEBYTECODE` to avoid stale `__pycache__` permission collisions.

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$PY = "G:\FOR THESIS\MAIN\.venv-e1\Scripts\python.exe"
cd "G:\FOR THESIS\MAIN"
```

Clean and regenerated-clean sanity checks:

```powershell
& $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition clean --severity 0 --cpu --batch-size 512
if ($LASTEXITCODE -ne 0) { throw "E3 clean sanity failed" }

& $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition clean_regen --severity 0 --embedding-dir .\corruption_study\data\corrupted_embeddings\clean_regen --cpu --batch-size 512
if ($LASTEXITCODE -ne 0) { throw "E3 clean_regen sanity failed" }
```

Image-corrupted-only evaluations:

```powershell
foreach ($sev in 2,3,5) {
  foreach ($seed in 42,123,999) {
    & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition image_corrupted_only --image-corruption gaussian_noise --severity $sev --corruption-seed $seed --image-root ".\corruption_study\data\corrupted_images\gaussian_noise\sev_$sev\cseed_$seed" --embedding-dir ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\gaussian_noise\sev_$sev\cseed_$seed" --cpu --batch-size 512
    if ($LASTEXITCODE -ne 0) { throw "E3 gaussian_noise failed sev=$sev seed=$seed" }
  }
}

foreach ($corr in "brightness_increase","brightness_decrease","motion_blur","defocus_blur","contrast","jpeg_compression") {
  foreach ($sev in 2,3,5) {
    & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition image_corrupted_only --image-corruption $corr --severity $sev --image-root ".\corruption_study\data\corrupted_images\$corr\sev_$sev" --embedding-dir ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\$corr\sev_$sev" --cpu --batch-size 512
    if ($LASTEXITCODE -ne 0) { throw "E3 image failed corr=$corr sev=$sev" }
  }
}
```

IoT-corrupted-only evaluations:

```powershell
foreach ($corr in "sensor_noise","dropout") {
  foreach ($sev in 2,3,5) {
    foreach ($seed in 42,123,999) {
      & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition iot_corrupted_only --iot-corruption $corr --severity $sev --corruption-seed $seed --iot-features ".\corruption_study\data\corrupted_iot\$corr\sev_$sev\cseed_$seed\test_iot_features.csv" --embedding-dir ".\corruption_study\data\corrupted_embeddings\iot_corrupted_only\$corr\sev_$sev\cseed_$seed" --cpu --batch-size 512
      if ($LASTEXITCODE -ne 0) { throw "E3 IoT failed corr=$corr sev=$sev seed=$seed" }
    }
  }
}

foreach ($sev in 2,3,5) {
  & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition iot_corrupted_only --iot-corruption drift --severity $sev --iot-features ".\corruption_study\data\corrupted_iot\drift\sev_$sev\test_iot_features.csv" --embedding-dir ".\corruption_study\data\corrupted_embeddings\iot_corrupted_only\drift\sev_$sev" --cpu --batch-size 512
  if ($LASTEXITCODE -ne 0) { throw "E3 drift failed sev=$sev" }
}
```

Representative both-corrupted evaluations:

```powershell
foreach ($sev in 2,3,5) {
  foreach ($seed in 42,123,999) {
    & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition both_corrupted --image-corruption gaussian_noise --iot-corruption sensor_noise --severity $sev --corruption-seed $seed --embedding-dir ".\corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_$sev\cseed_$seed" --cpu --batch-size 512
    if ($LASTEXITCODE -ne 0) { throw "E3 both gaussian+sensor failed sev=$sev seed=$seed" }
  }
}

foreach ($sev in 2,3,5) {
  & $PY .\corruption_study\eval\run_corruption_eval.py --models e3_8_vqc e3_8_attack_best --condition both_corrupted --image-corruption defocus_blur --iot-corruption drift --severity $sev --embedding-dir ".\corruption_study\data\corrupted_embeddings\both_corrupted\img-defocus_blur__iot-drift\sev_$sev" --cpu --batch-size 512
  if ($LASTEXITCODE -ne 0) { throw "E3 both defocus+drift failed sev=$sev" }
}
```

Refresh reports after E3 runs finish:

```powershell
& $PY .\corruption_study\eval\enrich_metrics_with_bootstrap.py --results-dir .\corruption_study\results --n-bootstrap 1000 --seed 20260609
& $PY .\corruption_study\eval\summarize_results.py --results-dir .\corruption_study\results
& $PY .\corruption_study\eval\generate_report_artifacts.py --results-dir .\corruption_study\results
& $PY .\corruption_study\eval\generate_thesis_report.py --results-dir .\corruption_study\results --output .\corruption_study\results\THESIS_STYLE_ROBUSTNESS_REPORT.md
& $PY .\corruption_study\eval\bootstrap_difference_of_drops.py --results-dir .\corruption_study\results --n-bootstrap 1000 --seed 20260610
& $PY .\corruption_study\eval\generate_plots.py --results-dir .\corruption_study\results
```

Completion note as of 2026-06-26: the full E3 corruption matrix was rerun from a clean E3 output state using the parallel runner in Section 9. Rerunning the serial commands in this section will overwrite the completed E3 metrics/prediction/confusion files with fresh timestamps.

## 9. Recommended Parallel E3 Runner For Better CPU Utilization

The serial E3 commands in Section 8 are deterministic but underuse CPU because a single PennyLane `lightning.qubit` VQC forward pass does not saturate all cores. To use more hardware, run independent corruption conditions in parallel as separate Python processes.

Helper script:

```text
corruption_study\run_e3_corruption_parallel.ps1
```

Start from a clean E3 output state and run the full E3 corruption matrix with 4 concurrent jobs:

```powershell
cd "G:\FOR THESIS\MAIN"
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 4
```

If the machine remains underutilized and memory is stable, increase throttle gradually:

```powershell
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 6
```

Use a lower throttle if the machine becomes unresponsive:

```powershell
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 2
```

Run inference only and skip report regeneration, useful for debugging:

```powershell
.\corruption_study\run_e3_corruption_parallel.ps1 -CleanE3Outputs -Throttle 4 -SkipReports
```

Per-condition logs are written under:

```text
corruption_study\results\e3_parallel_logs\
```

Cleanup-only command for partial E3 generated outputs:

```powershell
$results = Resolve-Path 'corruption_study\results'
Get-ChildItem -LiteralPath $results -Recurse -File | Where-Object {
  $_.Name -like 'e3_8_vqc*' -or
  $_.Name -like 'e3_8_attack_best*' -or
  $_.Name -eq 'e3_corruption_eval_run.log' -or
  $_.FullName -like '*e3_parallel_logs*'
} | Remove-Item -Force
```

Current cleanup status as of 2026-06-26:

```text
All partial E3 generated result files were removed before the completed rerun.
Completed E3 result artifacts are now present in corruption_study/results.
Runtime-only e3_parallel_logs were removed after completion.
```
