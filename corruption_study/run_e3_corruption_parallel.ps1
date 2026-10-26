param(
    [int]$Throttle = 4,
    [switch]$CleanE3Outputs,
    [switch]$SkipReports
)

$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"
$Root = Split-Path -Parent $PSScriptRoot
$PY = Join-Path $Root ".venv-e1\Scripts\python.exe"
$Eval = Join-Path $Root "corruption_study\eval\run_corruption_eval.py"
$Results = Join-Path $Root "corruption_study\results"
$Predictions = Join-Path $Results "predictions"
$Confusion = Join-Path $Results "confusion"
$LogDir = Join-Path $Results "e3_parallel_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (!(Test-Path $PY)) { throw "Missing .venv-e1 Python: $PY" }

if ($CleanE3Outputs) {
    $patterns = @(
        "$Results\e3_8_vqc*__metrics.json",
        "$Results\e3_8_attack_best*__metrics.json",
        "$Predictions\e3_8_vqc*__predictions.csv",
        "$Predictions\e3_8_attack_best*__predictions.csv",
        "$Confusion\e3_8_vqc*__confusion.csv",
        "$Confusion\e3_8_attack_best*__confusion.csv",
        "$Results\e3_corruption_eval_run.log",
        "$LogDir\*.log"
    )
    foreach ($pattern in $patterns) {
        Get-ChildItem -Path $pattern -File -ErrorAction SilentlyContinue | Remove-Item -Force
    }
}

$models = @("e3_8_vqc", "e3_8_attack_best")
$tasks = New-Object System.Collections.Generic.List[object]
function Add-Task([string]$Name, [string[]]$ArgsList) {
    $tasks.Add([pscustomobject]@{ Name = $Name; Args = $ArgsList }) | Out-Null
}

Add-Task "clean" (@("--models") + $models + @("--condition", "clean", "--severity", "0"))
Add-Task "clean_regen" (@("--models") + $models + @("--condition", "clean_regen", "--severity", "0", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\clean_regen"))

foreach ($sev in 2,3,5) {
    foreach ($seed in 42,123,999) {
        Add-Task "image_gaussian_noise_sev${sev}_seed${seed}" (@("--models") + $models + @("--condition", "image_corrupted_only", "--image-corruption", "gaussian_noise", "--severity", "$sev", "--corruption-seed", "$seed", "--image-root", ".\corruption_study\data\corrupted_images\gaussian_noise\sev_$sev\cseed_$seed", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\gaussian_noise\sev_$sev\cseed_$seed"))
    }
}

foreach ($corr in "brightness_increase", "brightness_decrease", "motion_blur", "defocus_blur", "contrast", "jpeg_compression") {
    foreach ($sev in 2,3,5) {
        Add-Task "image_${corr}_sev${sev}" (@("--models") + $models + @("--condition", "image_corrupted_only", "--image-corruption", $corr, "--severity", "$sev", "--image-root", ".\corruption_study\data\corrupted_images\$corr\sev_$sev", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\image_corrupted_only\$corr\sev_$sev"))
    }
}

foreach ($corr in "sensor_noise", "dropout") {
    foreach ($sev in 2,3,5) {
        foreach ($seed in 42,123,999) {
            Add-Task "iot_${corr}_sev${sev}_seed${seed}" (@("--models") + $models + @("--condition", "iot_corrupted_only", "--iot-corruption", $corr, "--severity", "$sev", "--corruption-seed", "$seed", "--iot-features", ".\corruption_study\data\corrupted_iot\$corr\sev_$sev\cseed_$seed\test_iot_features.csv", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\iot_corrupted_only\$corr\sev_$sev\cseed_$seed"))
        }
    }
}

foreach ($sev in 2,3,5) {
    Add-Task "iot_drift_sev${sev}" (@("--models") + $models + @("--condition", "iot_corrupted_only", "--iot-corruption", "drift", "--severity", "$sev", "--iot-features", ".\corruption_study\data\corrupted_iot\drift\sev_$sev\test_iot_features.csv", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\iot_corrupted_only\drift\sev_$sev"))
}

foreach ($sev in 2,3,5) {
    foreach ($seed in 42,123,999) {
        Add-Task "both_gaussian_sensor_sev${sev}_seed${seed}" (@("--models") + $models + @("--condition", "both_corrupted", "--image-corruption", "gaussian_noise", "--iot-corruption", "sensor_noise", "--severity", "$sev", "--corruption-seed", "$seed", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\both_corrupted\img-gaussian_noise__iot-sensor_noise\sev_$sev\cseed_$seed"))
    }
}

foreach ($sev in 2,3,5) {
    Add-Task "both_defocus_drift_sev${sev}" (@("--models") + $models + @("--condition", "both_corrupted", "--image-corruption", "defocus_blur", "--iot-corruption", "drift", "--severity", "$sev", "--embedding-dir", ".\corruption_study\data\corrupted_embeddings\both_corrupted\img-defocus_blur__iot-drift\sev_$sev"))
}

Write-Host "Launching $($tasks.Count) E3 tasks with throttle=$Throttle"
$running = @()
$failures = New-Object System.Collections.Generic.List[string]

foreach ($task in $tasks) {
    while ($running.Count -ge $Throttle) {
        $done = Wait-Job -Job $running -Any
        $output = Receive-Job -Job $done
        $logPath = Join-Path $LogDir "$($done.Name).log"
        $output | Set-Content -LiteralPath $logPath
        if ($done.State -ne "Completed") { $failures.Add($done.Name) | Out-Null }
        Remove-Job -Job $done
        $running = @($running | Where-Object { $_.Id -ne $done.Id })
    }

    $jobName = $task.Name
    $jobArgs = $task.Args
    $running += Start-Job -Name $jobName -ArgumentList $PY, $Eval, $jobArgs, $Root -ScriptBlock {
        param($PY, $Eval, $ArgsList, $Root)
        $env:PYTHONDONTWRITEBYTECODE = "1"
        Set-Location $Root
        & $PY $Eval @ArgsList --cpu --batch-size 512
        if ($LASTEXITCODE -ne 0) { throw "run_corruption_eval failed with exit code $LASTEXITCODE" }
    }
}

while ($running.Count -gt 0) {
    $done = Wait-Job -Job $running -Any
    $output = Receive-Job -Job $done
    $logPath = Join-Path $LogDir "$($done.Name).log"
    $output | Set-Content -LiteralPath $logPath
    if ($done.State -ne "Completed") { $failures.Add($done.Name) | Out-Null }
    Remove-Job -Job $done
    $running = @($running | Where-Object { $_.Id -ne $done.Id })
}

if ($failures.Count -gt 0) {
    throw "Failed E3 tasks: $($failures -join ', ')"
}

Write-Host "All E3 inference tasks completed."

if (!$SkipReports) {
    & $PY .\corruption_study\eval\enrich_metrics_with_bootstrap.py --results-dir .\corruption_study\results --n-bootstrap 1000 --seed 20260609
    if ($LASTEXITCODE -ne 0) { throw "bootstrap enrichment failed" }
    & $PY .\corruption_study\eval\summarize_results.py --results-dir .\corruption_study\results
    if ($LASTEXITCODE -ne 0) { throw "summarize failed" }
    & $PY .\corruption_study\eval\generate_report_artifacts.py --results-dir .\corruption_study\results
    if ($LASTEXITCODE -ne 0) { throw "report artifacts failed" }
    & $PY .\corruption_study\eval\generate_thesis_report.py --results-dir .\corruption_study\results --output .\corruption_study\results\THESIS_STYLE_ROBUSTNESS_REPORT.md
    if ($LASTEXITCODE -ne 0) { throw "thesis report failed" }
    & $PY .\corruption_study\eval\bootstrap_difference_of_drops.py --results-dir .\corruption_study\results --n-bootstrap 1000 --seed 20260610
    if ($LASTEXITCODE -ne 0) { throw "difference bootstrap failed" }
    & $PY .\corruption_study\eval\generate_plots.py --results-dir .\corruption_study\results
    if ($LASTEXITCODE -ne 0) { throw "plot generation failed" }
}

Write-Host "E3 parallel corruption run complete."