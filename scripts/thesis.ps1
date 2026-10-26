param(
    [ValidateSet("env", "e0", "e1", "e0e1", "e2-extract", "e2-train", "e2")]
    [string]$Action = "env",
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonPath = "",
    [string]$CacheRoot = "",
    [string]$RunName = "",
    [string]$E0RunName = "",
    [string]$E1RunName = "",
    [string]$E2RunName = "",
    [int]$BatchSize = 512,
    [int]$Epochs = 60,
    [int]$Patience = 8,
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    $PythonPath = Join-Path $ProjectRoot ".venv-e1\Scripts\python.exe"
}

if ([string]::IsNullOrWhiteSpace($CacheRoot)) {
    $CacheRoot = Join-Path $ProjectRoot ".cache"
}

function Initialize-ThesisEnvironment {
    Set-Location -LiteralPath $ProjectRoot

    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONIOENCODING = "utf-8"

    $venvRoot = Split-Path -Parent (Split-Path -Parent $PythonPath)
    $venvScripts = Split-Path -Parent $PythonPath
    if (Test-Path -LiteralPath $venvScripts) {
        $env:VIRTUAL_ENV = $venvRoot
        $pathParts = $env:PATH -split ';' | Where-Object { $_ -and ($_ -ne $venvScripts) }
        $env:PATH = @($venvScripts) + $pathParts -join ';'
    }

    $torchCache = Join-Path $CacheRoot "torch"
    $matplotlibCache = Join-Path $CacheRoot "matplotlib"
    New-Item -ItemType Directory -Force -Path $torchCache | Out-Null
    New-Item -ItemType Directory -Force -Path $matplotlibCache | Out-Null

    $env:TORCH_HOME = $torchCache
    $env:XDG_CACHE_HOME = $CacheRoot
    $env:MPLCONFIGDIR = $matplotlibCache
}

function Invoke-Python {
    param(
        [Parameter(Mandatory=$true)][string[]]$CommandArgs
    )

    & $PythonPath @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-LoggedPython {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$StdoutPath,
        [Parameter(Mandatory=$true)][string]$StderrPath,
        [Parameter(Mandatory=$true)][string[]]$CommandArgs
    )

    Remove-Item -LiteralPath $StdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $StderrPath -Force -ErrorAction SilentlyContinue

    Write-Host "[$Name] starting"
    Write-Host "[$Name] stdout: $StdoutPath"
    Write-Host "[$Name] stderr: $StderrPath"

    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $PythonPath @CommandArgs > $StdoutPath 2> $StderrPath
    $ErrorActionPreference = $previousErrorActionPreference
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        Write-Host "[$Name] failed with exit code $exitCode"
        Write-Host "[$Name] last stderr lines:"
        Get-Content -LiteralPath $StderrPath -Tail 40 -ErrorAction SilentlyContinue
        exit $exitCode
    }

    Write-Host "[$Name] finished"
}

function Get-LogPath {
    param(
        [Parameter(Mandatory=$true)][string]$SubDir,
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][string]$Stream
    )

    $root = Join-Path $ProjectRoot "audit_outputs"
    $dir = Join-Path $root $SubDir
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    return Join-Path $dir "$($Name)_$($Stream).log"
}

Initialize-ThesisEnvironment

if (-not (Test-Path -LiteralPath $PythonPath)) {
    throw "Python executable not found: $PythonPath"
}

# Training actions write regular experiment outputs under the historical run
# names below. Downstream E2/E3 consumers read from canonical/ snapshots; promote
# a retrain into canonical/ explicitly before using it as a downstream input.

if ($Action -eq "env") {
    Write-Host "Project root: $ProjectRoot"
    Write-Host "Python: $PythonPath"
    Write-Host "VIRTUAL_ENV: $env:VIRTUAL_ENV"
    Write-Host "TORCH_HOME: $env:TORCH_HOME"
    Write-Host "XDG_CACHE_HOME: $env:XDG_CACHE_HOME"
    Write-Host "MPLCONFIGDIR: $env:MPLCONFIGDIR"
    Invoke-Python -CommandArgs @("-c", "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')")
    exit 0
}

if ($Action -eq "e0" -or $Action -eq "e0e1") {
    if ([string]::IsNullOrWhiteSpace($E0RunName)) {
        $E0RunName = if ([string]::IsNullOrWhiteSpace($RunName)) { "e0_manual_bs768_w0_20260602" } else { $RunName }
    }
    Invoke-LoggedPython `
        -Name "E0" `
        -StdoutPath (Get-LogPath -SubDir "e0_logs" -Name $E0RunName -Stream "stdout") `
        -StderrPath (Get-LogPath -SubDir "e0_logs" -Name $E0RunName -Stream "stderr") `
        -CommandArgs @(
            "e0_image_only\learning\train.py",
            "--run-name", $E0RunName,
            "--batch-size", "768",
            "--num-workers", "0",
            "--lr", "0.0003",
            "--weight-decay", "0.0001",
            "--unfreeze-last-blocks", "2",
            "--epochs", "40",
            "--patience", "7",
            "--scheduler-patience", "3",
            "--scheduler-factor", "0.5",
            "--cudnn-benchmark",
            "--allow-tf32"
        )
}

if ($Action -eq "e1" -or $Action -eq "e0e1") {
    if ([string]::IsNullOrWhiteSpace($E1RunName)) {
        $E1RunName = if ([string]::IsNullOrWhiteSpace($RunName)) { "e1_h128_64_nodrop_wd_bs512_sched" } else { $RunName }
    }
    Invoke-LoggedPython `
        -Name "E1" `
        -StdoutPath (Get-LogPath -SubDir "e1_logs" -Name $E1RunName -Stream "stdout") `
        -StderrPath (Get-LogPath -SubDir "e1_logs" -Name $E1RunName -Stream "stderr") `
        -CommandArgs @(
            "e1_iot_only\scripts\train_e1_iot_mlp.py",
            "--run-name", $E1RunName,
            "--batch-size", "512",
            "--hidden1", "128",
            "--hidden2", "64",
            "--dropout1", "0.0",
            "--dropout2", "0.0",
            "--weight-decay", "0.0001",
            "--lr", "0.001",
            "--epochs", "40",
            "--patience", "7",
            "--scheduler",
            "--scheduler-patience", "3",
            "--scheduler-factor", "0.5"
        )
}

if ($Action -eq "e2-extract" -or $Action -eq "e2") {
    $args = @(
        "e2_multimodal_fusion\scripts\extract_e2_embeddings.py",
        "--batch-size", "$BatchSize",
        "--num-workers", "0"
    )
    if ($Cpu) { $args += "--cpu" }
    Invoke-Python -CommandArgs $args
}

if ($Action -eq "e2-train" -or $Action -eq "e2") {
    if ([string]::IsNullOrWhiteSpace($E2RunName)) {
        $E2RunName = if ([string]::IsNullOrWhiteSpace($RunName)) { "e2_frozen_e0_manual_e1_h128_64_fusion" } else { $RunName }
    }
    $args = @(
        "e2_multimodal_fusion\scripts\train_e2_fusion.py",
        "--run-name", $E2RunName,
        "--batch-size", "$BatchSize",
        "--epochs", "$Epochs",
        "--patience", "$Patience",
        "--lr", "0.001",
        "--weight-decay", "0.0001"
    )
    if ($Cpu) { $args += "--cpu" }
    Invoke-Python -CommandArgs $args
}

Write-Host "Action completed: $Action"
