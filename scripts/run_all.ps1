# Run the whole PathSnap pipeline, stage by stage.
#
#   .\scripts\run_all.ps1            full run
#   .\scripts\run_all.ps1 -Quick     smaller forests / fewer trials, for a sanity pass
#
# Each stage writes to results/ and can also be run on its own.

param(
    [switch]$Quick,
    [int]$Trials = 30
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

if (-not (Test-Path (Join-Path $root "data\raw\train"))) {
    Write-Error "data/raw/train not found - unpack the dataset first (see README section 3)."
}

function Invoke-Stage {
    param([string]$Name, [string[]]$Arguments)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    Write-Host "  $Name" -ForegroundColor Cyan
    Write-Host ("=" * 70) -ForegroundColor DarkGray
    $started = Get-Date
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) { Write-Error "$Name failed (exit $LASTEXITCODE)" }
    $mins = ((Get-Date) - $started).TotalMinutes
    Write-Host ("  done in {0:N1} min" -f $mins) -ForegroundColor Green
}

$quickFlag = if ($Quick) { @("--quick") } else { @() }
$trialCount = if ($Quick) { 8 } else { $Trials }

Invoke-Stage "1/8  Build candidate dataset"        @("-I", "scripts\01_build_dataset.py")
Invoke-Stage "2/8  Exploratory data analysis"      @("-I", "scripts\02_eda.py")
Invoke-Stage "3/8  Train and compare models"       (@("-I", "scripts\03_train_models.py") + $quickFlag)
Invoke-Stage "3b/8 Confusion matrices"             @("-I", "scripts\03b_confusion_matrices.py")
Invoke-Stage "4/8  Bayesian optimisation"          @("-I", "scripts\04_bayesian_optimisation.py", "--trials", $trialCount)
Invoke-Stage "5/8  HMM and Viterbi"                @("-I", "scripts\05_hmm_viterbi.py")
Invoke-Stage "6/8  Generate Folium maps"           @("-I", "scripts\06_generate_maps.py")
Invoke-Stage "7/8  Export inference artifacts"     @("-I", "scripts\07_export_artifacts.py")
Invoke-Stage "8/8  Build the report skeleton"      @("-I", "scripts\08_build_report.py")

Write-Host ""
Write-Host "Pipeline complete." -ForegroundColor Green
Write-Host "  results/model_comparison.csv       model comparison table"
Write-Host "  results/bayesopt_summary.csv       tuned vs untuned"
Write-Host "  results/hmm_comparison_per_trip.csv  classifier vs HMM"
Write-Host "  results/maps/index.html            interactive maps"
Write-Host "  results/report/technical_report.md report skeleton"
Write-Host ""
Write-Host "Serve it:  uvicorn api.main:app --host 0.0.0.0 --port 8000"
