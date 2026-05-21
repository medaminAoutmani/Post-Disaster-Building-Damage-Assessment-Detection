# Run the focused Week 6 Morocco-adaptation experiments from the workspace root.
# Usage:
#   .\run_week6_experiments.ps1
#   .\run_week6_experiments.ps1 -FullSuite

param(
    [switch]$FullSuite,
    [switch]$HPO,
    [int]$Epochs = 12,
    [int]$BatchSize = 4,
    [int]$ImageSize = 512,
    [int]$Trials = 12
)

Set-StrictMode -Version Latest

# Ensure this script is run from the project root
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Scaffolding Week 6 results tree..."
python src/week6/week6_experiment_runner.py --scaffold-only

if ($HPO) {
    Write-Host "`nRunning cheap Optuna HPO for Morocco adaptation..."
    python src/week6/week6_experiment_runner.py `
        --experiment attention_unet `
        --morocco-adaptation `
        --optuna-study `
        --optuna-trials $Trials `
        --optuna-epochs 6 `
        --optuna-max-train-samples 2000 `
        --optuna-max-val-samples 500 `
        --image-size $ImageSize `
        --early-stopping-patience 3
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Optuna HPO failed with exit code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
    exit 0
}

if ($FullSuite) {
    $experiments = @(
        'baseline',
        'focal_loss',
        'tversky_loss',
        'focal_tversky',
        'weighted_sampler',
        'attention_unet',
        'unetplusplus',
        'deeplabv3',
        'resnet50'
    )
} else {
    # Morocco adaptation: keep the research signal, skip the slowest broad-suite runs.
    $experiments = @(
        'baseline',
        'focal_loss',
        'weighted_sampler',
        'attention_unet',
        'resnet50'
    )
}

foreach ($experiment in $experiments) {
    Write-Host "\nRunning experiment: $experiment"
    python src/week6/week6_experiment_runner.py `
        --experiment $experiment `
        --morocco-adaptation `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --image-size $ImageSize `
        --early-stopping-patience 5
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment '$experiment' failed with exit code $LASTEXITCODE. Stopping script."
        exit $LASTEXITCODE
    }
}

Write-Host "\nSummarizing Week 6 results..."
python src/week6/week6_experiment_runner.py --summarize-only

Write-Host "\nAll Week 6 experiments completed successfully."
