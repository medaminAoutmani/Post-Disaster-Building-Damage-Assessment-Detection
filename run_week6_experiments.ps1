# Run all Week 6 experiments in order from the workspace root.
# Usage: .\run_week6_experiments.ps1

Set-StrictMode -Version Latest

# Ensure this script is run from the project root
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Scaffolding Week 6 results tree..."
python src/week6/week6_experiment_runner.py --scaffold-only

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

foreach ($experiment in $experiments) {
    Write-Host "\nRunning experiment: $experiment"
    python src/week6/week6_experiment_runner.py --experiment $experiment
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Experiment '$experiment' failed with exit code $LASTEXITCODE. Stopping script."
        exit $LASTEXITCODE
    }
}

Write-Host "\nSummarizing Week 6 results..."
python src/week6/week6_experiment_runner.py --summarize-only

Write-Host "\nAll Week 6 experiments completed successfully."
