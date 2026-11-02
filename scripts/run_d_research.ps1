param([string]$Python = 'python')
$ErrorActionPreference='Stop'
Set-Location (Join-Path $PSScriptRoot '..')
. ./scripts/project_cache.ps1
$env:PYTHONPATH="$PWD/.runtime;$PWD"
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
& $Python -u scripts/prepare_d_research_data.py
if ($LASTEXITCODE -ne 0) { throw 'Study data preparation failed' }
& $Python scripts/check_d_legacy_parity.py
if ($LASTEXITCODE -ne 0) { throw 'Legacy regression check failed' }
if (!(Test-Path outputs/extended_video_validation_gpu518/summary.json) -or
    (Get-Content outputs/extended_video_validation_gpu518/summary.json -Raw | ConvertFrom-Json).clips.Count -ne 6) {
    & $Python -u scripts/run_extended_video_validation.py --device cuda --input-size 518 --output outputs/extended_video_validation_gpu518
    if ($LASTEXITCODE -ne 0) { throw 'Reference video preparation failed' }
}
& $Python -u scripts/research_direction_d.py
if ($LASTEXITCODE -ne 0) { throw 'D study failed' }
& $Python -u scripts/benchmark_d_pipeline.py
if ($LASTEXITCODE -ne 0) { throw 'Pipeline timing failed' }
& $Python -u scripts/analyze_d_research.py
if ($LASTEXITCODE -ne 0) { throw 'Evidence validation failed' }
