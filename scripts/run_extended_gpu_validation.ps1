param([string]$Python = 'python', [switch]$IncludeCpuReference)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
. ./scripts/project_cache.ps1
$env:PYTHONPATH = "$PWD/.runtime_depthpro;$PWD/.runtime;$PWD"
& $Python -u scripts/run_ibims_refinement_cpu.py --model depth-pro --device cuda --precision fp32 --output outputs/ibims_refinement_gpu
if ($LASTEXITCODE -ne 0) { throw 'Depth Pro audit failed' }
$env:PYTHONPATH = "$PWD/.runtime;$PWD"
& $Python -u scripts/run_ibims_refinement_cpu.py --model unidepth-v2 --device cpu
if ($LASTEXITCODE -ne 0) { throw 'UniDepth image audit failed' }
if ($IncludeCpuReference) {
    & $Python -u scripts/run_extended_video_validation.py
    if ($LASTEXITCODE -ne 0) { throw 'CPU video reference failed' }
}
& $Python -u scripts/run_extended_video_validation.py --device cuda --input-size 518 --output outputs/extended_video_validation_gpu518
if ($LASTEXITCODE -ne 0) { throw 'Video audit failed' }
if (Test-Path outputs/extended_video_validation_cpu/summary.json) {
    & $Python scripts/summarize_extended_validation.py
    if ($LASTEXITCODE -ne 0) { throw 'CPU evidence audit failed' }
}
& $Python scripts/summarize_extended_validation.py --video-root outputs/extended_video_validation_gpu518 --evidence docs/evidence/extended_validation_gpu518 --report docs/extended_validation_gpu518.md
if ($LASTEXITCODE -ne 0) { throw 'GPU evidence audit failed' }
if (Test-Path outputs/extended_video_validation_cpu/summary.json) {
    & $Python scripts/compare_video_resolutions.py
    if ($LASTEXITCODE -ne 0) { throw 'Resolution audit failed' }
}
