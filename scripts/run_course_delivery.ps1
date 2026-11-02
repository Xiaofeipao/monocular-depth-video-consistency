param([string]$Python='python',[string]$Archive='depth_results.zip')
$ErrorActionPreference='Stop'
Set-Location (Join-Path $PSScriptRoot '..')
. ./scripts/project_cache.ps1
$env:PYTHONPATH="$PWD/.runtime;$PWD"
# Part 1: saved stereo rescoring and framewise relative baseline.
& $Python scripts/rescore_course_stereo.py
if ($LASTEXITCODE -ne 0) { throw 'Stereo rescoring failed' }
& $Python scripts/complete_submission_video.py --model dav2
if ($LASTEXITCODE -ne 0) { throw 'DAV2 video inference failed' }
# Part 2: relative video baseline; reuse the existing 654-image metric runs.
& $Python scripts/complete_submission_video.py --model vda
if ($LASTEXITCODE -ne 0) { throw 'VDA video inference failed' }
# Part 3: fixed C component removals, cases and provided/self-recorded apps.
$env:PYTHONPATH="$PWD/.runtime_depthpro;$PWD/.runtime;$PWD"
if (!(Test-Path outputs/course_submission/C_ablation/summary.json)) {
    & $Python scripts/ablate_course_refinement.py
    if ($LASTEXITCODE -ne 0) { throw 'C ablation failed' }
}
& $Python scripts/build_course_media.py
if ($LASTEXITCODE -ne 0) { throw 'Media export failed' }
& $Python scripts/prepare_course_delivery.py
if ($LASTEXITCODE -ne 0) { throw 'Evidence preparation failed' }
# Use a NEW archive name when repeating, preserving previous packages.
& $Python -m src.package_submission --manifest configs/submission.example.json --output $Archive
if ($LASTEXITCODE -ne 0) { throw 'Package validation failed' }
