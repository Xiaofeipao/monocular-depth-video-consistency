param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$DAV2Repo,
    [Parameter(Mandatory=$true)][string]$VDARepo,
    [Parameter(Mandatory=$true)][string]$Video,
    [Parameter(Mandatory=$true)][string]$Output,
    [int]$MaxFrames = 32
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_cache.ps1')
if (Test-Path -LiteralPath $Output) { throw 'Choose a new output directory' }
$env:PYTHONPATH = (Join-Path $PSScriptRoot '..\.runtime') + [IO.Path]::PathSeparator + $env:PYTHONPATH
function Invoke-CheckedPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "GPU pipeline failed with exit code $LASTEXITCODE" }
}
Invoke-CheckedPython -Arguments @('-m','pipelines.part1.run_framewise_depth','--input',$Video,'--repo',$DAV2Repo,'--checkpoint','checkpoints/depth_anything_v2_vits.pth','--output',"$Output/dav2",'--device','cuda','--fp16','--fps','10','--max-frames',"$MaxFrames",'--max-side','640','--input-size','518','--vis-min','0','--vis-max','50')
Invoke-CheckedPython -Arguments @('-m','pipelines.part2.run_video_depth','--input',$Video,'--repo',$VDARepo,'--checkpoint','checkpoints/video_depth_anything_vits.pth','--output',"$Output/vda",'--device','cuda','--fp16','--fps','10','--max-frames',"$MaxFrames",'--max-side','640','--input-size','518','--vis-min','0','--vis-max','50')
Invoke-CheckedPython -Arguments @('-m','pipelines.part2.run_video_depth','--input',$Video,'--repo',$VDARepo,'--checkpoint','checkpoints/metric_video_depth_anything_vits.pth','--metric','--output',"$Output/metric_vda",'--device','cuda','--fp16','--fps','10','--max-frames',"$MaxFrames",'--max-side','640','--input-size','518','--vis-min','0.1','--vis-max','20')
Invoke-CheckedPython -Arguments @('-m','pipelines.part3.compare_directions','--run',"$Output/metric_vda",'--output',"$Output/directions",'--repeats','3')
Invoke-CheckedPython -Arguments @('-m','pipelines.part3.run_temporal_fusion','--run',"$Output/metric_vda",'--output',"$Output/raft_A6",'--flow','raft','--flow-checkpoint','checkpoints/raft_small_C_T_V2-01064c6d.pth','--device','cuda')
Invoke-CheckedPython -Arguments @('-m','src.evaluate_video','--run',"$Output/raft_A6")
