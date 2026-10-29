param(
    [string]$Python = 'python',
    [string]$Output = 'outputs/full_image_benchmark'
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'project_cache.ps1')
function Invoke-ProjectPython([string[]]$Arguments) {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline failed: $Arguments" }
}
$originalPythonPath = $env:PYTHONPATH
try {
    New-Item -ItemType Directory -Force -Path $Output | Out-Null
    $env:PYTHONPATH = (Join-Path (Get-Location) '.runtime') + [IO.Path]::PathSeparator + $originalPythonPath
    # Frozen protocol settings are copied unchanged except the local source root.
    $configScript = @'
from pathlib import Path
import sys,yaml
root=Path(sys.argv[1])
for name,repo in [('depth_pro','third_party/ml-depth-pro'),('unidepth_v2','third_party/UniDepth')]:
    config=yaml.safe_load(Path('configs/part2/'+name+'_nyuv2.yaml').read_text())
    config['model']['repository']=repo
    config['visualization']['max_panels']=32
    (root/(name+'.yaml')).write_text(yaml.safe_dump(config,sort_keys=False))
'@
    Invoke-ProjectPython -Arguments @('-c',$configScript,$Output)
    Invoke-ProjectPython -Arguments @('-m','pipelines.part1.run_middlebury','--output-dir',"$Output/middlebury")
    $env:PYTHONPATH = (Join-Path (Get-Location) '.runtime_depthpro') + [IO.Path]::PathSeparator + $env:PYTHONPATH
    Invoke-ProjectPython -Arguments @('-m','pipelines.part2.run_depth_pro_nyuv2','--config',"$Output/depth_pro.yaml",'--output-dir',"$Output/depth_pro_nyuv2",'--device','cuda:0','--resume')
    $env:PYTHONPATH = (Join-Path (Get-Location) '.runtime') + [IO.Path]::PathSeparator + $originalPythonPath
    Invoke-ProjectPython -Arguments @('-m','pipelines.part2.run_unidepth_v2_nyuv2','--config',"$Output/unidepth_v2.yaml",'--output-dir',"$Output/unidepth_v2_nyuv2",'--device','cuda:0','--resume')
    foreach ($model in @('depth_pro','unidepth_v2')) {
        Invoke-ProjectPython -Arguments @('scripts/evaluate_nyuv2_refinement.py','--run',"$Output/${model}_nyuv2",'--output',"$Output/${model}_C_cuda")
    }
} finally {
    $env:PYTHONPATH = $originalPythonPath
}
