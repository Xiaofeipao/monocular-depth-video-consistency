# Dot-source before dependency installation or model commands:
# . ./scripts/project_cache.ps1
$projectCacheRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../.project_cache'))
$projectCacheLocations = @{
    PIP_CACHE_DIR = 'pip'
    HF_HOME = 'huggingface'
    HUGGINGFACE_HUB_CACHE = 'huggingface/hub'
    HF_HUB_CACHE = 'huggingface/hub'
    HF_XET_CACHE = 'huggingface/xet'
    TORCH_HOME = 'torch'
    XDG_CACHE_HOME = 'xdg'
    TMP = 'tmp'
    TEMP = 'tmp'
}
foreach ($projectCacheVariable in $projectCacheLocations.Keys) {
    $projectCachePath = Join-Path $projectCacheRoot $projectCacheLocations[$projectCacheVariable]
    New-Item -ItemType Directory -Force -Path $projectCachePath | Out-Null
    [Environment]::SetEnvironmentVariable($projectCacheVariable,$projectCachePath,'Process')
}
