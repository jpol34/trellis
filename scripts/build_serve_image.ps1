# Run this yourself (not through the agent): builds `stack/Dockerfile.serve` (the RunPod
# Serverless inference image, no checkpoint baked in -- see #20) and pushes it to the registry
# RunPod will pull from. Requires Docker installed locally and already logged in to that
# registry (`docker login <registry host>`) -- this script doesn't touch registry credentials
# itself, same as `trellis-pod-up`'s `RUNPOD_REGISTRY_ID` only ever references a RunPod-stored
# credential by id, never a raw docker login secret.
#
#   ./scripts/build_serve_image.ps1
#   ./scripts/build_serve_image.ps1 -ImageTag "myregistry.io/trellis-serve:v2"

param(
    [string]$ImageTag = ""
)

$ErrorActionPreference = "Stop"

Push-Location (Join-Path $PSScriptRoot "..")
try {
    if (-not $ImageTag) {
        $ImageTag = (Get-Content .env -ErrorAction SilentlyContinue |
            Where-Object { $_ -match '^RUNPOD_SERVE_IMAGE=(.+)$' } |
            ForEach-Object { $Matches[1] })
    }
    if (-not $ImageTag) {
        Write-Host "No image tag given and RUNPOD_SERVE_IMAGE not set in .env." -ForegroundColor Red
        Write-Host "Pass -ImageTag, or set RUNPOD_SERVE_IMAGE=<registry>/<repo>:<tag> in .env first."
        exit 1
    }

    Write-Host "Building $ImageTag from stack/Dockerfile.serve ..."
    docker build -f stack/Dockerfile.serve -t $ImageTag .

    Write-Host "Pushing $ImageTag ..."
    docker push $ImageTag

    Write-Host ""
    Write-Host "Pushed $ImageTag." -ForegroundColor Green
    Write-Host "If this is a new tag, update RUNPOD_SERVE_IMAGE in .env and the RunPod Serverless" -ForegroundColor Yellow
    Write-Host "endpoint's image reference to match before the next deploy." -ForegroundColor Yellow
} finally {
    Pop-Location
}
