# Run this yourself (not through the agent): validates ticket #16 (bounded concurrent
# generation) with a live, billed 20-record pilot against the real Anthropic API.
#
# Sources ANTHROPIC_API_KEY (and GENERATION_CLAUDE_MODEL, if you keep it in Strongbox too)
# for this process only -- nothing is written to disk. If GENERATION_CLAUDE_MODEL isn't a
# Strongbox secret for you, set it yourself before running this script, e.g.:
#   $env:GENERATION_CLAUDE_MODEL = "claude-sonnet-4-5"
#
#   ./scripts/run_ticket16_pilot.ps1

$ErrorActionPreference = "Stop"

$env:ANTHROPIC_API_KEY = Get-StrongboxSecret -Name "ANTHROPIC_API_KEY"
if (-not $env:GENERATION_CLAUDE_MODEL) {
    try {
        $env:GENERATION_CLAUDE_MODEL = Get-StrongboxSecret -Name "GENERATION_CLAUDE_MODEL"
    } catch {
        throw "GENERATION_CLAUDE_MODEL is not set and isn't in Strongbox either -- " +
            "set `$env:GENERATION_CLAUDE_MODEL` yourself before re-running this script."
    }
}

$outDir = "data/pilot_concurrency_test"

Write-Host "Running 20-record pilot at --concurrency 10 (sequential baseline: ~2.6 records/min)..."
$start = Get-Date
uv run trellis-gen-training --pilot 20 --concurrency 10 --out-dir $outDir
$elapsed = (Get-Date) - $start
Write-Host ""
Write-Host "Elapsed: $($elapsed.TotalSeconds.ToString('F1'))s ($((20 / $elapsed.TotalMinutes).ToString('F2')) records/min)"

Write-Host ""
Write-Host "--- Verification ---"
$checkpointPath = Join-Path $outDir "checkpoint.jsonl"
if (Test-Path $checkpointPath) {
    Write-Host "FAIL: checkpoint.jsonl still present at $checkpointPath (should be deleted on success)"
} else {
    Write-Host "OK: checkpoint.jsonl deleted after success"
}

$allIds = @()
foreach ($split in "train", "val", "held_out_internal") {
    $recordsPath = Join-Path $outDir "$split/records.jsonl"
    if (Test-Path $recordsPath) {
        $lines = Get-Content $recordsPath
        $ids = $lines | ForEach-Object { ($_ | ConvertFrom-Json).item_id }
        $allIds += $ids
        Write-Host "$split : $($lines.Count) records"
    } else {
        Write-Host "$split : (missing)"
    }
}
$uniqueCount = ($allIds | Select-Object -Unique).Count
Write-Host "Total records: $($allIds.Count), unique item_ids: $uniqueCount"
if ($allIds.Count -ne 20 -or $uniqueCount -ne 20) {
    Write-Host "FAIL: expected exactly 20 unique item_ids across splits"
} else {
    Write-Host "OK: exactly 20 unique item_ids, no duplicates"
}

Write-Host ""
Write-Host "Deleting scratch pilot output at $outDir ..."
Remove-Item -Recurse -Force $outDir
Write-Host "Done."
