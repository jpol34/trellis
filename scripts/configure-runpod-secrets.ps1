# Run this yourself (not through the agent): merges RUNPOD_API_KEY, sourced from Strongbox
# rather than pasted into chat or hardcoded, into the local .env. Requires the Strongbox
# PowerShell module. Any other existing keys in .env are preserved untouched.
#
#   ./scripts/configure-runpod-secrets.ps1

$ErrorActionPreference = "Stop"

$runpodKey = Get-StrongboxSecret -Name "RUNPOD_API_KEY"

$envPath = Join-Path $PSScriptRoot "..\.env"

$lines = [ordered]@{}
if (Test-Path $envPath) {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^([^=]+)=(.*)$') {
            $lines[$Matches[1]] = $Matches[2]
        }
    }
}
$lines["RUNPOD_API_KEY"] = $runpodKey

($lines.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n" |
    Set-Content -Path $envPath -NoNewline

Write-Host "Wrote $envPath (merged). It is gitignored -- never commit it."
