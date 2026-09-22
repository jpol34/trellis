# Run this yourself (not through the agent): merges TYPESAFE_API_KEY, sourced from Strongbox
# rather than pasted into chat or hardcoded, into the local .env for the Jev reference arm.
# Requires the Strongbox PowerShell module. All other existing keys in .env are preserved.
#
# typesafe.ai keys are provisioned per-project on their side (see laya-bench's own
# scripts/configure-typesafe-secret.ps1), so this pulls a trellis-scoped secret name rather than
# a generic one. Update $secretName below if the Strongbox entry for trellis's key differs.
#
#   ./scripts/configure-typesafe-secret.ps1

$ErrorActionPreference = "Stop"

$secretName = "TYPESAFE_API_KEY::jpol34/trellis"
$typesafeKey = Get-StrongboxSecret -Name $secretName

$envPath = Join-Path $PSScriptRoot "..\.env"

$lines = [ordered]@{}
if (Test-Path $envPath) {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^([^=]+)=(.*)$') {
            $lines[$Matches[1]] = $Matches[2]
        }
    }
}
$lines["TYPESAFE_API_KEY"] = $typesafeKey

($lines.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n" |
    Set-Content -Path $envPath -NoNewline

Write-Host "Wrote $envPath (merged). It is gitignored — never commit it."
