# Run this yourself (not through the agent): merges ANTHROPIC_API_KEY and OPENAI_API_KEY,
# sourced from Strongbox rather than pasted into chat or hardcoded, into the local .env.
# Requires the Strongbox PowerShell module. Any other existing keys in .env are preserved
# untouched.
#
#   ./scripts/configure-generation-secrets.ps1

$ErrorActionPreference = "Stop"

$anthropicKey = Get-StrongboxSecret -Name "ANTHROPIC_API_KEY"
$openaiKey = Get-StrongboxSecret -Name "OPENAI_API_KEY"

$envPath = Join-Path $PSScriptRoot "..\.env"

$lines = [ordered]@{}
if (Test-Path $envPath) {
    foreach ($line in Get-Content $envPath) {
        if ($line -match '^([^=]+)=(.*)$') {
            $lines[$Matches[1]] = $Matches[2]
        }
    }
}
$lines["ANTHROPIC_API_KEY"] = $anthropicKey
$lines["OPENAI_API_KEY"] = $openaiKey

($lines.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join "`n" |
    Set-Content -Path $envPath -NoNewline

Write-Host "Wrote $envPath (merged). It is gitignored -- never commit it."
