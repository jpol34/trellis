# Run this yourself (not through the agent): the real, billed, three-arm #15 eval run against
# the held-out 100-transcript eval set (data/eval_set/eval.jsonl).
#
# Sources OPENAI_API_KEY and TYPESAFE_API_KEY from Strongbox for this process only -- nothing is
# written to disk, and no secret value is ever printed. The trellis arm needs neither (it's a
# local checkpoint), gpt-5.1 needs OPENAI_API_KEY, jev needs TYPESAFE_API_KEY (and, if you keep
# it in Strongbox too, JEV_BASE_URL -- otherwise the SDK's own default is used).
#
#   ./scripts/run_ticket15_eval.ps1
#   ./scripts/run_ticket15_eval.ps1 -Limit 10          # quick smoke run over the first 10 items
#   ./scripts/run_ticket15_eval.ps1 -Arm trellis        # only run one arm, e.g. to re-check
#                                                        # trellis alone without re-billing the
#                                                        # other two

param(
    [int]$Limit = 0,
    [string[]]$Arm = @(),
    [string]$OutPath = "data/eval_report.md"
)

$ErrorActionPreference = "Stop"

$env:OPENAI_API_KEY = Get-StrongboxSecret -Name "OPENAI_API_KEY"
$env:TYPESAFE_API_KEY = Get-StrongboxSecret -Name "TYPESAFE_API_KEY::jpol34/trellis"
try {
    $env:JEV_BASE_URL = Get-StrongboxSecret -Name "JEV_BASE_URL::jpol34/trellis"
} catch {
    Write-Host "JEV_BASE_URL not in Strongbox -- leaving unset, jev arm will use the SDK's default base URL."
}

$argList = @("run", "trellis-eval", "--out", $OutPath)
if ($Limit -gt 0) {
    $argList += @("--limit", $Limit)
}
foreach ($a in $Arm) {
    $argList += @("--arm", $a)
}

Write-Host "Running: uv $($argList -join ' ')"
uv @argList

Write-Host ""
Write-Host "Report written to $OutPath"
