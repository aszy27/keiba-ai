# station_run.ps1  (scraping station PC)  -- PRODUCTION, for after the cut-over.
# This PC only records pre-race odds: `python -m scrape snapshot` (same job as the repo's live/run_snapshot.ps1,
# but with this PC's paths). scrape weekly / odds-exotic stay on the main PC.
# Lives in <repo>\station\ and finds the repo from its own location.
# Writes: data\v2\odds_snapshots\YYYYMMDD.csv and exotic\YYYYMMDD_{trio,trifecta,quinella,exacta}.csv (synced to the main PC),
#         data\v2\odds_snapshots\limit_state.json and
#         data\scrape_cooldown.json (local only), logs\station\snapshot_YYYYMMDD.log (local only).
# Exits at once on non-race days, so it can run every day; re-runs skip (race, minutes) pairs already fetched.
$ErrorActionPreference = 'Continue'
$proj = Split-Path $PSScriptRoot -Parent
$py   = 'C:\Users\aasszy\anaconda3\envs\keiba-ai\python.exe'
$dir  = Join-Path $proj 'logs\station'
$log  = Join-Path $dir ('snapshot_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
New-Item -ItemType Directory -Force $dir | Out-Null

Add-Type -Name Power -Namespace KeibaStation -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
[KeibaStation.Power]::SetThreadExecutionState(0x80000001) | Out-Null   # no sleep while running

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') start ===" | Out-File -FilePath $log -Append -Encoding utf8
try {
    Set-Location $proj
    $env:PYTHONIOENCODING = 'utf-8'
    & cmd /c """$py"" -u -m scrape snapshot >> ""$log"" 2>&1"
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') end (exit=$LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8
} finally {
    [KeibaStation.Power]::SetThreadExecutionState(0x80000000) | Out-Null
}
