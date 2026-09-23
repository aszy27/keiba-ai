# v2/run_snapshot.ps1
# 開催日のオッズスナップショット取得（v2/scrape_odds_snapshot.py）をタスクスケジューラから起動するための入れ物。
#   - 開催日でなければスクリプト側がすぐ終了するので、毎日実行してよい
#   - 取得中はPCをスリープさせない（画面は消えてよい）
#   - 出力は logs/snapshot_YYYYMMDD.log に追記。30分ごとの再実行で取りこぼしを拾い直す
#     （取得済みの (レース, 何分前) はスクリプト側が飛ばす）
$ErrorActionPreference = 'Continue'
$proj = 'C:\Users\aassz\PycharmProjects\keiba'
$py   = 'C:\Users\aassz\anaconda3\envs\keiba-ai\python.exe'
$dir  = Join-Path $proj 'logs'
$log  = Join-Path $dir ('snapshot_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

Add-Type -Name Power -Namespace Keiba -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
# ES_CONTINUOUS(0x80000000) | ES_SYSTEM_REQUIRED(0x00000001) : 実行中はスリープさせない
[Keiba.Power]::SetThreadExecutionState(0x80000001) | Out-Null

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 起動 ===" | Out-File -FilePath $log -Append -Encoding utf8
try {
    Set-Location $proj
    $env:PYTHONIOENCODING = 'utf-8'
    # 文字化けを避けるため、リダイレクトは cmd 側（バイト単位）で行う
    & cmd /c """$py"" -m v2.scrape_odds_snapshot >> ""$log"" 2>&1"
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 終了 (exit=$LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8
} finally {
    [Keiba.Power]::SetThreadExecutionState(0x80000000) | Out-Null
}
