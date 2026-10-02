# live/run_exotic_fetch.ps1
# 組み合わせ券（3連複・3連単）の確定オッズを、平日（月〜木）に少しずつ取るためにタスクスケジューラ（keiba-exotic-odds）から起動する入れ物。
#   - 1回3,000リクエストまで、3秒に1回程度（python -m scrape odds-exotic --max-requests 3000 --sleep 3）
#   - 金〜日は何もしない（開催日のスナップショットと同じ API を取り合わないため。--weekdays-only）
#   - 取得済みのレースは飛ばすので、毎日続きから取る。失敗が20回続いたらブロックの兆候とみてその日は止める
#   - 出力は logs/exotic_YYYYMMDD.log。取得中はPCをスリープさせない
$ErrorActionPreference = 'Continue'
$proj = 'C:\Users\aassz\PycharmProjects\keiba'
$py   = 'C:\Users\aassz\anaconda3\envs\keiba-ai\python.exe'
$dir  = Join-Path $proj 'logs'
$log  = Join-Path $dir ('exotic_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

Add-Type -Name Power -Namespace KeibaExotic -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
[KeibaExotic.Power]::SetThreadExecutionState(0x80000001) | Out-Null

"=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 起動 ===" | Out-File -FilePath $log -Append -Encoding utf8
try {
    Set-Location $proj
    $env:PYTHONIOENCODING = 'utf-8'
    & cmd /c """$py"" -m scrape odds-exotic --years 2021,2022,2023,2024,2025 --types 7,8 --max-requests 3000 --sleep 3 --weekdays-only >> ""$log"" 2>&1"
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 終了 (exit=$LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8
} finally {
    [KeibaExotic.Power]::SetThreadExecutionState(0x80000000) | Out-Null
}
