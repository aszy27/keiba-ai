# live/run_exotic_fetch.ps1
# 組み合わせ券の確定オッズを、平日（月〜木）に少しずつ取るためにタスクスケジューラ（keiba-exotic-odds）から起動する入れ物。
#   起動するのは 0時と、ログオンしたとき（2分後）。途中でPCを切っても、次に起動したときに続きから取る
#   - 取る順番: 3連複・3連単（2021〜2025年）→ 3連単（2020年）→ 馬連・馬単（2020〜2025年）。前が終わったら同じ日の残りの回数で次へ進む
#     （docs/rebuild_plan.md「3連複・3連単の検証の手順」「馬連・馬単の検証の手順」）
#   - 1日（0時区切り）3,000リクエストまで、3秒に1回程度（--daily-max-requests 3000 --sleep 3）。使った回数は
#     data/v2/odds_exotic/daily_state.json に残すので、同じ日に何度起動しても合計3,000回を超えない。制限で止めた日は再開しない
#   - 金〜日は何もしない（開催日のスナップショットと同じ API を取り合わないため。--weekdays-only）
#   - 取得済みのレースは飛ばすので、続きから取る（シャットダウンで失うのは保存前の最大99R分だけ）。失敗が20回続いたらブロックの兆候とみてその日は止める
#   - 出力は logs/exotic_YYYYMMDD.log。取得中はPCをスリープさせない（シャットダウンは止めない）
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
    & cmd /c """$py"" -m scrape odds-exotic --job 2021-2025:7,8 --job 2020:8 --job 2020-2025:4,6 --daily-max-requests 3000 --sleep 3 --weekdays-only >> ""$log"" 2>&1"
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') 終了 (exit=$LASTEXITCODE) ===" | Out-File -FilePath $log -Append -Encoding utf8
} finally {
    [KeibaExotic.Power]::SetThreadExecutionState(0x80000000) | Out-Null
}
