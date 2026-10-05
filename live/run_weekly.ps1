# live/run_weekly.ps1
# 毎週の取り込みを、タスクスケジューラ（keiba-weekly-ingest、月曜の夜）から起動するための入れ物。
#   python -m scrape weekly → python -m prep.ingest → python -m prep.features → テスト → 前向き検証の集計（c_all・cand2）
#   - 取り込みの検査で ERROR があれば（prep.ingest の終了コードが 0 以外）、特徴量は作り直さずに止める
#   - 出力は logs/weekly_YYYYMMDD.log。取得中はPCをスリープさせない
#   - 特徴量の作り直しはメモリを多く使う（数GB）ので、ゲームなどで重いときは失敗することがある（ログに残る。手動でやり直せる）
$ErrorActionPreference = 'Continue'
$proj = 'C:\Users\aassz\PycharmProjects\keiba'
$py   = 'C:\Users\aassz\anaconda3\envs\keiba-ai\python.exe'
$dir  = Join-Path $proj 'logs'
$log  = Join-Path $dir ('weekly_{0}.log' -f (Get-Date -Format 'yyyyMMdd'))
if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }

Add-Type -Name Power -Namespace KeibaWeekly -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)]
public static extern uint SetThreadExecutionState(uint esFlags);
'@
[KeibaWeekly.Power]::SetThreadExecutionState(0x80000001) | Out-Null

function Step($name, $cmd) {
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $name ===" | Out-File -FilePath $log -Append -Encoding utf8
    & cmd /c """$py"" -u $cmd >> ""$log"" 2>&1"
    $code = $LASTEXITCODE
    "=== $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $name 終了 (exit=$code) ===" | Out-File -FilePath $log -Append -Encoding utf8
    return $code
}

try {
    Set-Location $proj
    $env:PYTHONIOENCODING = 'utf-8'
    $null = Step '結果・血統・ラップ・払戻・追い切り・オッズの取得' '-m scrape weekly'
    $ingest = Step '取り込み（検査つき）' '-m prep.ingest'
    if ($ingest -ne 0) {
        "取り込みの検査で ERROR があったので、特徴量は作り直さずに止める（data/v2/check_report.txt を確認）" | Out-File -FilePath $log -Append -Encoding utf8
    } else {
        $features = Step '特徴量の作り直し' '-m prep.features'
        $null = Step 'テスト' '-m pytest tests scrape/tests -q'
        if ($features -eq 0) {
            $null = Step '前向き検証の集計（c_all）' '-W ignore -m predict --candidate c_all'
            $null = Step '前向き検証の集計（cand2）' '-W ignore -m predict --candidate cand2'
        }
    }
} finally {
    [KeibaWeekly.Power]::SetThreadExecutionState(0x80000000) | Out-Null
}
