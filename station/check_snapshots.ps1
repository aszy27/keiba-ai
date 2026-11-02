# check_snapshots.ps1 -- run BEFORE connecting the two PCs in Syncthing.
# Checked files = exactly the files odds_snapshots.stignore syncs:
#   YYYYMMDD.csv, exotic\YYYYMMDD_trio.csv, exotic\YYYYMMDD_trifecta.csv,
#   exotic\YYYYMMDD_quinella.csv, exotic\YYYYMMDD_exacta.csv (quinella/exacta exist only from 2026-11-07)
# Step 1 (main PC):    make an inventory of its odds_snapshots folder
#   powershell -NoProfile -ExecutionPolicy Bypass -File check_snapshots.ps1 -Inventory C:\Users\aassz\PycharmProjects\keiba\data\v2\odds_snapshots -Out main_snapshots.csv
# Step 2 (station PC): compare the main PC's inventory with this PC's folder
#   powershell -NoProfile -ExecutionPolicy Bypass -File check_snapshots.ps1 -Compare main_snapshots.csv
# Result per file:
#   SAME        both PCs have it, identical (SHA256)       -> OK
#   DIFFERENT   both PCs have it, contents differ          -> must be resolved before connecting (see README)
#   MAIN_ONLY   only the main PC has it                    -> copy it to the station first (see README)
#   STATION_ONLY only the station has it                   -> OK (will be sent to the main PC)
param(
    [string]$Inventory,
    [string]$Out = 'snapshots_inventory.csv',
    [string]$Compare,
    [string]$StationDir = 'C:\Users\aasszy\PycharmProjects\keiba\data\v2\odds_snapshots'
)
$ErrorActionPreference = 'Stop'
$SyncedRe = '^(\d{8}\.csv|exotic\\\d{8}_(trio|trifecta|quinella|exacta)\.csv)$'
function Get-Rel($dir, $file) { $file.FullName.Substring((Resolve-Path $dir).Path.TrimEnd('\').Length + 1) }
function Get-Inv($dir) {
    Get-ChildItem -Path $dir -File -Recurse |
        ForEach-Object { $rel = Get-Rel $dir $_; if ($rel -match $SyncedRe) {
            [pscustomobject]@{ Name = $rel; Length = $_.Length;
                               LastWrite = $_.LastWriteTime.ToString('s'); Sha256 = (Get-FileHash $_.FullName -Algorithm SHA256).Hash } } } |
        Sort-Object Name
}
if ($Inventory) {
    $inv = @(Get-Inv $Inventory)
    $inv | Export-Csv -Path $Out -NoTypeInformation -Encoding UTF8
    Write-Host ("{0} files ({1} exotic) -> {2}" -f $inv.Count, @($inv | Where-Object { $_.Name -like 'exotic\*' }).Count, (Resolve-Path $Out))
    $extra = Get-ChildItem -Path $Inventory -File -Recurse | ForEach-Object { Get-Rel $Inventory $_ } | Where-Object { $_ -notmatch $SyncedRe }
    if ($extra) { Write-Host 'other files in the folder (not synced):'; $extra | ForEach-Object { '  ' + $_ } }
    return
}
if (-not $Compare) { throw 'use -Inventory <dir> or -Compare <main inventory csv>' }
$main = @{}; Import-Csv $Compare | ForEach-Object { $main[$_.Name] = $_ }
$sta  = @{}; Get-Inv $StationDir | ForEach-Object { $sta[$_.Name] = $_ }
$rows = foreach ($n in (@($main.Keys) + @($sta.Keys) | Sort-Object -Unique)) {
    $m = $main[$n]; $s = $sta[$n]
    $state = if ($m -and $s) { if ($m.Sha256 -eq $s.Sha256) { 'SAME' } else { 'DIFFERENT' } } elseif ($m) { 'MAIN_ONLY' } else { 'STATION_ONLY' }
    [pscustomobject]@{ Name = $n; State = $state; MainBytes = $m.Length; StationBytes = $s.Length; MainWrite = $m.LastWrite; StationWrite = $s.LastWrite }
}
$rows | Format-Table -AutoSize
$rows | Group-Object State | ForEach-Object { '{0}: {1}' -f $_.Name, $_.Count }
if ($rows | Where-Object { $_.State -in 'DIFFERENT', 'MAIN_ONLY' }) { Write-Host 'NOT READY: resolve DIFFERENT / MAIN_ONLY before connecting.' -ForegroundColor Red }
else { Write-Host 'READY: no date file conflicts.' -ForegroundColor Green }
