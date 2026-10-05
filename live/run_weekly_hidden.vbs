' live/run_weekly_hidden.vbs
' Launch run_weekly.ps1 from Task Scheduler without any console window.
Dim ps1
ps1 = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\run_weekly.ps1"
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, True
