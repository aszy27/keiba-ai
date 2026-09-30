' v2/live/run_snapshot_hidden.vbs
' Launch run_snapshot.ps1 from Task Scheduler without any console window.
' (powershell.exe -WindowStyle Hidden still flashes a console; wscript.exe has no window.)
Dim ps1
ps1 = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\run_snapshot.ps1"
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, True
