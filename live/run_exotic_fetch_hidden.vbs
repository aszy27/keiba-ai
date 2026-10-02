' live/run_exotic_fetch_hidden.vbs
' Launch run_exotic_fetch.ps1 from Task Scheduler without any console window.
Dim ps1
ps1 = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & "\run_exotic_fetch.ps1"
CreateObject("WScript.Shell").Run "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & ps1 & """", 0, True
