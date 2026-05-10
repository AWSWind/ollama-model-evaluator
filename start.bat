@echo off
rem One-double-click Windows launcher.
rem Runs scripts\start.ps1 so the user never has to touch PowerShell.
rem Forwards every argument so `start.bat -Dev -Port 9000` still works.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
