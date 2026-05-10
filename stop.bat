@echo off
rem Windows convenience: double-click to stop everything launched by start.bat.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop.ps1" %*
