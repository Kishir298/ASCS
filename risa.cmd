@echo off
rem risa launcher (AppControl-safe shim for `python -m agent`)
setlocal
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%.venv\Scripts\python.exe" -m agent %*
endlocal