@echo off
rem risa launcher (AppControl-safe shim for the terminal entry point)
setlocal
set "SCRIPT_DIR=%~dp0"
"%SCRIPT_DIR%.venv\Scripts\python.exe" -m agent.terminal %*
endlocal