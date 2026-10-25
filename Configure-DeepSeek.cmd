@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please install the independent environment first. See README.md.
  pause
  exit /b 1
)
echo Do not edit this script. Enter your key at the hidden input prompt below.
".venv\Scripts\python.exe" -m medical_jury.cli configure
pause
