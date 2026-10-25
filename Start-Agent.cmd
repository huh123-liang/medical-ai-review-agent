@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Please install the independent environment first. See README.md.
  pause
  exit /b 1
)
echo Starting the LOCAL API only. No model request is sent on startup.
echo Close this window or press Ctrl+C to stop. Old platform remains unchanged.
".venv\Scripts\python.exe" -m medical_jury.cli serve --port 4205
pause
