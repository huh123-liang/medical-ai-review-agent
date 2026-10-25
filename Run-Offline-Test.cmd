@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest
if errorlevel 1 (
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m medical_jury.cli evaluate --input examples\synthetic-request.json --provider mock
pause
