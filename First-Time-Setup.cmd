@echo off
cd /d "%~dp0"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
if not exist ".venv\Scripts\python.exe" (
  python -c "import sys; assert sys.version_info[:2] in ((3,11),(3,12),(3,13)), 'Install Python 3.12 first'"
  if errorlevel 1 goto failed
  python -m venv .venv
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -m pip install -c constraints-tested.txt -e ".[test]"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m medical_jury.cli init
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest
if errorlevel 1 goto failed
echo Setup complete. No model request was sent. See README.md.
pause
exit /b 0
:failed
echo Setup stopped. Use Python 3.11-3.13, preferably 3.12. Existing platform was not modified.
pause
exit /b 1
