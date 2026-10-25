@echo off
cd /d "%~dp0"
echo OFFLINE MOCK PIPELINE TEST - no clinical agreement measurement, no paid model calls.
".venv\Scripts\python.exe" -m medical_jury.cli benchmark --provider mock --split dev --limit 10
pause
