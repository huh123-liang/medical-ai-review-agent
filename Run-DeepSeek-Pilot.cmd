@echo off
cd /d "%~dp0"
echo REAL DEEPSEEK CALLS - DEVELOPMENT split only.
echo Local total budget <= CNY30; this pilot's conservative batch cap is CNY0.50.
echo Model is selected in .env.local. Default: deepseek-flash. No provider fallback.
".venv\Scripts\python.exe" -m medical_jury.cli benchmark --provider deepseek --split dev --limit 10 --max-batch-cost 0.50
".venv\Scripts\python.exe" -m medical_jury.cli budget
pause
