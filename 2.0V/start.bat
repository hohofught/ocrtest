@echo off
TITLE ScanBot Launcher

:: 1. Change directory to the batch file's location
cd /d %~dp0

:: 2. Start the Flask web app (web_app.py)
echo [1/2] Starting Flask (web_app.py) server...
START "ScanBot Server" python ocr.py

:: 3. Wait 5 seconds for the server to initialize
echo [2/2] Waiting 5 seconds for the server to initialize...
timeout /t 5 /nobreak > nul
