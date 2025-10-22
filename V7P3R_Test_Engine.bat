@echo off

echo Starting V7P3R Current Test Engine...

cd /d "%~dp0"

REM Use the working Python 3.13 installation
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/v7p3r_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting V7P3R Current Test Engine!
    pause > nul
    echo.
    pause
)