@echo off
REM V7P3R Chess Engine - Arena Deployment Launcher
REM Launches V7P3R engine with UCI protocol for Arena compatibility

echo Starting V7P3R Chess Engine v12.x...

REM Change to the engine directory
cd /d "%~dp0"

REM Run the V7P3R UCI interface
python src\v7p3r_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting V7P3R engine!
    echo Make sure Python and python-chess are installed.
    echo.
    pause
)