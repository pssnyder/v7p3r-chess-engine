@echo off
REM VPR Chess Engine - Arena Deployment Launcher
REM Launches VPR engine with UCI protocol for Arena compatibility

echo Starting VPR Chess Engine v1.0...

REM Change to the engine directory
cd /d "%~dp0"

REM Run the VPR UCI interface
python src\vpr_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting VPR engine!
    echo Make sure Python and python-chess are installed.
    echo.
    pause
)