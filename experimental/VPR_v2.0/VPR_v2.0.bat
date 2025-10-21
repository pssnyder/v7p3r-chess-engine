@echo off
REM VPR Pure Potential Engine - Arena Deployment Launcher
REM The Revolutionary Piece Potential Engine for Arena Chess GUI

echo Starting VPR Pure Potential Engine v2.0...
echo "Piece value = attacks + mobility (NO material assumptions)"

REM Change to the engine directory
cd /d "%~dp0"

REM Use the working Python 3.13 installation
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/vpr_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting TAL-BOT engine!
    echo Make sure Python and python-chess are installed.
    echo Press any key to view troubleshooting info...
    pause > nul
    echo.
    echo Troubleshooting:
    echo 1. Install Python: https://python.org/downloads
    echo 2. Install python-chess: pip install python-chess==1.999
    echo 3. Verify installation: python -c "import chess; print('OK')"
    echo 4. Check if src\vpr_uci.py exists in engine directory
    echo.
    pause
)