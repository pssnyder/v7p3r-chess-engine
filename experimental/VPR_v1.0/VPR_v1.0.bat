@echo off
REM TAL-BOT Chess Engine - Arena Deployment Launcher
REM The Revolutionary Entropy Engine for Arena Chess GUI

echo Starting TAL-BOT (The Entropy Engine) v1.0...
echo "Into the dark forest where only we know the way out"

REM Change to the engine directory
cd /d "%~dp0"

REM Run the TAL-BOT UCI interface (in src subdirectory)
python src\vpr_uci.py

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