@echo off
REM VPR Pure Potential Engine v3.0 - Arena Deployment Launcher
REM Revolutionary piece potential based chess AI

echo Starting VPR Pure Potential Engine v3.0...
echo "Piece value = attacks + mobility (NO material assumptions)"
echo "Focus on highest/lowest potential pieces only"

REM Change to the engine directory
cd /d "%~dp0"

REM Use the working Python 3.13 installation
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/vpr_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting VPR v3.0 engine!
    echo Make sure Python and python-chess are installed.
    echo Press any key to view troubleshooting info...
    pause > nul
    echo.
    echo Troubleshooting:
    echo 1. Python 3.13 path: C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe
    echo 2. Install python-chess: pip install python-chess
    echo 3. Verify installation: python -c "import chess; print('OK')"
    echo 4. Check if src\vpr_uci.py exists in engine directory
    echo.
    echo VPR v3.0 Features:
    echo - Pure piece potential evaluation (attacks + mobility)
    echo - Focus on highest/lowest potential pieces only
    echo - Lenient pruning preserves chaos
    echo - 12K+ NPS, depth 15+ performance
    echo.
    pause
)