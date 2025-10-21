@echo off
REM VPR Lightning Fast Engine v4.0 - Arena Deployment Launcher
REM Revolutionary Bitboard-Based Piece Potential Engine for Arena Chess GUI

echo Starting VPR Lightning Fast Engine v4.0...
echo "Bitboard flash-layer comparisons for blazing speed!"
echo "Piece value = attacks + mobility (NO material assumptions)"

REM Change to the engine directory
cd /d "%~dp0"

REM Use the working Python 3.13 installation
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/vpr_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting VPR Lightning Fast Engine!
    echo Make sure Python and python-chess are installed.
    echo Press any key to view troubleshooting info...
    pause > nul
    echo.
    echo Troubleshooting:
    echo 1. Python 3.13 path: C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe
    echo 2. Install python-chess: pip install python-chess
    echo 3. Verify installation: python -c "import chess; print('OK')"
    echo 4. Check if src\vpr_uci.py exists in engine directory
    echo 5. VPR v4.0 - Lightning Fast Bitboard Engine
    echo.
    pause
)