@echo off
REM VPR Chaos Capitalization Engine v6.0 - Arena Deployment Launcher
REM Revolutionary Dual-Brain Chess Engine: "We create chaos, they respond traditionally, we capitalize"

echo Starting VPR Chaos Capitalization Engine v6.0...
echo "Revolutionary dual-brain asymmetric thinking architecture!"
echo "Position-aware algorithm selection with MCTS for ultra-chaotic positions"

REM Change to the engine directory
cd /d "%~dp0"

REM Use the working Python 3.13 installation
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/vpr_uci.py

REM Pause if there's an error
if errorlevel 1 (
    echo.
    echo Error starting VPR Chaos Capitalization Engine v6.0!
    echo Make sure Python and python-chess are installed.
    echo Press any key to view troubleshooting info...
    pause > nul
    echo.
    echo Troubleshooting:
    echo 1. Python 3.13 path: C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe
    echo 2. Install python-chess: pip install python-chess
    echo 3. Verify installation: python -c "import chess; print('OK')"
    echo 4. Check if src\vpr_uci.py exists in engine directory
    echo 5. VPR v6.0 - Revolutionary Dual-Brain Chaos Capitalization Engine
    echo.
    pause
)