@echo off

REM V7P3R v13.0 Tactical Enhancement Engine for Arena
REM Updated: October 2025
REM Features: Pin/Fork/Skewer detection, Dynamic piece values, Tal complexity

cd /d "%~dp0"

REM Launch engine with Python 3.13
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/v7p3r_uci.py

REM Error handling for Arena
if errorlevel 1 (
    echo Error: Could not start V7P3R v13.0 Engine
    echo Check that Python 3.13 is installed at the specified path
    pause
)