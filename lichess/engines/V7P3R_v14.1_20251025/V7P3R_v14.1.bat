@echo off

REM V7P3R v14.1
REM Updated: October 2025
REM Uses py launcher for cross-machine compatibility

cd /d "%~dp0"

REM Launch engine with Python
py -3 src/v7p3r_uci.py

REM Error handling for Arena
if errorlevel 1 (
    echo Error: Could not start V7P3R v14.1 Engine
    echo Check that Python 3.13 is installed at the specified path
    echo Required packages: python-chess (install via: pip install python-chess)
    pause
)