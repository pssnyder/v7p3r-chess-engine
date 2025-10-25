@echo off

REM V7P3R v13.x Revolutionary Move Ordering Engine for Arena & Lichess
REM Updated: October 2025
REM Features: 84% search tree reduction, V13.x move ordering, tactical detection
REM Performance: 1100+ NPS, 100% UCI compliance, tournament-ready

cd /d "%~dp0"

REM Launch engine with Python 3.13
"C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/v7p3r_uci.py

REM Error handling for Arena
if errorlevel 1 (
    echo Error: Could not start V7P3R v13.x Engine
    echo Check that Python 3.13 is installed at the specified path
    echo Required packages: python-chess (install via: pip install python-chess)
    pause
)