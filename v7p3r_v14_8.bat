@echo off
REM V7P3R v14.8 UCI Launcher for Universal Puzzle Analyzer
REM This wrapper allows the Python-based V7P3R engine to be tested by tools expecting .bat/.exe engines

cd /d "%~dp0"
python src\v7p3r_uci.py
