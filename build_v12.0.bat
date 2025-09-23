@echo off
REM V7P3R Chess Engine v12.0 Build Script
REM Clean Foundation: v10.8 Baseline + Enhanced Nudge System

echo Building V7P3R Chess Engine v12.0...
echo ==========================================

REM Clean previous builds
if exist "dist\V7P3R_v12.0.exe" del "dist\V7P3R_v12.0.exe"
if exist "build\V7P3R_v12.0" rmdir /s /q "build\V7P3R_v12.0"

echo.
echo Creating executable with PyInstaller...
pyinstaller --onefile ^
    --name V7P3R_v12.0 ^
    --distpath dist ^
    --workpath build ^
    --specpath . ^
    --add-data "src\v7p3r_enhanced_nudges.json;." ^
    --hidden-import chess ^
    --hidden-import chess.engine ^
    --hidden-import chess.pgn ^
    --hidden-import json ^
    --hidden-import time ^
    --hidden-import sys ^
    --hidden-import threading ^
    --hidden-import random ^
    --hidden-import math ^
    src\v7p3r_uci.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo ==========================================
echo V7P3R v12.0 build completed!
echo Executable: dist\V7P3R_v12.0.exe
echo.
echo Testing executable...
echo.

REM Test the executable
if exist "dist\V7P3R_v12.0.exe" (
    echo Running quick UCI test...
    echo quit | "dist\V7P3R_v12.0.exe" > build\test_output.txt
    if %ERRORLEVEL% EQU 0 (
        echo SUCCESS: Executable runs correctly!
    ) else (
        echo WARNING: Executable may have issues
    )
) else (
    echo ERROR: Executable not found!
    exit /b 1
)

echo.
echo ==========================================
echo V7P3R v12.0 RELEASE BUILD COMPLETE
echo ==========================================
echo.
echo V12.0 FOUNDATION FEATURES:
echo - Clean v10.8 Stable Baseline (19.5/30 tournament recovery)
echo - Enhanced Nudge System (2160 positions with tactical metadata)
echo - Improved Time Management (v11 lessons learned)
echo - Code Cleanup (removed experimental features)
echo - Standalone Executable (embedded nudge database)
echo - PyInstaller Resource Support (dev + production modes)
echo.
echo Next steps:
echo 1. Copy to engine-tester: cp dist\V7P3R_v12.0.exe engine-tester\engines\V7P3R\
echo 2. Run 500-puzzle tactical validation vs v10.8 baseline
echo 3. Measure incremental improvement from enhanced nudges
echo 4. Establish v12.0 as clean development foundation
echo.
pause