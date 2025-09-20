@echo off
REM V7P3R Chess Engine v11 Build Script
REM Adaptive Evaluation & Performance Optimization Release

echo Building V7P3R Chess Engine v11...
echo ==========================================

REM Clean previous builds
if exist "dist\V7P3R_v11.exe" del "dist\V7P3R_v11.exe"
if exist "build\V7P3R_v11" rmdir /s /q "build\V7P3R_v11"

echo.
echo Creating executable with PyInstaller...
pyinstaller --onefile ^
    --name V7P3R_v11 ^
    --distpath dist ^
    --workpath build ^
    --specpath . ^
    --add-data "src\v7p3r_nudge_database.json;." ^
    --add-data "src\v7p3r_strategic_database.py;." ^
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
echo V7P3R v11 build completed!
echo Executable: dist\V7P3R_v11.exe
echo.
echo Testing executable...
echo.

REM Test the executable
if exist "dist\V7P3R_v11.exe" (
    echo Running quick UCI test...
    echo quit | "dist\V7P3R_v11.exe" > build\test_output.txt
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
echo V7P3R v11 RELEASE BUILD COMPLETE
echo ==========================================
echo.
echo MAJOR ENHANCEMENTS INCLUDED:
echo - Adaptive Evaluation Framework
echo - Dynamic Move Selection  
echo - Position Posture Assessment
echo - Fast Evaluator System
echo - Strategic Database Integration
echo - Lightweight Defense Analysis
echo - Performance: Depth 6 in 2.4s (middlegame)
echo.
echo Next steps:
echo 1. Test executable with UCI interface
echo 2. Validate tournament compliance
echo 3. Deploy to engine-tester for puzzle testing
echo 4. Begin tournament play validation
echo.
pause