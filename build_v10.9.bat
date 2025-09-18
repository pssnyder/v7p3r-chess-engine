@echo off
REM V7P3R Chess Engine v10.9 Build Script
REM Critical Perspective Bug Fix Release

echo Building V7P3R Chess Engine v10.9...
echo =======================================

REM Clean previous builds
if exist "dist\V7P3R_v10.9.exe" del "dist\V7P3R_v10.9.exe"
if exist "build\V7P3R_v10.9" rmdir /s /q "build\V7P3R_v10.9"

echo.
echo Creating executable with PyInstaller...
pyinstaller --onefile ^
    --name V7P3R_v10.9 ^
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
    src\v7p3r.py

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Build failed!
    pause
    exit /b 1
)

echo.
echo =======================================
echo V7P3R v10.9 build completed!
echo Executable: dist\V7P3R_v10.9.exe
echo.
echo Testing executable...
echo.

REM Test the executable
if exist "dist\V7P3R_v10.9.exe" (
    echo Running quick UCI test...
    echo quit | "dist\V7P3R_v10.9.exe" > build\test_output.txt
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
echo =======================================
echo V7P3R v10.9 RELEASE BUILD COMPLETE
echo =======================================
echo.
echo Next steps:
echo 1. Test executable with UCI interface
echo 2. Commit to git as v10.9 stable point
echo 3. Begin v11 depth optimization work
echo.
pause