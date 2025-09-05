@echo off
REM V7P3R v10.2 Build Script - Windows Version
REM Builds the V7P3R_v10.2.exe engine with PV Following functionality

echo 🚀 Building V7P3R v10.2 Chess Engine
echo ======================================

REM Navigate to project directory
cd /d "s:\Maker Stuff\Programming\Chess Engines\V7P3R Chess Engine\v7p3r-chess-engine"

REM Clean previous build artifacts
echo 🧹 Cleaning previous build artifacts...
if exist "build\V7P3R_v10.2_RELEASE\*" (
    rmdir /s /q "build\V7P3R_v10.2_RELEASE"
    mkdir "build\V7P3R_v10.2_RELEASE"
)
if exist "dist\V7P3R_v10.2.exe" del "dist\V7P3R_v10.2.exe"

REM Run PyInstaller with the v10.2 spec file
echo 🔨 Building V7P3R_v10.2.exe...
pyinstaller V7P3R_v10.2_RELEASE.spec --distpath dist --workpath build\V7P3R_v10.2_RELEASE

REM Check if build was successful
if exist "dist\V7P3R_v10.2.exe" (
    echo ✅ BUILD SUCCESS!
    echo 📁 Executable created: dist\V7P3R_v10.2.exe
    
    REM Test the executable
    echo 🧪 Testing executable...
    echo uci | "dist\V7P3R_v10.2.exe"
    
    echo.
    echo 🎉 V7P3R v10.2 BUILD COMPLETE!
    echo 💫 Features included:
    echo    - Enhanced PV Following System
    echo    - Instant move recognition using board states
    echo    - Clean UCI output
    echo    - Optimized search and evaluation
    echo    - Ready for Arena GUI and tournaments
    
) else (
    echo ❌ BUILD FAILED!
    echo Check the build log above for errors.
    pause
    exit /b 1
)

pause
