@echo off
echo ===============================================
echo V7P3R Chess Engine v11.3 Build Script
echo ===============================================
echo.

echo 🔨 Building V7P3R v11.3 with incremental heuristics...
echo.

REM Create build directory
if not exist "build\" mkdir build
if not exist "build\v11.3\" mkdir build\v11.3

REM Copy source files
echo 📂 Copying source files...
copy src\*.py build\v11.3\
copy requirements.txt build\v11.3\

REM Copy supporting files
copy src\nudge_database.json build\v11.3\ 2>nul

echo.
echo ✅ V7P3R v11.3 build completed!
echo 📁 Build location: build\v11.3\
echo.

echo 🎯 V11.3 Features Implemented:
echo   ✓ Draw penalty heuristic (reduce repetitive play)
echo   ✓ Enhanced endgame king evaluation (centralization, pawn support)
echo   ✓ Move classification system (offensive/defensive/developing)
echo   ✓ King restriction "closing the box" heuristic
echo   ✓ Phase-aware evaluation priorities (opening/middlegame/endgame)
echo.

echo 🧪 Ready for comprehensive testing and puzzle analysis!
pause