@echo off
REM V7P3R Weekly Analytics - Windows Batch Script
REM Runs Docker Compose to generate weekly analytics reports
REM CONFIGURED FOR BACKGROUND EXECUTION - Won't hog your CPU

echo ========================================
echo V7P3R Weekly Analytics (Background Mode)
echo ========================================
echo.
echo Resource Limits:
echo   - Max 4 CPU cores (leaves 12 for you)
echo   - Max 4GB RAM
echo   - 4 parallel workers (safe for your system)
echo.
echo This will analyze the last 7 days of games.
echo Expected time: ~3-4 hours (slower but system-friendly)
echo.
echo Starting Docker Compose...
echo.

cd /d "%~dp0"

docker-compose up --build

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo SUCCESS: Analytics complete!
    echo ========================================
    echo.
    echo Reports saved to: analytics_reports\
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo ERROR: Analytics failed!
    echo ========================================
    echo.
    echo Check Docker Desktop is running.
    echo Review logs above for details.
    echo.
    pause
    exit /b 1
)
