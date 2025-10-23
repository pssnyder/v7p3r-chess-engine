@echo off
echo Testing V7P3R v13.0 UCI Interface...
echo.

cd /d "%~dp0"

REM Test basic UCI commands
(
echo uci
echo isready
echo position startpos
echo go depth 3
echo quit
) | "C:\Users\patss\AppData\Local\Programs\Python\Python313\python.exe" src/v7p3r_uci.py

echo.
echo Test completed. If you see engine responses above, the bat file is working correctly.
pause