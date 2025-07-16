@echo off
REM Run V7P3R Chess Engine in the background with default settings
echo Starting V7P3R Chess Engine in background...
start /b python play_chess.py --background > engine_output.log 2>&1
echo Engine started! Check engine_output.log for progress.
