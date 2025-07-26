# Run V7P3R Chess Engine in the background with default settings
Write-Host "Starting V7P3R Chess Engine in background..."
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "play_chess.py", "--background" -RedirectStandardOutput "engine_output.log" -RedirectStandardError "engine_error.log"
Write-Host "Engine started! Check engine_output.log for progress."
