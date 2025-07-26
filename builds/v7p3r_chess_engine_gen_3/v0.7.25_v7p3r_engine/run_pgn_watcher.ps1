# run_pgn_watcher.ps1
# Script to launch the PGN Watcher for visualizing active chess games

Write-Host "Starting V7P3R PGN Watcher for visual game display..."
Write-Host "This will display the current active game visually without impacting engine performance."
Write-Host "Press CTRL+C to exit the watcher when done."
Write-Host ""

# Run the PGN watcher
python active_game_watcher.py
