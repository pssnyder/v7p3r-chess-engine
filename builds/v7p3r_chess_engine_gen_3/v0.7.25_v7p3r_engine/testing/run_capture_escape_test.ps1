# run_capture_escape_test.ps1
# This script runs the capture-to-escape-check test with the specialized config

# Set working directory to the script location
$scriptPath = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
Set-Location -Path $scriptPath

# Run the test
Write-Host "Running capture-to-escape-check test..."
python test_capture_escape.py

# Pause to see results
Write-Host "`nPress any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
