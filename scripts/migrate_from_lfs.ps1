# Script to migrate files out of Git LFS
Write-Host "Starting migration of files out of Git LFS..." -ForegroundColor Green

# Get current status
Write-Host "Current Git status before migration:" -ForegroundColor Yellow
git status --short

# Backup current changes
Write-Host "Stashing current changes..." -ForegroundColor Yellow
git stash push -m "Backup before LFS migration"

# List all files currently tracked by LFS
Write-Host "Files currently tracked by LFS:" -ForegroundColor Yellow
git lfs ls-files > lfs_files_before.txt
Get-Content lfs_files_before.txt

# Migrate files out of LFS for specific file types
Write-Host "Migrating files out of LFS..." -ForegroundColor Green

# The migrate export command moves files from LFS back to regular Git
git lfs migrate export --include="*.pgn,*.log,*.yaml,*.yml,*.json,*.png" --everything

# Check what's still in LFS
Write-Host "Files still in LFS after migration:" -ForegroundColor Yellow
git lfs ls-files > lfs_files_after.txt
Get-Content lfs_files_after.txt

# Show the difference
Write-Host "Comparison of LFS files before and after:" -ForegroundColor Cyan
Write-Host "Before migration:" -ForegroundColor Red
Get-Content lfs_files_before.txt | Measure-Object | Select-Object Count
Write-Host "After migration:" -ForegroundColor Green
Get-Content lfs_files_after.txt | Measure-Object | Select-Object Count

Write-Host "Migration complete!" -ForegroundColor Green
