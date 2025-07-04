# LFS Cleanup Helper Script
# This script helps manage Git LFS files and cleanup

param(
    [switch]$CheckSize,
    [switch]$ListLFS,
    [switch]$PruneCache,
    [switch]$ShowStats,
    [switch]$Help
)

function Show-Help {
    Write-Host "LFS Cleanup Helper Script" -ForegroundColor Green
    Write-Host "Usage: .\lfs_cleanup_helper.ps1 [OPTIONS]" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Options:" -ForegroundColor Cyan
    Write-Host "  -CheckSize    Check the size of the Git repository"
    Write-Host "  -ListLFS      List all files currently tracked by LFS"
    Write-Host "  -PruneCache   Prune LFS cache to free up space"
    Write-Host "  -ShowStats    Show detailed LFS statistics"
    Write-Host "  -Help         Show this help message"
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Magenta
    Write-Host "  .\lfs_cleanup_helper.ps1 -CheckSize"
    Write-Host "  .\lfs_cleanup_helper.ps1 -ListLFS"
    Write-Host "  .\lfs_cleanup_helper.ps1 -PruneCache"
}

function Get-RepoSize {
    Write-Host "Calculating repository size..." -ForegroundColor Yellow
    $gitSize = Get-ChildItem -Recurse .git -ErrorAction SilentlyContinue | 
               Measure-Object -Property Length -Sum | 
               ForEach-Object { $_.Sum / 1MB }
    
    Write-Host "Git repository size: $($gitSize.ToString('N2')) MB" -ForegroundColor Green
    
    if (Test-Path ".git\lfs\objects") {
        $lfsSize = Get-ChildItem -Recurse ".git\lfs\objects" -ErrorAction SilentlyContinue |
                   Measure-Object -Property Length -Sum |
                   ForEach-Object { $_.Sum / 1MB }
        Write-Host "LFS objects size: $($lfsSize.ToString('N2')) MB" -ForegroundColor Green
    }
}

function Get-LFSFiles {
    Write-Host "Files tracked by LFS:" -ForegroundColor Yellow
    git lfs ls-files
}

function Invoke-LFSPrune {
    Write-Host "Pruning LFS cache..." -ForegroundColor Yellow
    git lfs prune --force
    Write-Host "LFS cache pruned successfully!" -ForegroundColor Green
}

function Show-LFSStats {
    Write-Host "=== LFS Statistics ===" -ForegroundColor Cyan
    
    Write-Host "`nLFS Configuration:" -ForegroundColor Yellow
    git lfs env | Select-String "LocalMediaDir|TempDir|LfsStorageDir"
    
    Write-Host "`nFiles tracked by LFS:" -ForegroundColor Yellow
    $lfsFiles = git lfs ls-files
    if ($lfsFiles) {
        $lfsFiles | Measure-Object | ForEach-Object { Write-Host "Total LFS files: $($_.Count)" -ForegroundColor Green }
    } else {
        Write-Host "No files tracked by LFS" -ForegroundColor Red
    }
    
    Write-Host "`nRepository sizes:" -ForegroundColor Yellow
    Get-RepoSize
}

# Main execution
if ($Help) {
    Show-Help
    return
}

if ($CheckSize) {
    Get-RepoSize
}

if ($ListLFS) {
    Get-LFSFiles
}

if ($PruneCache) {
    Invoke-LFSPrune
}

if ($ShowStats) {
    Show-LFSStats
}

if (-not ($CheckSize -or $ListLFS -or $PruneCache -or $ShowStats -or $Help)) {
    Write-Host "No options specified. Use -Help for usage information." -ForegroundColor Red
    Show-Help
}
