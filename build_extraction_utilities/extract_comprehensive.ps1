# Beta Candidate Extraction Script - Comprehensive Version
# This script extracts ALL essential files from each beta-candidate tag

Write-Host "Starting COMPREHENSIVE extraction of beta candidates..." -ForegroundColor Green

# Define file patterns to include
$includePatterns = @("*.py", "*.json", "*.yaml", "*.yml", "*.md", "*.db", "*.txt", "stockfish.exe")
$excludePatterns = @("__pycache__/*", "*.pyc", "*.log")

# Beta candidates data
$betaCandidates = @(
    @{Tag="beta-candidate-0"; Date="2025-07-15"; Desc="capture_escape_functionality"},
    @{Tag="beta-candidate-1"; Date="2025-07-03"; Desc=""},
    @{Tag="beta-candidate-2"; Date="2025-07-01"; Desc=""},
    @{Tag="beta-candidate-3"; Date="2025-06-27"; Desc=""},
    @{Tag="beta-candidate-4"; Date="2025-06-09"; Desc=""},
    @{Tag="beta-candidate-5"; Date="2025-05-31"; Desc=""},
    @{Tag="beta-candidate-6"; Date="2025-05-30"; Desc=""},
    @{Tag="beta-candidate-7"; Date="2025-06-04"; Desc=""},
    @{Tag="beta-candidate-8"; Date="2025-07-14"; Desc="new_updates_properly_functioning_sims"},
    @{Tag="beta-candidate-9"; Date="2025-07-07"; Desc=""},
    @{Tag="beta-candidate-10"; Date="2025-06-30"; Desc=""},
    @{Tag="beta-candidate-11"; Date="2025-06-09"; Desc=""},
    @{Tag="beta-candidate-12"; Date="2025-06-07"; Desc=""},
    @{Tag="beta-candidate-13"; Date="2025-06-05"; Desc=""},
    @{Tag="beta-candidate-14"; Date="2025-06-04"; Desc=""},
    @{Tag="beta-candidate-15"; Date="2025-06-02"; Desc=""},
    @{Tag="beta-candidate-16"; Date="2025-06-01"; Desc="functioning_ai_v_alt_ai_engine"}
)

foreach ($candidate in $betaCandidates) {
    $tag = $candidate.Tag
    $dateStr = $candidate.Date
    $desc = $candidate.Desc
    
    # Create version directory name
    $date = [DateTime]::Parse($dateStr)
    $month = $date.Month
    $day = $date.Day
    
    if ($desc -ne "") {
        $buildDirName = "v0.$month.$day" + "_$desc"
    } else {
        $buildDirName = "v0.$month.$day" + "_$tag"
    }
    
    $buildPath = "builds\$buildDirName"
    Write-Host "`nProcessing $tag -> $buildDirName" -ForegroundColor Yellow
    
    # Create build directory
    if (!(Test-Path $buildPath)) {
        New-Item -Path $buildPath -ItemType Directory -Force | Out-Null
    }
    
    # Get all files from this tag
    $allFiles = git show $tag --name-only --format="" | Where-Object { $_ -ne "" }
    Write-Host "  Found $($allFiles.Count) total files" -ForegroundColor Cyan
    
    $extracted = 0
    $skipped = 0
    
    foreach ($file in $allFiles) {
        $shouldInclude = $false
        $shouldExclude = $false
        
        # Check include patterns
        foreach ($pattern in $includePatterns) {
            if ($file -like $pattern) {
                $shouldInclude = $true
                break
            }
        }
        
        # Check exclude patterns
        if ($shouldInclude) {
            foreach ($pattern in $excludePatterns) {
                if ($file -like $pattern) {
                    $shouldExclude = $true
                    break
                }
            }
        }
        
        if ($shouldInclude -and -not $shouldExclude) {
            # Extract this file
            $targetPath = Join-Path $buildPath $file
            $targetDir = Split-Path $targetPath -Parent
            
            # Create directory structure
            if ($targetDir -and !(Test-Path $targetDir)) {
                New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
            }
            
            # Extract file from git
            try {
                git show "$tag`:$file" | Out-File -FilePath $targetPath -Encoding UTF8 -ErrorAction Stop
                Write-Host "  ✓ $file" -ForegroundColor Green
                $extracted++
            } catch {
                Write-Host "  ✗ Failed: $file" -ForegroundColor Red
            }
        } else {
            $skipped++
        }
    }
    
    # Create build info
    $buildInfo = @{
        tag = $tag
        date = $dateStr
        description = $desc
        extractedFiles = $extracted
        skippedFiles = $skipped
        totalFiles = $allFiles.Count
        extractedOn = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    }
    
    $buildInfo | ConvertTo-Json | Out-File -FilePath (Join-Path $buildPath "BUILD_INFO.json") -Encoding UTF8
    
    Write-Host "  ✓ Build complete: $extracted extracted, $skipped skipped" -ForegroundColor Green
}

Write-Host "`n🎉 All beta candidates extracted!" -ForegroundColor Green
Write-Host "📁 Check builds\ directory for results" -ForegroundColor Cyan
