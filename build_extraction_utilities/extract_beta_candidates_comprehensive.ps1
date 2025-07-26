# Beta Candidate Extraction Script - Comprehensive Version
# This script extracts ALL essential files from each beta-candidate tag into organized build directories
# Goal: Create completely self-contained builds for each beta candidate

# Define comprehensive file patterns to extract for complete self-contained builds
$coreFilePatterns = @(
    "*.py",     # All Python code
    "*.json",   # All JSON configs and data
    "*.yaml",   # All YAML configs
    "*.yml",    # All YML configs 
    "*.md",     # All documentation
    "*.db",     # All database files
    "*.txt",    # Text files (requirements, etc.)
    "stockfish.exe"  # Chess engine executable
)

# Minimal exclusions - only temp/cache files that definitely don't belong in builds
$excludePatterns = @(
    "__pycache__/*",
    "*.pyc",
    "*.log"
)

# Beta candidates with their info
$betaCandidates = @(
    @{Tag="beta-candidate-16"; Date="2025-06-01"; Desc="functioning_ai_v_alt_ai_engine"},
    @{Tag="beta-candidate-15"; Date="2025-06-02"; Desc=""},
    @{Tag="beta-candidate-14"; Date="2025-06-04"; Desc=""},
    @{Tag="beta-candidate-13"; Date="2025-06-05"; Desc=""},
    @{Tag="beta-candidate-12"; Date="2025-06-07"; Desc=""},
    @{Tag="beta-candidate-11"; Date="2025-06-09"; Desc=""},
    @{Tag="beta-candidate-7"; Date="2025-06-04"; Desc=""},
    @{Tag="beta-candidate-6"; Date="2025-05-30"; Desc=""},
    @{Tag="beta-candidate-5"; Date="2025-05-31"; Desc=""},
    @{Tag="beta-candidate-4"; Date="2025-06-09"; Desc=""},
    @{Tag="beta-candidate-3"; Date="2025-06-27"; Desc=""},
    @{Tag="beta-candidate-10"; Date="2025-06-30"; Desc=""},
    @{Tag="beta-candidate-2"; Date="2025-07-01"; Desc=""},
    @{Tag="beta-candidate-1"; Date="2025-07-03"; Desc=""},
    @{Tag="beta-candidate-9"; Date="2025-07-07"; Desc=""},
    @{Tag="beta-candidate-8"; Date="2025-07-14"; Desc="new_updates_properly_functioning_sims"},
    @{Tag="beta-candidate-0"; Date="2025-07-15"; Desc="capture_escape_functionality"}
)

Write-Host "Starting COMPREHENSIVE extraction of $($betaCandidates.Count) beta candidates..." -ForegroundColor Green
Write-Host "This will include ALL .py, .json, .yaml, .md, and .db files for complete self-contained builds" -ForegroundColor Cyan

foreach ($candidate in $betaCandidates) {
    $tag = $candidate.Tag
    $dateStr = $candidate.Date
    $desc = $candidate.Desc
    
    # Parse date and create version string
    $date = [DateTime]::Parse($dateStr)
    $month = $date.Month
    $day = $date.Day
    
    # Create build directory name
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
    
    # Get list of files at this tag
    $files = git show $tag --name-only --format="" | Where-Object { $_ -ne "" }
    
    Write-Host "  Found $($files.Count) files in $tag" -ForegroundColor Cyan
    $extractedCount = 0
    $skippedCount = 0
    
    foreach ($file in $files) {
        # Check if file matches core patterns and doesn't match exclude patterns
        $shouldInclude = $false
        foreach ($pattern in $coreFilePatterns) {
            if ($file -like $pattern) {
                $shouldInclude = $true
                break
            }
        }
        
        # If it's not in our include patterns, check if it's a directory or other important file
        if (-not $shouldInclude) {
            # Include any file that doesn't have an extension (might be important)
            if (-not $file.Contains('.')) {
                $shouldInclude = $true
            }
            # Include any config-related files
            elseif ($file -match 'config|setup|requirements|install') {
                $shouldInclude = $true
            }
        }
        
        if ($shouldInclude) {
            $shouldExclude = $false
            foreach ($excludePattern in $excludePatterns) {
                if ($file -like $excludePattern) {
                    $shouldExclude = $true
                    break
                }
            }
            
            if (!$shouldExclude) {
                # Extract the file from git
                $targetPath = Join-Path $buildPath $file
                $targetDir = Split-Path $targetPath -Parent
                
                # Create directory if it doesn't exist
                if ($targetDir -and !(Test-Path $targetDir)) {
                    New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
                }
                
                # Extract file content from git
                try {
                    $gitOutput = git show "$tag`:$file" 2>&1
                    if ($LASTEXITCODE -eq 0) {
                        $gitOutput | Out-File -FilePath $targetPath -Encoding UTF8
                        Write-Host "  ✓ Extracted: $file" -ForegroundColor Green
                        $extractedCount++
                    } else {
                        Write-Host "  ✗ Git error for: $file" -ForegroundColor Red
                    }
                } catch {
                    Write-Host "  ✗ Failed to extract: $file - $($_.Exception.Message)" -ForegroundColor Red
                }
            } else {
                Write-Host "  - Skipped (excluded): $file" -ForegroundColor DarkGray
                $skippedCount++
            }
        } else {
            Write-Host "  - Skipped (not included): $file" -ForegroundColor DarkGray
            $skippedCount++
        }
    }
    
    # Create a build info file
    $buildInfo = @{
        tag = $tag
        date = $dateStr
        description = $desc
        extractedOn = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        filesExtracted = $extractedCount
        filesSkipped = $skippedCount
        totalFiles = $files.Count
        coreFiles = (Get-ChildItem -Path $buildPath -Recurse -File | Where-Object { $_.Name -ne "BUILD_INFO.json" }).Name
    } | ConvertTo-Json -Depth 3
    
    $buildInfo | Out-File -FilePath (Join-Path $buildPath "BUILD_INFO.json") -Encoding UTF8
    
    Write-Host "  ✓ Created build: $buildDirName ($extractedCount extracted, $skippedCount skipped)" -ForegroundColor Green
}

Write-Host "`n🎉 Extraction complete! Created builds for $($betaCandidates.Count) beta candidates." -ForegroundColor Green
Write-Host "📁 Build directories are in: builds\" -ForegroundColor Cyan
Write-Host "📊 Each build contains a BUILD_INFO.json with extraction details" -ForegroundColor Cyan
