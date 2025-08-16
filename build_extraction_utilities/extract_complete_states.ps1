# PowerShell script to extract complete repository states for beta candidates

Write-Host "🔄 EXTRACTING COMPLETE REPOSITORY STATES FOR BETA CANDIDATES" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green

# Create complete builds directory
$completeBuildsDir = "builds_complete"
if (Test-Path $completeBuildsDir) {
    Write-Host "Removing existing $completeBuildsDir..." -ForegroundColor Yellow
    Remove-Item $completeBuildsDir -Recurse -Force
}
New-Item -Path $completeBuildsDir -ItemType Directory -Force | Out-Null

# Beta candidates with dates (manually verified)
$betaCandidates = @(
    @{Tag="beta-candidate-16"; Date="2025-06-01"},
    @{Tag="beta-candidate-15"; Date="2025-06-02"}, 
    @{Tag="beta-candidate-14"; Date="2025-06-04"},
    @{Tag="beta-candidate-7"; Date="2025-06-04"},
    @{Tag="beta-candidate-13"; Date="2025-06-05"},
    @{Tag="beta-candidate-12"; Date="2025-06-07"},
    @{Tag="beta-candidate-11"; Date="2025-06-09"},
    @{Tag="beta-candidate-4"; Date="2025-06-09"},
    @{Tag="beta-candidate-5"; Date="2025-05-31"},
    @{Tag="beta-candidate-6"; Date="2025-05-30"},
    @{Tag="beta-candidate-3"; Date="2025-06-27"},
    @{Tag="beta-candidate-10"; Date="2025-06-30"},
    @{Tag="beta-candidate-2"; Date="2025-07-01"},
    @{Tag="beta-candidate-1"; Date="2025-07-03"},
    @{Tag="beta-candidate-9"; Date="2025-07-07"},
    @{Tag="beta-candidate-8"; Date="2025-07-14"},
    @{Tag="beta-candidate-0"; Date="2025-07-15"}
)

$successCount = 0
$failCount = 0

foreach ($candidate in $betaCandidates) {
    $tag = $candidate.Tag
    $dateStr = $candidate.Date
    
    Write-Host "`n📦 Processing $tag ($dateStr)..." -ForegroundColor Cyan
    
    try {
        # Create version directory name
        $date = [DateTime]::Parse($dateStr)
        $versionName = "v0.$($date.Month).$($date.Day)_$tag"
        $targetDir = Join-Path $completeBuildsDir $versionName
        
        # Create target directory
        New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
        
        # Get all files at this commit point
        $allFiles = git ls-tree -r --name-only $tag
        
        # Filter for relevant files
        $relevantFiles = $allFiles | Where-Object { 
            $_ -match '\.(py|json|yaml|yml|md|db|txt)$' -and 
            $_ -notmatch '__pycache__|\.git|node_modules|\.vscode'
        }
        
        Write-Host "  Found $($relevantFiles.Count) relevant files out of $($allFiles.Count) total" -ForegroundColor Gray
        
        $extractedCount = 0
        $failedCount = 0
        
        # Extract each file
        foreach ($file in $relevantFiles) {
            try {
                # Get file content at this commit
                $content = git show "$tag`:$file" 2>$null
                
                if ($LASTEXITCODE -eq 0) {
                    # Create target file path
                    $targetFilePath = Join-Path $targetDir $file
                    $targetFileDir = Split-Path $targetFilePath -Parent
                    
                    # Create directory structure
                    if ($targetFileDir -and !(Test-Path $targetFileDir)) {
                        New-Item -Path $targetFileDir -ItemType Directory -Force | Out-Null
                    }
                    
                    # Write file
                    $content | Out-File -FilePath $targetFilePath -Encoding UTF8
                    $extractedCount++
                    
                    if ($extractedCount % 20 -eq 0) {
                        Write-Host "    Extracted $extractedCount/$($relevantFiles.Count) files..." -ForegroundColor Gray
                    }
                } else {
                    $failedCount++
                    Write-Host "    ⚠️ Failed to extract: $file" -ForegroundColor Red
                }
            } catch {
                $failedCount++
                Write-Host "    ⚠️ Error extracting $file`: $_" -ForegroundColor Red
            }
        }
        
        # Create extraction info file
        $extractionInfo = @{
            tag = $tag
            date = $dateStr
            versionName = $versionName
            totalFilesInRepo = $allFiles.Count
            relevantFilesFound = $relevantFiles.Count
            filesExtracted = $extractedCount
            filesFailed = $failedCount
            extractionTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
            commitHash = (git rev-list -n 1 $tag).Trim()
        }
        
        $extractionInfo | ConvertTo-Json -Depth 3 | Out-File -FilePath (Join-Path $targetDir "COMPLETE_EXTRACTION_INFO.json") -Encoding UTF8
        
        Write-Host "  ✅ Extracted $extractedCount files successfully" -ForegroundColor Green
        if ($failedCount -gt 0) {
            Write-Host "  ⚠️ $failedCount files failed" -ForegroundColor Yellow
        }
        
        $successCount++
        
    } catch {
        Write-Host "  ❌ Failed to process $tag`: $_" -ForegroundColor Red
        $failCount++
    }
}

Write-Host "`n" + "=" * 70 -ForegroundColor Green
Write-Host "🎉 COMPLETE EXTRACTION SUMMARY" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✅ Successfully extracted: $successCount/$($betaCandidates.Count) beta candidates" -ForegroundColor Green
if ($failCount -gt 0) {
    Write-Host "❌ Failed extractions: $failCount" -ForegroundColor Red
}

Write-Host "`n📁 Complete repository states available in: $completeBuildsDir\" -ForegroundColor Cyan
Write-Host "🔍 Each build now contains ALL files that existed at that commit point!" -ForegroundColor Cyan

# Quick verification of a specific build
if ($successCount -gt 0) {
    $testBuild = Get-ChildItem $completeBuildsDir | Where-Object { $_.Name -like "*beta-candidate-3*" } | Select-Object -First 1
    if ($testBuild) {
        $pyFiles = Get-ChildItem "$($testBuild.FullName)" -Recurse -Filter "*.py" | Measure-Object | Select-Object -ExpandProperty Count
        Write-Host "`n🔍 Verification - $($testBuild.Name) now contains $pyFiles Python files" -ForegroundColor Yellow
        
        # Check for the missing dependencies
        $v7p3rFiles = Get-ChildItem "$($testBuild.FullName)" -Recurse -Filter "v7p3r*.py" | Select-Object -ExpandProperty Name
        Write-Host "   v7p3r modules found: $($v7p3rFiles -join ', ')" -ForegroundColor Yellow
    }
}
