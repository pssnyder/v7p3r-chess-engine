# Complete extraction for all beta candidates
Write-Host "🔄 EXTRACTING COMPLETE REPOSITORY STATES FOR ALL BETA CANDIDATES" -ForegroundColor Green

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

$baseDir = "builds_complete"
if (!(Test-Path $baseDir)) { New-Item -Path $baseDir -ItemType Directory -Force | Out-Null }

$successCount = 0

foreach ($candidate in $betaCandidates) {
    $tag = $candidate.Tag
    $dateStr = $candidate.Date
    
    try {
        $date = [DateTime]::Parse($dateStr)
        $versionName = "v0.$($date.Month).$($date.Day)_$tag" + "_COMPLETE"
        $targetDir = Join-Path $baseDir $versionName
        
        Write-Host "`n📦 Extracting $tag -> $versionName" -ForegroundColor Cyan
        
        if (Test-Path $targetDir) { Remove-Item $targetDir -Recurse -Force }
        New-Item -Path $targetDir -ItemType Directory -Force | Out-Null
        
        $allFiles = git ls-tree -r --name-only $tag | Where-Object { 
            $_ -match '\.(py|json|yaml|yml|md|db|txt)$' -and 
            $_ -notmatch '__pycache__|\.git|node_modules|\.vscode'
        }
        
        Write-Host "  Extracting $($allFiles.Count) files..." -ForegroundColor Gray
        
        $extracted = 0
        foreach ($file in $allFiles) {
            try {
                $content = git show "$tag`:$file" 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $targetPath = Join-Path $targetDir $file
                    $dir = Split-Path $targetPath -Parent
                    if ($dir -and !(Test-Path $dir)) { 
                        New-Item -Path $dir -ItemType Directory -Force | Out-Null 
                    }
                    $content | Out-File -FilePath $targetPath -Encoding UTF8
                    $extracted++
                }
            } catch {
                # Silent fail for individual files
            }
        }
        
        # Create info file
        $info = @{
            tag = $tag
            date = $dateStr
            versionName = $versionName
            filesExtracted = $extracted
            extractionTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        }
        $info | ConvertTo-Json | Out-File (Join-Path $targetDir "EXTRACTION_INFO.json") -Encoding UTF8
        
        Write-Host "  ✅ Extracted $extracted files" -ForegroundColor Green
        $successCount++
        
    } catch {
        Write-Host "  ❌ Failed: $_" -ForegroundColor Red
    }
}

Write-Host "`n🎉 EXTRACTION COMPLETE!" -ForegroundColor Green
Write-Host "✅ Successfully processed $successCount/$($betaCandidates.Count) beta candidates" -ForegroundColor Green
Write-Host "📁 Complete builds available in: $baseDir\" -ForegroundColor Cyan
