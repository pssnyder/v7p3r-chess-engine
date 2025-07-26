# Simple Beta Candidate Extraction
$candidates = @(
    @{Tag="beta-candidate-0"; Date="2025-07-15"},
    @{Tag="beta-candidate-1"; Date="2025-07-03"},
    @{Tag="beta-candidate-2"; Date="2025-07-01"},
    @{Tag="beta-candidate-3"; Date="2025-06-27"},
    @{Tag="beta-candidate-4"; Date="2025-06-09"},
    @{Tag="beta-candidate-5"; Date="2025-05-31"},
    @{Tag="beta-candidate-6"; Date="2025-05-30"},
    @{Tag="beta-candidate-7"; Date="2025-06-04"},
    @{Tag="beta-candidate-8"; Date="2025-07-14"},
    @{Tag="beta-candidate-9"; Date="2025-07-07"},
    @{Tag="beta-candidate-10"; Date="2025-06-30"},
    @{Tag="beta-candidate-11"; Date="2025-06-09"},
    @{Tag="beta-candidate-12"; Date="2025-06-07"},
    @{Tag="beta-candidate-13"; Date="2025-06-05"},
    @{Tag="beta-candidate-14"; Date="2025-06-04"},
    @{Tag="beta-candidate-15"; Date="2025-06-02"},
    @{Tag="beta-candidate-16"; Date="2025-06-01"}
)

foreach ($c in $candidates) {
    $date = [DateTime]::Parse($c.Date)
    $buildDir = "builds\v0.$($date.Month).$($date.Day)_$($c.Tag)"
    Write-Host "Processing $($c.Tag) -> $buildDir"
    
    if (!(Test-Path $buildDir)) { New-Item $buildDir -Type Directory -Force | Out-Null }
    
    $files = git show $($c.Tag) --name-only --format=""
    foreach ($file in $files) {
        if ($file -and ($file -like "*.py" -or $file -like "*.json" -or $file -like "*.yaml" -or $file -like "*.yml" -or $file -like "*.md" -or $file -like "*.db" -or $file -like "*.txt")) {
            $target = Join-Path $buildDir $file
            $dir = Split-Path $target -Parent
            if ($dir -and !(Test-Path $dir)) { New-Item $dir -Type Directory -Force | Out-Null }
            git show "$($c.Tag):$file" | Out-File $target -Encoding UTF8
            Write-Host "  Extracted: $file"
        }
    }
}
Write-Host "Extraction complete!"
