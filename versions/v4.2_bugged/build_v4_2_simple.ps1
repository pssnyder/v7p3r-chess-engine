<#
PowerShell build script for V7P3R v4.2
Creates a single-file executable for tournament use
#>

param(
    [string]$Version = "v4.2"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$workDir = Join-Path $repoRoot "build\$Version"
$distDir = Join-Path $repoRoot "dist"

Write-Output "V7P3R Chess Engine v4.2 Build Script"
Write-Output "====================================="
Write-Output "Repository root: $repoRoot"
Write-Output "Work dir: $workDir"
Write-Output "Dist dir: $distDir"

# Prepare directories
if (Test-Path $workDir) {
    Write-Output "Removing previous work dir: $workDir"
    Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

if (-not (Test-Path $distDir)) {
    New-Item -ItemType Directory -Path $distDir -Force | Out-Null
}

# Copy source files
Write-Output "Copying src/ into work dir..."
$srcPath = Join-Path $repoRoot "src\*"
if (-not (Test-Path (Join-Path $repoRoot 'src'))) {
    Write-Error "Source folder not found at $repoRoot\src"
    exit 1
}
Copy-Item -Path $srcPath -Destination $workDir -Recurse -Force

# Copy config file
$configPath = Join-Path $repoRoot "config_default.json"
if (Test-Path $configPath) {
    Copy-Item -Path $configPath -Destination $workDir -Force
    Write-Output "Copied config_default.json to work dir"
}

# Change to work dir
Push-Location $workDir

# Check Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python executable not found in PATH"
    Pop-Location
    exit 1
}

Write-Output "Using Python: $($python.Source)"

# Install/check PyInstaller
Write-Output "Checking PyInstaller..."
try {
    & python -c "import PyInstaller" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Output "PyInstaller is available"
    }
    else {
        Write-Output "Installing PyInstaller..."
        & python -m pip install pyinstaller
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to install PyInstaller"
            Pop-Location
            exit 1
        }
    }
}
catch {
    Write-Output "Installing PyInstaller..."
    & python -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller"
        Pop-Location
        exit 1
    }
}

# Build executable
$exeName = "V7P3R_v4.2"
Write-Output "Building $exeName.exe..."
Write-Output "This may take a few minutes..."

& python -m PyInstaller --onefile --noconfirm --clean --name $exeName --add-data "config_default.json;." v7p3r_uci.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed"
    Pop-Location
    exit 1
}

# Copy to dist
$builtExe = Join-Path $workDir "dist\$exeName.exe"
if (Test-Path $builtExe) {
    $targetExe = Join-Path $distDir "$exeName.exe"
    Copy-Item -Path $builtExe -Destination $targetExe -Force
    
    $fileInfo = Get-Item $targetExe
    $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    
    Write-Output "Build completed successfully!"
    Write-Output "Executable: $targetExe"
    Write-Output "File size: $fileSizeMB MB"
}
else {
    Write-Error "Built exe not found at $builtExe"
    Pop-Location
    exit 1
}

Pop-Location
Write-Output "V7P3R v4.2 build complete - ready for Arena testing!"
