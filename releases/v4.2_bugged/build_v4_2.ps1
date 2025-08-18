<#
PowerShell build script for V7P3R v4.2
- Copies the `src/` tree into a per-build working folder under `build\v4.2`
- Installs PyInstaller if needed, runs PyInstaller to produce a single-file exe
- Places the final exe into the repository `dist\` folder as `V7P3R_v4.2.exe`

Notes:
- Run from the repository root where this script lives.
- This script is intended for Windows PowerShell v5.1 (no use of '&&').
- This version includes the v4.2 performance optimizations
#>

param(
    [string]$Version = "v4.2"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# repo root is the parent of the build folder where this script lives
$repoRoot = Split-Path -Parent $scriptDir
$workDir = Join-Path $repoRoot "build\$Version"
$distDir = Join-Path $repoRoot "dist"

Write-Output "V7P3R Chess Engine v4.2 Build Script"
Write-Output "=================================="
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

# Copy source files into the working dir (source is at repo root)
Write-Output "Copying src/ into work dir..."
$srcPath = Join-Path $repoRoot "src\*"
if (-not (Test-Path (Join-Path $repoRoot 'src'))) {
    Write-Error "Source folder not found at $repoRoot\src - ensure your src/ folder is at the repository root"
    Pop-Location
    exit 1
}
Copy-Item -Path $srcPath -Destination $workDir -Recurse -Force

# Also copy the default config file
$configPath = Join-Path $repoRoot "config_default.json"
if (Test-Path $configPath) {
    Copy-Item -Path $configPath -Destination $workDir -Force
    Write-Output "Copied config_default.json to work dir"
}

# Change to work dir for the build
Push-Location $workDir

# Ensure Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python executable not found in PATH. Install Python and retry."
    Pop-Location
    exit 1
}

Write-Output "Using Python: $($python.Source)"

# Check Python version
$pyVersion = & python --version 2>&1
Write-Output "Python version: $pyVersion"

# Ensure PyInstaller is installed (try importing, otherwise install)
Write-Output "Ensuring PyInstaller is installed..."
try {
    & python -c "import PyInstaller" 2>&1 | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Output "PyInstaller is already installed"
    } else {
        throw "PyInstaller not found"
    }
} catch {
    Write-Output "PyInstaller not found; installing via pip..."
    & python -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller. Please install it manually and re-run this script."
        Pop-Location
        exit $LASTEXITCODE
    }
}

# Check for required dependencies
Write-Output "Checking dependencies..."
$requiredModules = @("chess", "pathlib")
foreach ($module in $requiredModules) {
    try {
        & python -c "import $module" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Output "✓ $module is available"
        } else {
            throw "Module $module not found"
        }
    } catch {
        Write-Warning "Required module '$module' may not be installed. Install with: pip install $module"
    }
}
}

# Run PyInstaller to build a single-file exe
$exeName = "V7P3R_v4.2"
Write-Output "Running PyInstaller to build $exeName.exe..."
Write-Output "This may take a few minutes..."

# Use --noconfirm to overwrite previous PyInstaller artifacts in the work dir
# Add --clean to ensure a fresh build
& python -m PyInstaller --onefile --noconfirm --clean --name $exeName v7p3r_uci.py

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE"
    Pop-Location
    exit $LASTEXITCODE
}

# Copy built exe to repo dist folder
$builtExe = Join-Path $workDir "dist\$exeName.exe"
if (Test-Path $builtExe) {
    $targetExe = Join-Path $distDir "$exeName.exe"
    Copy-Item -Path $builtExe -Destination $targetExe -Force
    
    # Get file size for verification
    $fileInfo = Get-Item $targetExe
    $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)
    
    Write-Output "✓ Built exe copied to $targetExe"
    Write-Output "✓ File size: $fileSizeMB MB"
    Write-Output "✓ Build completed successfully!"
    
    # Test the exe quickly
    Write-Output ""
    Write-Output "Testing executable..."
    try {
        $testOutput = & $targetExe "uci" 2>&1 | Select-Object -First 5
        if ($testOutput -match "id name") {
            Write-Output "✓ Executable responds to UCI commands"
        } else {
            Write-Warning "Executable may have issues - test output: $testOutput"
        }
    } catch {
        Write-Warning "Could not test executable: $_"
    }
    
} else {
    Write-Error "Built exe not found at $builtExe"
    Write-Output "Checking build directory contents:"
    Get-ChildItem $workDir -Recurse | Where-Object {$_.Name -like "*.exe"} | ForEach-Object {
        Write-Output "Found: $($_.FullName)"
    }
    Pop-Location
    exit 1
}

Pop-Location
Write-Output ""
Write-Output "V7P3R v4.2 build completed successfully!"
Write-Output "Executable location: $distDir\$exeName.exe"
Write-Output "Ready for Arena testing!""
