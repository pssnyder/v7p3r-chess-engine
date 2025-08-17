<#
PowerShell build script for V7P3R v4.1
- Copies the `src/` tree into a per-build working folder under `build\v4.1`
- Installs PyInstaller if needed, runs PyInstaller to produce a single-file exe
- Places the final exe into the repository `dist\` folder as `V7P3R_v4.1.exe`

Notes:
- Run from the repository root where this script lives.
- This script is intended for Windows PowerShell v5.1 (no use of '&&').
#>

param(
    [string]$Version = "v4.1"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
# repo root is the parent of the build folder where this script lives
$repoRoot = Split-Path -Parent $scriptDir
$workDir = Join-Path $repoRoot "build\$Version"
$distDir = Join-Path $repoRoot "dist"

Write-Output "Repository root: $root"
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

# Change to work dir for the build
Push-Location $workDir

# Ensure Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Error "Python executable not found in PATH. Install Python and retry."
    Pop-Location
    exit 1
}

# Ensure PyInstaller is installed (try importing, otherwise install)
Write-Output "Ensuring PyInstaller is installed..."
try {
    & python -c "import PyInstaller"; $pyOk = $true
} catch {
    Write-Output "PyInstaller not found; installing via pip..."
    & python -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller. Please install it manually and re-run this script."
        Pop-Location
        exit $LASTEXITCODE
    }
}

# Run PyInstaller to build a single-file exe
$exeName = "V7P3R_v4.1"
Write-Output "Running PyInstaller to build $exeName.exe..."
# Use --noconfirm to overwrite previous PyInstaller artifacts in the work dir
& python -m PyInstaller --onefile --noconfirm --name $exeName v7p3r_uci.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller build failed with exit code $LASTEXITCODE"
    Pop-Location
    exit $LASTEXITCODE
}

# Copy built exe to repo dist folder
$builtExe = Join-Path $workDir "dist\$exeName.exe"
if (Test-Path $builtExe) {
    Copy-Item -Path $builtExe -Destination (Join-Path $distDir "$exeName.exe") -Force
    Write-Output "Built exe copied to $distDir\$exeName.exe"
} else {
    Write-Error "Built exe not found at $builtExe"
    Pop-Location
    exit 1
}

Pop-Location
Write-Output "Build completed."
