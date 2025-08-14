<#
Minimal build script: build_simple_v4_1.ps1
- Builds a single-file exe directly from src\v7p3r_uci.py
- No file movement or source copying
- Places working files in repo 'build' and final exe in repo 'dist' as V7P3R_v4.1.exe

Usage (from repo root):
    powershell -NoProfile -ExecutionPolicy Bypass -File .\build_simple_v4_1.ps1
#>

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
# If script is invoked from build folder, repoRoot will be build folder; get parent
if ((Split-Path -Leaf $repoRoot) -ieq 'build') {
    $repoRoot = Split-Path -Parent $repoRoot
}

$entry = Join-Path $repoRoot "src\v7p3r_uci.py"
$distDir = Join-Path $repoRoot "dist"
$workDir = Join-Path $repoRoot "build"

Write-Output "Repo root: $repoRoot"
Write-Output "Entry: $entry"

if (-not (Test-Path $entry)) {
    Write-Error "Entry point not found: $entry"
    exit 1
}

if (-not (Test-Path $distDir)) { New-Item -ItemType Directory -Path $distDir | Out-Null }
if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Path $workDir | Out-Null }

# Ensure Python available
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) {
    Write-Error "Python not found on PATH. Install Python and retry."
    exit 1
}

# Ensure PyInstaller is installed
try {
    & python -c "import PyInstaller"; $ok = $true
} catch {
    Write-Output "PyInstaller not found; installing via pip..."
    & python -m pip install --upgrade pyinstaller
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install PyInstaller"
        exit $LASTEXITCODE
    }
}

# Run PyInstaller directly on the src entrypoint
$exeName = "V7P3R_v4.1"
Write-Output "Building $exeName from $entry..."
# Include essential data files so the frozen exe can find config and opening book data
$configPath = Join-Path $repoRoot "src\config_default.json"
$pgnDir = Join-Path $repoRoot "src\pgn_opening_data"

if (Test-Path $configPath) {
    $addData1 = "$configPath;."
} else {
    Write-Warning "config_default.json not found at $configPath - build may fail at runtime"
    $addData1 = ""
}

if (Test-Path $pgnDir) {
    $addData2 = "$pgnDir;pgn_opening_data"
} else {
    Write-Warning "pgn_opening_data folder not found at $pgnDir - opening book won't be included"
    $addData2 = ""
}

$addArgs = @()
if ($addData1 -ne "") { $addArgs += "--add-data"; $addArgs += $addData1 }
if ($addData2 -ne "") { $addArgs += "--add-data"; $addArgs += $addData2 }

& python -m PyInstaller --onefile --noconfirm --name $exeName --distpath $distDir --workpath $workDir $addArgs $entry

if ($LASTEXITCODE -ne 0) {
    Write-Error "PyInstaller failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Output "Build finished. Exe should be at: $distDir\$exeName.exe"
