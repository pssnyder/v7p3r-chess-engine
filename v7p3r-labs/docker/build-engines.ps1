# PowerShell version of build-engines.sh for Windows
# Build Docker images for all V7P3R engine versions

$ErrorActionPreference = "Stop"

$VERSIONS = @("12.6", "14.1", "16.1", "17.7", "18.4")
$REGISTRY = "gcr.io/rts-labs-f3981"
$IMAGE_NAME = "v7p3r"

Write-Host "Building V7P3R engine Docker images..." -ForegroundColor Green
Write-Host "Registry: $REGISTRY"
Write-Host "Versions: $($VERSIONS -join ', ')"
Write-Host ""

Set-Location $PSScriptRoot

foreach ($VERSION in $VERSIONS) {
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "Building version $VERSION" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    
    # Determine source directory
    $VERSION_DIR = "..\lichess\engines\V7P3R_v$VERSION"
    
    if (-not (Test-Path $VERSION_DIR)) {
        Write-Host "ERROR: Source directory not found: $VERSION_DIR" -ForegroundColor Red
        exit 1
    }
    
    # Copy source to temporary build context
    $BUILD_DIR = ".\build_v$($VERSION.Replace('.', '_'))"
    if (Test-Path $BUILD_DIR) {
        Remove-Item -Recurse -Force $BUILD_DIR
    }
    New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    Copy-Item -Recurse "$VERSION_DIR\src" "$BUILD_DIR\"
    Copy-Item "Dockerfile" "$BUILD_DIR\"
    
    # Build image
    $IMAGE_TAG = "$REGISTRY/$IMAGE_NAME`:$VERSION"
    Write-Host "Building: $IMAGE_TAG" -ForegroundColor Yellow
    
    docker build -t $IMAGE_TAG $BUILD_DIR
    
    # Tag as latest if v18.4
    if ($VERSION -eq "18.4") {
        docker tag $IMAGE_TAG "$REGISTRY/$IMAGE_NAME`:latest"
        Write-Host "Tagged as latest" -ForegroundColor Green
    }
    
    # Clean up build directory
    Remove-Item -Recurse -Force $BUILD_DIR
    
    Write-Host "✓ Built $IMAGE_TAG" -ForegroundColor Green
    Write-Host ""
}

Write-Host "=========================================" -ForegroundColor Green
Write-Host "All images built successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Authenticate: gcloud auth configure-docker gcr.io"
Write-Host "2. Push images: .\push-engines.ps1"
Write-Host "=========================================" -ForegroundColor Green
