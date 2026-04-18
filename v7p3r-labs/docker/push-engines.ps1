# PowerShell script to push Docker images to GCR
# Run after build-engines.ps1

$ErrorActionPreference = "Stop"

$VERSIONS = @("12.6", "14.1", "16.1", "17.7", "18.4")
$REGISTRY = "gcr.io/rts-labs-f3981"
$IMAGE_NAME = "v7p3r"

Write-Host "Pushing V7P3R engine images to GCR..." -ForegroundColor Green
Write-Host ""

# Authenticate to GCR
Write-Host "Configuring Docker authentication for GCR..." -ForegroundColor Yellow
gcloud auth configure-docker gcr.io

Write-Host ""

foreach ($VERSION in $VERSIONS) {
    $IMAGE_TAG = "$REGISTRY/$IMAGE_NAME`:$VERSION"
    Write-Host "Pushing: $IMAGE_TAG" -ForegroundColor Cyan
    docker push $IMAGE_TAG
    Write-Host "✓ Pushed $IMAGE_TAG" -ForegroundColor Green
    Write-Host ""
}

# Push latest tag
Write-Host "Pushing latest tag..." -ForegroundColor Cyan
docker push "$REGISTRY/$IMAGE_NAME`:latest"
Write-Host "✓ Pushed latest tag" -ForegroundColor Green

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "All images pushed successfully!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
