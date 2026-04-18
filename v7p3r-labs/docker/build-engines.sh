#!/bin/bash
# Build Docker images for all V7P3R engine versions
# Reads from ../lichess/engines/ directory

set -e

VERSIONS=("12.6" "14.1" "16.1" "17.7" "18.4")
REGISTRY="gcr.io/rts-labs-f3981"
IMAGE_NAME="v7p3r"

echo "Building V7P3R engine Docker images..."
echo "Registry: $REGISTRY"
echo "Versions: ${VERSIONS[@]}"
echo ""

cd "$(dirname "$0")"

for VERSION in "${VERSIONS[@]}"; do
    echo "========================================="
    echo "Building version $VERSION"
    echo "========================================="
    
    # Determine source directory
    VERSION_DIR="../lichess/engines/V7P3R_v${VERSION}"
    
    if [ ! -d "$VERSION_DIR" ]; then
        echo "ERROR: Source directory not found: $VERSION_DIR"
        exit 1
    fi
    
    # Copy source to temporary build context
    BUILD_DIR="./build_v${VERSION}"
    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR"
    cp -r "$VERSION_DIR/src" "$BUILD_DIR/"
    cp Dockerfile "$BUILD_DIR/"
    
    # Build image
    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"
    echo "Building: $IMAGE_TAG"
    
    docker build -t "$IMAGE_TAG" "$BUILD_DIR"
    
    # Tag as latest if v18.4
    if [ "$VERSION" == "18.4" ]; then
        docker tag "$IMAGE_TAG" "$REGISTRY/$IMAGE_NAME:latest"
        echo "Tagged as latest"
    fi
    
    # Clean up build directory
    rm -rf "$BUILD_DIR"
    
    echo "✓ Built $IMAGE_TAG"
    echo ""
done

echo "========================================="
echo "All images built successfully!"
echo ""
echo "Next steps:"
echo "1. Authenticate: gcloud auth configure-docker gcr.io"
echo "2. Push images: ./push-engines.sh"
echo "========================================="
