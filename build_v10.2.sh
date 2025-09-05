#!/bin/bash
# V7P3R v10.2 Build Script
# Builds the V7P3R_v10.2.exe engine with PV Following functionality

echo "🚀 Building V7P3R v10.2 Chess Engine"
echo "======================================"

# Navigate to project directory
cd "s:/Maker Stuff/Programming/Chess Engines/V7P3R Chess Engine/v7p3r-chess-engine"

# Clean previous build artifacts
echo "🧹 Cleaning previous build artifacts..."
rm -rf build/V7P3R_v10.2_RELEASE/*
rm -rf dist/V7P3R_v10.2.exe

# Run PyInstaller with the v10.2 spec file
echo "🔨 Building V7P3R_v10.2.exe..."
pyinstaller V7P3R_v10.2_RELEASE.spec --distpath dist --workpath build/V7P3R_v10.2_RELEASE

# Check if build was successful
if [ -f "dist/V7P3R_v10.2.exe" ]; then
    echo "✅ BUILD SUCCESS!"
    echo "📁 Executable created: dist/V7P3R_v10.2.exe"
    
    # Get file size
    file_size=$(stat -f%z "dist/V7P3R_v10.2.exe" 2>/dev/null || stat -c%s "dist/V7P3R_v10.2.exe" 2>/dev/null || echo "Unknown")
    echo "📏 File size: ${file_size} bytes"
    
    # Test the executable
    echo "🧪 Testing executable..."
    echo "uci" | "dist/V7P3R_v10.2.exe" | head -20
    
    echo ""
    echo "🎉 V7P3R v10.2 BUILD COMPLETE!"
    echo "💫 Features included:"
    echo "   - Enhanced PV Following System"
    echo "   - Instant move recognition using board states"
    echo "   - Clean UCI output"
    echo "   - Optimized search and evaluation"
    echo "   - Ready for Arena GUI and tournaments"
    
else
    echo "❌ BUILD FAILED!"
    echo "Check the build log above for errors."
    exit 1
fi
