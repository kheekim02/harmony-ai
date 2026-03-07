#!/bin/bash
# Build script for HarmonyAI macOS .app bundle

echo "Building HarmonyAI..."

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Clean previous builds
rm -rf build dist *.spec

# Run PyInstaller
# --windowed: Creates a macOS .app bundle
# --name: App name
# --hidden-import: Ensure librosa and soundfile dependencies are found
# --noconfirm: Overwrite output directory without asking

pyinstaller --windowed \
            --name "HarmonyAI" \
            --hidden-import "librosa" \
            --hidden-import "soundfile" \
            --hidden-import "numpy" \
            --hidden-import "pyworld" \
            --hidden-import "pkg_resources" \
            --hidden-import "setuptools" \
            --noconfirm \
            main.py

echo "Build complete! Check the 'dist' folder for HarmonyAI.app"
