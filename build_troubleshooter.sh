#!/bin/bash
# Exit on error
set -e

echo "Building HarmonyAI Troubleshooter..."

# Activate virtual environment
source .venv/bin/activate

# Clean previous builds
rm -rf build_troubleshooter dist/HarmonyAI_Troubleshooter.app

# Run PyInstaller with PySide6 hooks
pyinstaller \
    --name="HarmonyAI_Troubleshooter" \
    --windowed \
    --noconfirm \
    --clean \
    --log-level=INFO \
    --hidden-import=PySide6.QtCore \
    --hidden-import=PySide6.QtGui \
    --hidden-import=PySide6.QtWidgets \
    --hidden-import=PySide6.QtMultimedia \
    --hidden-import=librosa \
    --hidden-import=soundfile \
    --hidden-import=parselmouth \
    --hidden-import=scipy.signal \
    --workpath build_troubleshooter \
    troubleshoot_gui.py

echo "Build complete! Check the 'dist' folder for HarmonyAI_Troubleshooter.app"
