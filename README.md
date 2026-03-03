# HarmonyAI — Auto Vocal Harmony Generator 🎤✨

HarmonyAI is a standalone macOS desktop application that takes a vocal audio file and automatically generates musically-correct harmony lines. It operates **100% offline**—no cloud uploads, no API keys, no subscriptions.

## Features
- **Instant Processing:** Drop a vocal track in and get instant harmonies.
- **Scale-Aware Pitch Shifting:** Automatically detects the key from the pitch contour and calculates exact diatonic semitone shifts for harmonies (e.g., proper diatonic 3rds, 5ths, or octaves).
- **Dual Waveform GUI:** Beautiful, real-time waveform display of the original and harmony tracks.
- **Built-in Mixer:** Adjust the harmony volume and stereo pan width on the fly.
- **Export to WAV:** Save the harmony track alone or the full mixed result.
- **Local-First:** Completely private. All AI pitch detection (via pYIN) and audio processing runs on-device.

## Harmony Types Supported
- Upper 3rd (two scale degrees up)
- Lower 3rd (two scale degrees down)
- Upper 5th (four scale degrees up)
- Octave Up
- Octave Down

## How It Works
1. **Load Audio:** The app loads and normalizes your vocal track.
2. **Pitch Detection:** Extracts the frame-by-frame fundamental frequency (F0) using the pYIN algorithm.
3. **Key Detection:** Maps pitches to a chroma histogram and uses the Krumhansl-Kessler algorithm to detect the most likely major or minor key.
4. **Scale-Aware Shift:** Calculates the proper interval shift per note matching the detected scale, meaning a "3rd" might be +3 semitones or +4 semitones depending on the note's position in the scale.
5. **Phase Vocoder Shift:** Pitch-shifts audio segments and smoothly crossfades them to create the harmony line.
6. **Mix:** Returns a panned stereo mix.

## Installation

### Pre-built `.app` (macOS)
If you have the pre-built application, simply double-click `HarmonyAI.app` to launch. No dependencies needed.

### Running from Source
Requires Python 3.12+

```bash
# Clone the repository
git clone https://github.com/yourusername/harmony-ai
cd harmony-ai

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (PySide6, librosa, soundfile, numpy)
pip install -r requirements.txt

# Run the app
python main.py
```

### Building the `.app` Bundle
To package the app yourself:
```bash
pip install pyinstaller
./build.sh
```
The resulting `HarmonyAI.app` will be in the `dist/` directory.

## Tech Stack
- **GUI:** Python / PySide6 (Qt)
- **Audio I/O:** `soundfile`
- **DSP & Pitch Detection:** `librosa` / `numpy`
- **Packaging:** `PyInstaller`

## License
MIT License
