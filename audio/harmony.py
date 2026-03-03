"""Harmony generation: scale-aware pitch shifting of vocal audio."""

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key


def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    key_string: str,
    harmony_type: str = 'Upper 3rd',
    hop_length: int = 512,
) -> np.ndarray:
    """Generate a harmony track from the original audio.

    This is the core algorithm:
    1. For each pitched frame, compute the scale-aware semitone shift
    2. Group consecutive frames with similar shifts into segments
    3. Pitch-shift each segment
    4. Crossfade between segments

    Args:
        audio: Original mono audio array
        sr: Sample rate
        f0: Pitch contour from pYIN (Hz, NaN for unvoiced)
        key_string: Detected key like 'C Major'
        harmony_type: One of HARMONY_TYPES keys
        hop_length: Analysis hop length (must match pitch detection)

    Returns:
        Harmony audio array (same length as input)
    """
    root, scale_type = parse_key(key_string)
    interval = HARMONY_TYPES.get(harmony_type, 2)

    n_frames = len(f0)
    n_samples = len(audio)

    # Compute per-frame shift amounts in semitones
    shifts = np.zeros(n_frames)
    for i in range(n_frames):
        if np.isnan(f0[i]) or f0[i] <= 0:
            shifts[i] = 0.0
        else:
            shifts[i] = compute_harmony_shift(f0[i], root, scale_type, interval)

    # Smooth shifts with a rolling median to prevent "warbling"
    smoothed = np.copy(shifts)
    for i in range(2, n_frames - 2):
        smoothed[i] = np.median(shifts[i-2:i+3])
    shifts = smoothed

    # Convert semitone shifts to frequency multipliers
    shifts_hz_multiplier = 2.0 ** (shifts / 12.0)

    # Calculate timestamps for each frame
    times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)

    # ---------------------------------------------------------
    # Generate Harmony using PSOLA via Parselmouth (Praat)
    # ---------------------------------------------------------
    # Parselmouth expects a 2D array (channels, samples) or 1D array
    sound = parselmouth.Sound(audio, sampling_frequency=sr)

    # Create manipulation object (time step 0.01, min pitch 75Hz, max pitch 600Hz)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)

    # Extract original pitch tier and then remove it to replace with our custom one
    pitch_tier = call(manipulation, "Extract pitch tier")
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Create a new Empty PitchTier
    duration = call(sound, "Get total duration")
    new_pitch_tier = call("Create PitchTier", "harmony", 0.0, duration)

    # Populate the PitchTier with our shifted continuously varying pitch contour
    for i, t in enumerate(times):
        if i < len(f0) and not np.isnan(f0[i]) and f0[i] > 0:
            new_f0 = f0[i] * shifts_hz_multiplier[i]
            call(new_pitch_tier, "Add point", t, new_f0)

    # Replace the pitch tier in the manipulation object
    call([new_pitch_tier, manipulation], "Replace pitch tier")

    # Resynthesize the audio using Pitch-Synchronous Overlap Add (PSOLA)
    harmony_sound = call(manipulation, "Get resynthesis (overlap-add)")

    # Extract the resulting numpy array
    harmony = harmony_sound.values[0]

    # Ensure it exactly matches the original length (pad or truncate if necessary)
    if len(harmony) < n_samples:
        harmony = np.pad(harmony, (0, n_samples - len(harmony)))
    elif len(harmony) > n_samples:
        harmony = harmony[:n_samples]

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return np.array(harmony, dtype=np.float32)
