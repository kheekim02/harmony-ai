"""Harmony generation: PSOLA-based pitch shifting of vocal audio.

Uses Praat's Pitch-Synchronous Overlap-and-Add (PSOLA) algorithm via
parselmouth for time-domain pitch shifting. PSOLA works by slicing the
waveform at individual pitch periods and overlapping them at a new rate,
avoiding the phase coherence issues of spectral vocoders like PyWorld.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

from music.theory import HARMONY_TYPES


def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
    key_string: str,
    harmony_type: str = 'Upper 3rd',
    hop_length: int = 512,
) -> np.ndarray:
    """Generate a harmony track from the original audio using Praat PSOLA.

    Args:
        audio: Original mono audio array (float32)
        sr: Sample rate
        f0: Pitch contour (Hz, NaN for unvoiced) — used for reference only
        times: Array of timestamps for each frame
        key_string: Detected key like 'C Major'
        harmony_type: One of HARMONY_TYPES keys

    Returns:
        Harmony audio array (same length as input)
    """
    n_samples_orig = len(audio)

    # ---------------------------------------------------------
    # 1. Compute the pitch shift in semitones
    # ---------------------------------------------------------
    fixed_semitone_map = {
        'Upper 3rd': 4.0,
        'Lower 3rd': -4.0,
        '5th': 7.0,
        'Octave Up': 12.0,
        'Octave Down': -12.0,
    }
    shift_semitones = fixed_semitone_map.get(harmony_type, 4.0)
    factor = 2.0 ** (shift_semitones / 12.0)

    # ---------------------------------------------------------
    # 2. Praat PSOLA Pitch Shifting (Time-Domain)
    # ---------------------------------------------------------
    # Create a Praat Sound object from the numpy array
    snd = parselmouth.Sound(audio, sampling_frequency=sr)

    # Create a Manipulation object (Praat's pitch manipulation framework)
    # Parameters: time_step, minimum_pitch, maximum_pitch
    manipulation = call(snd, "To Manipulation", 0.01, 75, 600)

    # Extract the pitch tier
    pitch_tier = call(manipulation, "Extract pitch tier")

    # Multiply all pitches by the semitone factor
    call(pitch_tier, "Multiply frequencies", snd.xmin, snd.xmax, factor)

    # Replace the pitch tier in the manipulation
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Resynthesize using PSOLA (overlap-add, time-domain)
    shifted_snd = call(manipulation, "Get resynthesis (overlap-add)")

    # Extract the numpy array from Praat Sound
    harmony = shifted_snd.values[0].astype(np.float32)

    # ---------------------------------------------------------
    # 3. Final Output Formatting
    # ---------------------------------------------------------
    # Ensure exact length match with original
    if len(harmony) < n_samples_orig:
        harmony = np.pad(harmony, (0, n_samples_orig - len(harmony)))
    elif len(harmony) > n_samples_orig:
        harmony = harmony[:n_samples_orig]

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return harmony
