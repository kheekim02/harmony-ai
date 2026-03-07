"""Harmony generation: scale-aware pitch shifting of vocal audio."""

import librosa
import numpy as np
import pyworld as pw

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key


def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
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
        f0: Pitch contour from Praat (Hz, NaN for unvoiced)
        times: Array of timestamps for each frame
        key_string: Detected key like 'C Major'
        harmony_type: One of HARMONY_TYPES keys

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
    shifts = np.round(smoothed)

    # ---------------------------------------------------------
    # Generate Harmony using PyWorld Vocoder
    # ---------------------------------------------------------
    # PyWorld is a high-quality vocoder that natively separates Pitch (F0),
    # Formants (SP), and Aperiodicity/Noise (AP). It prevents metallic chipmunk
    # artifacts by allowing us to individually manipulate the pitch while keeping
    # the noise and throat shape completely intact.
    
    # 1. Parameter Extraction
    # PyWorld's `dio` and `stonemask` algorithms generate a highly accurate f0 contour.
    # We use the original audio (with percussives) because PyWorld handles AP separation.
    
    # Cast explicitly to float64, which PyWorld requires
    audio_f64 = audio.astype(np.float64)
    
    # Extract f0
    _f0, t = pw.dio(audio_f64, sr, f0_floor=65.0, f0_ceil=1047.0)
    _f0 = pw.stonemask(audio_f64, _f0, t, sr)
    
    # Extract Formants (Spectral Envelope)
    sp = pw.cheaptrick(audio_f64, _f0, t, sr)
    
    # Extract Noise (Aperiodicity)
    ap = pw.d4c(audio_f64, _f0, t, sr)

    # 2. Pitch Manipulation
    # The `shifts` array contains our scale-aware diatonic semitone shifts.
    # We interpolate our internally detected shifts to match the time resolution 
    # of the new PyWorld temporal grid.
    
    # Interpolate our `shifts` array (which aligns with `times`) to match `t`
    if len(times) == len(shifts):
        interpolated_shifts = np.interp(t, times, shifts)
    else:
        # Fallback if lengths somehow mismatch
        interpolated_shifts = np.zeros_like(t)

    # Calculate exactly what frequency to shift the contour to
    f0_shifted = np.copy(_f0)
    
    # Only shift voiced frames
    voiced_indices = _f0 > 0.0
    shift_multipliers = 2.0 ** (interpolated_shifts / 12.0)
    
    f0_shifted[voiced_indices] = _f0[voiced_indices] * shift_multipliers[voiced_indices]

    # 3. Resynthesis
    # Feed the newly shifted pitch, along with the untouched Formants and Noise,
    # back into PyWorld to generate the final high-fidelity audio.
    harmony = pw.synthesize(f0_shifted, sp, ap, sr)
    
    # Convert back to float32
    harmony = harmony.astype(np.float32)

    # Ensure it exactly matches the original length (pad or truncate if necessary)
    if len(harmony) < n_samples:
        harmony = np.pad(harmony, (0, n_samples - len(harmony)))
    elif len(harmony) > n_samples:
        harmony = harmony[:n_samples]

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return harmony

    return np.array(harmony, dtype=np.float32)
