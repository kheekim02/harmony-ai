"""Harmony generation: scale-aware pitch shifting of vocal audio."""

import librosa
import numpy as np
import scipy.signal
import pyworld as pw

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key

def fill_all_gaps(f0: np.ndarray) -> np.ndarray:
    """
    Interpolates over all gaps of 0.0 in the F0 array, completely bridging
    unvoiced regions to force a continuous pitch contour. This prevents
    metallic toggling between voiced/unvoiced states in PyWorld.
    """
    f0_filled = np.copy(f0)
    
    # Standard numpy trick to linearly interpolate over all zeros
    f0_filled[f0_filled == 0.0] = np.nan
    valid_indices = ~np.isnan(f0_filled)
    
    if np.any(valid_indices):
        f0_filled = np.interp(
            np.arange(len(f0_filled)),
            np.arange(len(f0_filled))[valid_indices],
            f0_filled[valid_indices]
        )
    else:
        # Failsafe if the entire track is unvoiced
        f0_filled.fill(0.0)
        
    return f0_filled

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

    # ---------------------------------------------------------
    # Generate Harmony using PyWorld Vocoder
    # ---------------------------------------------------------
    # PyWorld is a high-quality vocoder that natively separates Pitch (F0),
    # Formants (SP), and Aperiodicity/Noise (AP). It prevents metallic chipmunk
    # artifacts by allowing us to individually manipulate the pitch while keeping
    # the noise and throat shape completely intact.
    
    # 1. Parameter Extraction
    # PyWorld requires float64
    audio_f64 = audio.astype(np.float64)
    
    # Use Harvest algorithm instead of DIO. Harvest is much more robust
    # at extracting stable F0 contours from noisy or complex audio (vocal fry),
    # which flattens out high-frequency metallic trembling during vocal shifts.
    _f0, t = pw.harvest(audio_f64, sr, f0_floor=65.0, f0_ceil=1047.0)
    
    # 2. Extract Spectrogram and Aperiodicity using the RAW _f0
    # PyWorld must compute AP based on the original un-filled gaps, 
    # otherwise we get robotic drones during breaths.
    sp = pw.cheaptrick(audio_f64, _f0, t, sr)
    ap = pw.d4c(audio_f64, _f0, t, sr)

    # 3. Pitch Manipulation & Contour Smoothing
    # Fill ALL gaps to completely eliminate source-filter dropouts
    _f0_filled = fill_all_gaps(_f0) 

    native_shifts = np.zeros_like(_f0_filled)
    for i in range(len(_f0_filled)):
        if _f0_filled[i] > 0.0:
            native_shifts[i] = compute_harmony_shift(_f0_filled[i], root, scale_type, interval)
        else:
            native_shifts[i] = 0.0

    # Apply heavy median smoothing to the shift map (kernel_size=15 is approx 75ms).
    # This prevents the target pitch from wobbling mathematically over breaths 
    # or micro-fluctuations.
    smoothed_shifts = scipy.signal.medfilt(native_shifts, kernel_size=15)

    # Apply light median smoothing to the base filled F0 contour itself.
    f0_smooth = scipy.signal.medfilt(_f0_filled, kernel_size=5)

    # Apply shifted multipliers (now fully continuous across the entire track)
    f0_shifted = np.copy(f0_smooth)
    shift_multipliers = 2.0 ** (smoothed_shifts / 12.0)
    
    # Since we filled all gaps, the F0 is continuous. We can multiply everything.
    f0_shifted = f0_smooth * shift_multipliers

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
