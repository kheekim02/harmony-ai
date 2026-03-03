"""Pitch detection using librosa's pYIN algorithm."""

import numpy as np
import parselmouth

def detect_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 65.0,   # C2
    fmax: float = 1047.0,  # C6
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect pitch frame-by-frame using Praat's autocorrelation algorithm.

    Args:
        audio: Mono audio array
        sr: Sample rate
        fmin: Minimum expected frequency
        fmax: Maximum expected frequency

    Returns:
        Tuple of (f0, times, voiced_flag, voiced_prob)
        - f0: array of fundamental frequencies (Hz), NaN for unvoiced
        - times: array of timestamps for each frame
        - voiced_flag: boolean array, True where pitch is detected
        - voiced_prob: probability of each frame being voiced (mocked for Praat)
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)

    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    
    # Praat returns 0.0 for unvoiced frames. Convert to np.nan
    voiced_flag = f0 > 0.0
    f0[~voiced_flag] = np.nan
    
    # ---------------------------------------------------------
    # Advanced Pitch Smoothing / Subharmonic Drop Removal
    # ---------------------------------------------------------
    # When a singer drops into vocal fry, autocorrelation algorithms often
    # incorrectly report exactly half the true pitch (an octave drop) or a fifth.
    # We use a robust median filter to locate and eliminate these sudden deep "V" shapes.
    
    # Run a large rolling window median filter, but ONLY apply it if the 
    # current pitch deviates from the local median by more than a tritone (6 semitones).
    smoothed_f0 = np.copy(f0)
    window_size = 51  # ~510ms window to outvote fry bursts lasting up to 250ms
    half_window = window_size // 2
    
    for i in range(len(f0)):
        if np.isnan(f0[i]):
            continue
            
        # Get the current window (ignoring NaNs)
        start = max(0, i - half_window)
        end = min(len(f0), i + half_window + 1)
        window = f0[start:end]
        valid_window = window[~np.isnan(window)]
        
        if len(valid_window) > 0:
            local_median = np.median(valid_window)
            
            # If the current pitch is outside 5.5 semitones of the local median,
            # it is almost certainly a subharmonic tracking error (fry drop).
            ratio = f0[i] / local_median
            semitones_diff = 12.0 * np.log2(ratio)
            
            if abs(semitones_diff) > 5.5:
                smoothed_f0[i] = local_median

    # Replace original f0 with the smoothed contour
    f0 = smoothed_f0
    
    # Mock probabilities since Praat autocorrelation doesn't output them natively
    voiced_prob = np.where(voiced_flag, 1.0, 0.0)

    return f0, times, voiced_flag, voiced_prob
