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
    
    # Mock probabilities since Praat autocorrelation doesn't output them natively in the same way
    voiced_prob = np.where(voiced_flag, 1.0, 0.0)
    
    f0[~voiced_flag] = np.nan

    return f0, times, voiced_flag, voiced_prob
