"""Pitch detection using librosa's pYIN algorithm."""

import librosa
import numpy as np


def detect_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 65.0,   # C2
    fmax: float = 1047.0,  # C6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect pitch frame-by-frame using pYIN.

    Args:
        audio: Mono audio array
        sr: Sample rate
        fmin: Minimum expected frequency
        fmax: Maximum expected frequency

    Returns:
        Tuple of (f0, voiced_flag, voiced_prob)
        - f0: array of fundamental frequencies (Hz), NaN for unvoiced
        - voiced_flag: boolean array, True where pitch is detected
        - voiced_prob: probability of each frame being voiced
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        audio,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=2048,
        hop_length=512,
    )

    return f0, voiced_flag, voiced_prob


def get_frame_times(audio: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    """Get the time (in seconds) for each analysis frame."""
    n_frames = 1 + len(audio) // hop_length
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop_length)
