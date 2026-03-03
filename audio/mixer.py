"""Audio mixing and export."""

import numpy as np
from .loader import save_audio


def mix_audio(
    original: np.ndarray,
    harmony: np.ndarray,
    harmony_volume: float = 0.7,
    stereo_width: float = 0.4,
) -> np.ndarray:
    """Mix original and harmony audio into a stereo output.

    Args:
        original: Original mono audio
        harmony: Harmony mono audio
        harmony_volume: Volume of harmony (0.0 - 1.0)
        stereo_width: Stereo separation (0.0 = mono, 1.0 = hard pan)

    Returns:
        Stereo audio array of shape (2, n_samples)
    """
    # Ensure same length
    min_len = min(len(original), len(harmony))
    orig = original[:min_len]
    harm = harmony[:min_len] * harmony_volume

    # Stereo panning
    # Original: slightly left of center
    # Harmony: slightly right of center
    orig_left = orig * (0.5 + stereo_width * 0.25)
    orig_right = orig * (0.5 - stereo_width * 0.25)
    harm_left = harm * (0.5 - stereo_width * 0.25)
    harm_right = harm * (0.5 + stereo_width * 0.25)

    left = orig_left + harm_left
    right = orig_right + harm_right

    stereo = np.stack([left, right])

    # Normalize
    peak = np.max(np.abs(stereo))
    if peak > 1.0:
        stereo = stereo / peak * 0.95

    return stereo


def export_mix(
    filepath: str,
    original: np.ndarray,
    harmony: np.ndarray,
    sr: int = 44100,
    harmony_volume: float = 0.7,
    stereo_width: float = 0.4,
):
    """Mix and export to WAV file."""
    stereo = mix_audio(original, harmony, harmony_volume, stereo_width)
    save_audio(filepath, stereo, sr)


def export_harmony_only(filepath: str, harmony: np.ndarray, sr: int = 44100):
    """Export just the harmony track."""
    save_audio(filepath, harmony, sr)
