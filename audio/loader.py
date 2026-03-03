"""Audio file loading and normalization."""

import librosa
import numpy as np
import soundfile as sf


DEFAULT_SR = 22050


def load_audio(filepath: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and normalize.

    Args:
        filepath: Path to WAV, MP3, FLAC, or OGG file
        sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    y, loaded_sr = librosa.load(filepath, sr=sr, mono=True)

    # Normalize to [-1, 1]
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak

    return y, sr


def save_audio(filepath: str, audio: np.ndarray, sr: int = 44100):
    """Save audio array to WAV file.

    Args:
        filepath: Output file path
        audio: Audio array (mono or stereo)
        sr: Sample rate
    """
    # Ensure no clipping
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    sf.write(filepath, audio.T if audio.ndim > 1 else audio, sr, subtype='PCM_16')


def get_duration(audio: np.ndarray, sr: int) -> float:
    """Get duration in seconds."""
    return len(audio) / sr
