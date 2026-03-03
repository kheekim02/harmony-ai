"""Key detection from pitch data using Krumhansl-Kessler profiles."""

import numpy as np
from music.scales import NOTE_NAMES, MAJOR_PROFILE, MINOR_PROFILE


def detect_key(f0: np.ndarray) -> tuple[str, float]:
    """Detect the musical key from a pitch contour.

    Uses Krumhansl-Kessler key-finding algorithm:
    1. Build a chroma histogram from detected pitches
    2. Correlate with major and minor key profiles
    3. Return the best-matching key

    Args:
        f0: Array of fundamental frequencies (Hz), NaN for unvoiced

    Returns:
        Tuple of (key_string, confidence)
        e.g., ('C Major', 0.85)
    """
    # Filter out NaN and zero values
    valid_pitches = f0[~np.isnan(f0)]
    valid_pitches = valid_pitches[valid_pitches > 0]

    if len(valid_pitches) == 0:
        return 'C Major', 0.0

    # Convert to MIDI and then chroma (0-11)
    midi_notes = 69 + 12 * np.log2(valid_pitches / 440.0)
    chromas = np.round(midi_notes).astype(int) % 12

    # Build chroma histogram
    histogram = np.zeros(12)
    for c in chromas:
        histogram[c] += 1

    # Normalize
    total = histogram.sum()
    if total > 0:
        histogram = histogram / total

    # Correlate with all 24 keys
    best_key = 'C Major'
    best_corr = -1.0

    major_prof = np.array(MAJOR_PROFILE)
    minor_prof = np.array(MINOR_PROFILE)

    # Normalize profiles
    major_prof = major_prof / np.linalg.norm(major_prof)
    minor_prof = minor_prof / np.linalg.norm(minor_prof)
    hist_norm = histogram / (np.linalg.norm(histogram) + 1e-10)

    for i, note in enumerate(NOTE_NAMES):
        # Rotate histogram so that this note is at index 0
        rotated = np.roll(hist_norm, -i)

        # Correlate with major profile
        corr_major = np.dot(rotated, major_prof)
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f'{note} Major'

        # Correlate with minor profile
        corr_minor = np.dot(rotated, minor_prof)
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f'{note} Minor'

    return best_key, float(best_corr)
