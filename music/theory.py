"""Music theory: interval calculations and note mapping."""

import numpy as np
from .scales import NOTE_NAMES, build_scale


def hz_to_midi(freq: float) -> float:
    """Convert frequency in Hz to MIDI note number (float)."""
    if freq <= 0:
        return 0.0
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_chroma(midi: float) -> int:
    """Convert MIDI note to chroma class (0-11)."""
    return int(round(midi)) % 12


def nearest_scale_degree(chroma: int, scale_chromas: list[int]) -> int:
    """Find the nearest scale degree for a chroma value.

    Returns:
        Index (0-6) of the nearest scale degree
    """
    min_dist = 12
    best_idx = 0
    for i, sc in enumerate(scale_chromas):
        dist = min(abs(chroma - sc), 12 - abs(chroma - sc))
        if dist < min_dist:
            min_dist = dist
            best_idx = i
    return best_idx


def compute_harmony_shift(
    pitch_hz: float,
    root_note: str,
    scale_type: str,
    interval: int,
) -> float:
    """Compute the semitone shift needed for a harmony at the given interval.

    This is the core scale-aware algorithm. Instead of blindly shifting by
    a fixed number of semitones, we:
    1. Find which scale degree the current note is closest to
    2. Move up/down by `interval` scale degrees
    3. Compute the exact semitone difference

    Args:
        pitch_hz: Current pitch in Hz
        root_note: Root of the key (e.g., 'C')
        scale_type: 'major' or 'minor'
        interval: Number of scale degrees to shift (e.g., 2 for a 3rd, 4 for a 5th)
                  Positive = up, negative = down

    Returns:
        Number of semitones to shift (float, can be fractional)
    """
    if pitch_hz <= 0:
        return 0.0

    midi = hz_to_midi(pitch_hz)
    chroma = midi_to_chroma(midi)
    scale = build_scale(root_note, scale_type)

    # Find nearest scale degree
    degree = nearest_scale_degree(chroma, scale)

    # Target degree (wrapping around the octave)
    target_degree = degree + interval
    octave_shift = 0

    while target_degree >= 7:
        target_degree -= 7
        octave_shift += 1
    while target_degree < 0:
        target_degree += 7
        octave_shift -= 1

    # Compute semitone distance
    original_chroma = scale[degree]
    target_chroma = scale[target_degree]

    # Semitone difference within the octave
    diff = target_chroma - original_chroma
    if interval > 0 and diff <= 0:
        diff += 12
    elif interval < 0 and diff >= 0:
        diff -= 12

    # Add octave shifts
    total_shift = diff + (octave_shift * 12)

    return float(total_shift)


# Predefined harmony types
HARMONY_TYPES = {
    'Upper 3rd': 2,      # 2 scale degrees up
    'Lower 3rd': -2,     # 2 scale degrees down
    'Upper 5th': 4,      # 4 scale degrees up
    'Octave Up': 7,      # 7 scale degrees up (= octave)
    'Octave Down': -7,   # 7 scale degrees down (= octave)
}
