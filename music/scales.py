"""Music scale definitions and utilities."""

# All 12 chromatic note names
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Scale interval patterns (in semitones from root)
SCALE_PATTERNS = {
    'major': [0, 2, 4, 5, 7, 9, 11],
    'minor': [0, 2, 3, 5, 7, 8, 10],
}

# Krumhansl-Kessler key profiles for key detection
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]


def build_scale(root_note: str, scale_type: str = 'major') -> list[int]:
    """Build a scale as list of MIDI-style chroma values (0-11).

    Args:
        root_note: Root note name (e.g., 'C', 'F#')
        scale_type: 'major' or 'minor'

    Returns:
        List of 7 chroma values (0-11)
    """
    root = NOTE_NAMES.index(root_note)
    pattern = SCALE_PATTERNS[scale_type]
    return [(root + interval) % 12 for interval in pattern]


def get_all_keys() -> list[str]:
    """Return all 24 major and minor key names."""
    keys = []
    for note in NOTE_NAMES:
        keys.append(f"{note} Major")
        keys.append(f"{note} Minor")
    return keys


def parse_key(key_string: str) -> tuple[str, str]:
    """Parse a key string like 'C Major' into (root, type).

    Returns:
        Tuple of (root_note, scale_type) e.g., ('C', 'major')
    """
    parts = key_string.strip().split()
    root = parts[0]
    scale_type = parts[1].lower() if len(parts) > 1 else 'major'
    return root, scale_type
