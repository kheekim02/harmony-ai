"""Harmony generation: scale-aware pitch shifting of vocal audio."""

import librosa
import numpy as np

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key


def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
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
        f0: Pitch contour from pYIN (Hz, NaN for unvoiced)
        key_string: Detected key like 'C Major'
        harmony_type: One of HARMONY_TYPES keys
        hop_length: Analysis hop length (must match pitch detection)

    Returns:
        Harmony audio array (same length as input)
    """
    root, scale_type = parse_key(key_string)
    interval = HARMONY_TYPES.get(harmony_type, 2)

    n_frames = len(f0)
    n_samples = len(audio)

    # Compute per-frame shift amounts
    shifts = np.zeros(n_frames)
    for i in range(n_frames):
        if np.isnan(f0[i]) or f0[i] <= 0:
            shifts[i] = 0.0
        else:
            shifts[i] = compute_harmony_shift(f0[i], root, scale_type, interval)

    # Smooth shifts with a rolling median to prevent "warbling" or rapid jumps
    # (e.g. from singer's vibrato crossing a scale degree boundary)
    smoothed = np.copy(shifts)
    for i in range(2, n_frames - 2):
        smoothed[i] = np.median(shifts[i-2:i+3])
    shifts = smoothed

    # Group frames into segments with the same shift (quantized to nearest semitone)
    segments = []
    current_shift = round(shifts[0])
    seg_start = 0

    for i in range(1, n_frames):
        rounded = round(shifts[i])
        if rounded != current_shift or i == n_frames - 1:
            seg_end = i if i < n_frames - 1 else n_frames
            
            # Ignore very short segments (< 5 frames = ~100ms) to prevent audio "jumping"
            # unless the note actually stopped (rounded == 0)
            if (seg_end - seg_start) < 5 and i < n_frames - 1 and rounded != 0 and current_shift != 0:
                pass  # Keep building current segment
            else:
                segments.append((seg_start, seg_end, current_shift))
                seg_start = i
                current_shift = rounded

    if seg_start < n_frames:
        segments.append((seg_start, n_frames, current_shift))

    # Build harmony audio by pitch-shifting each segment
    harmony = np.zeros(n_samples, dtype=np.float32)
    crossfade_len = min(hop_length * 2, 1024)  # Longer crossfade (~46ms) for smoother transitions

    for seg_start_frame, seg_end_frame, shift in segments:
        if shift == 0:
            # No shift needed (unvoiced or same note) — use original
            start_sample = seg_start_frame * hop_length
            end_sample = min(seg_end_frame * hop_length, n_samples)
            harmony[start_sample:end_sample] = audio[start_sample:end_sample]
            continue

        # Extract segment with a little padding
        start_sample = max(0, seg_start_frame * hop_length - crossfade_len)
        end_sample = min(n_samples, seg_end_frame * hop_length + crossfade_len)
        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) < 2048:
            # Too short to pitch shift meaningfully
            actual_start = seg_start_frame * hop_length
            actual_end = min(seg_end_frame * hop_length, n_samples)
            harmony[actual_start:actual_end] = audio[actual_start:actual_end]
            continue

        # Pitch shift this segment
        shifted = librosa.effects.pitch_shift(
            y=segment_audio,
            sr=sr,
            n_steps=float(shift),
        )

        # Place back with crossfade
        actual_start = seg_start_frame * hop_length
        actual_end = min(seg_end_frame * hop_length, n_samples)
        offset = actual_start - start_sample

        shifted_section = shifted[offset:offset + (actual_end - actual_start)]
        if len(shifted_section) < (actual_end - actual_start):
            # Pad if needed
            pad = np.zeros((actual_end - actual_start) - len(shifted_section))
            shifted_section = np.concatenate([shifted_section, pad])

        # Apply simple crossfade at boundaries
        fade_len = min(crossfade_len, len(shifted_section), actual_end - actual_start)
        if fade_len > 1:
            fade_in = np.linspace(0, 1, fade_len)
            fade_out = np.linspace(1, 0, fade_len)

            # Blend start
            if actual_start > 0:
                shifted_section[:fade_len] *= fade_in
                harmony[actual_start:actual_start + fade_len] *= fade_out

            # Blend end
            if actual_end < n_samples:
                shifted_section[-fade_len:] *= fade_out

        harmony[actual_start:actual_end] += shifted_section[:actual_end - actual_start]

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return harmony
