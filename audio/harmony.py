"""Harmony generation: scale-aware pitch shifting of vocal audio."""

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key


def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
    key_string: str,
    harmony_type: str = 'Upper 3rd',
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

    # Compute per-frame shift amounts in semitones
    shifts = np.zeros(n_frames)
    for i in range(n_frames):
        if np.isnan(f0[i]) or f0[i] <= 0:
            shifts[i] = 0.0
        else:
            shifts[i] = compute_harmony_shift(f0[i], root, scale_type, interval)

    # Dynamically determine min and max pitch from the detected f0 contour
    # This prevents octave offsets during PSOLA manipulation
    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
    if len(valid_f0) > 0:
        pitch_min = max(50.0, np.min(valid_f0) * 0.8)
        pitch_max = min(2000.0, np.max(valid_f0) * 1.5)
    else:
        pitch_min, pitch_max = 75.0, 600.0

    # Smooth shifts with a rolling median to prevent "warbling"
    smoothed = np.copy(shifts)
    for i in range(2, n_frames - 2):
        smoothed[i] = np.median(shifts[i-2:i+3])
    shifts = smoothed

    # Convert semitone shifts to frequency multipliers
    shifts_hz_multiplier = 2.0 ** (shifts / 12.0)

    # ---------------------------------------------------------
    # Generate Harmony using Librosa Phase Vocoder
    # ---------------------------------------------------------
    # We found that for certain raspy/sibilant voices, Praat's PSOLA 
    # creates unavoidable metallic/frying artifacts no matter the noise gate.
    # Librosa's phase vocoder is much more resilient to noisy vocal timbres.
    
    # We have an array `shifts` which tells us how many semitones to shift 
    # at each 10ms frame. Since librosa.effects.pitch_shift operates on audio
    # arrays, we'll chunk the audio into segments of constant shift.
    
    # 1. Smooth the semitone shifts heavily to avoid warbling
    smoothed_shifts = np.copy(shifts)
    for i in range(5, n_frames - 5):
        smoothed_shifts[i] = np.median(shifts[i-5:i+6])
    shifts = np.round(smoothed_shifts) # Round to nearest semitone to create solid blocks
    
    # 2. Segment the audio into blocks of continuous shift
    harmony = np.zeros_like(audio)
    
    current_shift = shifts[0]
    start_frame = 0
    
    # Convert frames to samples (1 frame = 10ms = sr * 0.01 samples)
    samples_per_frame = int(sr * 0.01)
    
    def process_segment(start_f, end_f, shift_amt):
        start_samp = start_f * samples_per_frame
        end_samp = end_f * samples_per_frame
        
        # Ensure we don't go out of bounds
        end_samp = min(end_samp, n_samples)
        if start_samp >= n_samples or start_samp == end_samp:
            return
            
        segment = audio[start_samp:end_samp]
        
        if shift_amt == 0.0 or np.max(np.abs(segment)) < 1e-4:
            # Unvoiced or un-shifted (just copy original)
            harmony[start_samp:end_samp] = segment
        else:
            # Pitch shift this block using Librosa
            shifted_segment = librosa.effects.pitch_shift(segment, sr=sr, n_steps=shift_amt)
            harmony[start_samp:start_samp+len(shifted_segment)] = shifted_segment

    for i in range(1, n_frames):
        if shifts[i] != current_shift:
            process_segment(start_frame, i, current_shift)
            current_shift = shifts[i]
            start_frame = i
            
    # Process the final segment
    process_segment(start_frame, n_frames, current_shift)

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return np.array(harmony, dtype=np.float32)
