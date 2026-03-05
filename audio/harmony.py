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

    # Smooth shifts with a rolling median to prevent "warbling"
    smoothed = np.copy(shifts)
    for i in range(2, n_frames - 2):
        smoothed[i] = np.median(shifts[i-2:i+3])
    shifts = np.round(smoothed)

    # ---------------------------------------------------------
    # Harmonic-Percussive Source Separation (HPSS)
    # ---------------------------------------------------------
    # PSOLA pitch shifting creates extreme metallic tearing when fed 
    # unvoiced signals or vocal fry. By pre-processing the audio with HPSS,
    # we mathematically extract the pure pitched vocal tone (Harmonic) from 
    # the breaths, sibilance, and fry noise (Percussive).
    # We use a margin of 2.0 to aggressively isolate the harmonic elements.
    
    harmonic, percussive = librosa.effects.hpss(audio, margin=2.0)

    # ---------------------------------------------------------
    # Generate Harmony using Praat LPC Resynthesis
    # ---------------------------------------------------------
    # By running LPC Resynthesis strictly on the clean Harmonic tone,
    # we mathematically preserve formants on high notes (no metallic chipmunk tearing)
    # while the HPSS pre-processor protects the transients from sounding "shaky".
    
    # Dynamically determine min and max pitch from the detected f0 contour
    valid_f0 = f0[~np.isnan(f0) & (f0 > 0)]
    if len(valid_f0) > 0:
        pitch_min = max(50.0, np.min(valid_f0) * 0.8)
        pitch_max = min(2000.0, np.max(valid_f0) * 1.5)
    else:
        pitch_min, pitch_max = 75.0, 600.0

    # Convert semitone shifts to frequency multipliers
    shifts_hz_multiplier = 2.0 ** (shifts / 12.0)

    # We strictly feed ONLY the clean harmonic audio track to the Praat engine.
    sound = parselmouth.Sound(harmonic, sampling_frequency=sr)

    # Create manipulation object using dynamic pitch bounds
    manipulation = call(sound, "To Manipulation", 0.01, pitch_min, pitch_max)

    # Extract original pitch tier and then remove it to replace with our custom one
    pitch_tier = call(manipulation, "Extract pitch tier")
    call([pitch_tier, manipulation], "Replace pitch tier")

    # Create a new Empty PitchTier
    duration = call(sound, "Get total duration")
    new_pitch_tier = call("Create PitchTier", "harmony", 0.0, duration)

    # Populate the PitchTier with our shifted continuously varying pitch contour
    for i, t in enumerate(times):
        if i < len(f0) and not np.isnan(f0[i]) and f0[i] > 0:
            new_f0 = f0[i] * shifts_hz_multiplier[i]
            call(new_pitch_tier, "Add point", t, new_f0)

    # Replace the pitch tier in the manipulation object
    call([new_pitch_tier, manipulation], "Replace pitch tier")

    # Resynthesize the audio using Linear Predictive Coding (LPC)
    # This prevents the formant distortion ("chipmunk effect") on high notes
    harmony_sound = call(manipulation, "Get resynthesis (LPC)")

    # Extract the resulting numpy array and importantly, resample it back
    # to the original sample rate (Praat LPC inherently downsamples to 10kHz)
    harmony = harmony_sound.values[0]
    out_sr = harmony_sound.sampling_frequency
    
    if out_sr != sr:
        harmony = librosa.resample(harmony, orig_sr=out_sr, target_sr=sr)

    # Ensure it exactly matches the original length (pad or truncate if necessary)
    if len(harmony) < n_samples:
        harmony = np.pad(harmony, (0, n_samples - len(harmony)))
    elif len(harmony) > n_samples:
        harmony = harmony[:n_samples]

    # Normalize to prevent clipping (before remixing percussive)
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    # ---------------------------------------------------------
    # Reconstruct Final Audio
    # ---------------------------------------------------------
    # We mix the cleanly pitch-shifted harmonic track back mathematically 
    # with the untouched, unshifted percussive noise.
    
    # Ensure percussive track matches the fixed harmony length
    if len(percussive) < len(harmony):
        percussive = np.pad(percussive, (0, len(harmony) - len(percussive)))
    elif len(percussive) > len(harmony):
        percussive = percussive[:len(harmony)]

    harmony = harmony + percussive

    # Normalize to prevent clipping
    peak = np.max(np.abs(harmony))
    if peak > 0:
        harmony = harmony / peak

    return np.array(harmony, dtype=np.float32)
