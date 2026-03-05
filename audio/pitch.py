"""Pitch detection using librosa's pYIN algorithm."""

import numpy as np
import parselmouth

def detect_pitch(
    audio: np.ndarray,
    sr: int,
    fmin: float = 101.0,  # Optimized for vocal fry
    fmax: float = 1047.0, # C6
    hnr_threshold: float = 5.0, # dB (Tightened gate after acoustic analysis)
    median_window: int = 201, # frames (Aggressive subharmonic drop filter)
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect pitch using Praat's autocorrelation and apply custom smoothing.

    Args:
        audio: Mono audio array
        sr: Sample rate
        fmin: Minimum expected frequency
        fmax: Maximum expected frequency
        hnr_threshold: Minimum Harmonics-to-Noise Ratio (dB) for a frame to be considered voiced.
                        Frames below this threshold are marked as unvoiced.
        median_window: Size of the median filter window (in frames) for subharmonic drop removal.

        tuple: (f0, times, hnr_contour)
        - f0: array of fundamental frequencies in Hz (NaN for unvoiced)
        - times: array of timestamps for each frame
        - hnr_contour: Harmonics-to-Noise Ratio (dB) for each frame
    """
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    
    # Extract pitch
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=fmin, pitch_ceiling=fmax)
    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    
    # Extract Harmonics-to-Noise Ratio (HNR)
    # This detects noisy frames (breaths, hard consonants, vocal fry) that shouldn't be pitch-shifted
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=fmin)
    hnr = harmonicity.values[0]
    hnr_times = harmonicity.xs()
    
    # Interpolate HNR to match the exact timestamps of the pitch contour
    interp_hnr = np.interp(times, hnr_times, hnr)
    
    # Praat returns 0.0 for unvoiced frames. Convert to np.nan. 
    # Also mask out frames that are too noisy based on the dynamic HNR threshold.
    voiced_flag = (f0 > 0.0) & (interp_hnr >= hnr_threshold)
    f0[~voiced_flag] = np.nan
    
    # ---------------------------------------------------------
    # Subharmonic Drop Removal (Vocal Fry Filter)
    # ---------------------------------------------------------
    # Autocorrelation sometimes locks onto the subharmonic (octave drop) during vocal fry.
    # We use a large rolling median filter to enforce a smooth continuous melody line,
    # but ONLY apply it if the detected pitch plunges significantly (e.g. > 6 semitones).
    
    # Run a large rolling window median filter, but ONLY apply it if the 
    # current pitch deviates from the local median by more than a tritone (6 semitones).
    smoothed_f0 = np.copy(f0)
    half_win = median_window // 2
    
    for i in range(len(f0)):
        if not np.isnan(f0[i]):
            start_idx = max(0, i - half_win)
            end_idx = min(len(f0), i + half_win + 1)
            
            window = f0[start_idx:end_idx]
            valid_window = window[~np.isnan(window)]
            
            if len(valid_window) > 3: # Ensure enough valid points for a meaningful median
                local_median = np.median(valid_window)
                
                # If the current pitch is outside 5.5 semitones of the local median,
                # it is almost certainly a subharmonic tracking error (fry drop).
                ratio = f0[i] / local_median
                semitones_diff = 12.0 * np.log2(ratio)
                
                if abs(semitones_diff) > 5.5: 
                    smoothed_f0[i] = local_median

    # Replace original f0 with the smoothed contour
    f0 = smoothed_f0
    
    # Instead of mock probabilities, return the HNR contour so we can use it 
    # for dynamic EQ gating during harmony generation.
    return f0, times, interp_hnr
