import librosa
import numpy as np
import scipy.signal
import pyworld as pw
import torch
from vocos import Vocos

from music.theory import compute_harmony_shift, HARMONY_TYPES
from music.scales import parse_key

# Global lazy-load cache so we only download/load the 50MB model once per app session
_VOCOS_MODEL = None

def get_vocos() -> Vocos:
    """Lazy-loads the Vocos HuggingFace neural model on first harmony generation."""
    global _VOCOS_MODEL
    if _VOCOS_MODEL is None:
        print("Initializing Neural Vocoder Pipeline...")
        _VOCOS_MODEL = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return _VOCOS_MODEL

def generate_harmony(
    audio: np.ndarray,
    sr: int,
    f0: np.ndarray,
    times: np.ndarray,
    key_string: str,
    harmony_type: str = 'Upper 3rd',
    hop_length: int = 512,
) -> np.ndarray:
    """Generate a harmony track from the original audio using a Neural Vocoder.

    Args:
        audio: Original mono audio array
        sr: Original Sample rate
        f0: Pitch contour from Praat (Hz, NaN for unvoiced)
        times: Array of timestamps for each frame
        key_string: Detected key like 'C Major'
        harmony_type: One of HARMONY_TYPES keys

    Returns:
        Harmony audio array (same length as input, at original sample rate)
    """
    root, scale_type = parse_key(key_string)
    interval = HARMONY_TYPES.get(harmony_type, 2)
    n_samples_orig = len(audio)

    # ---------------------------------------------------------
    # 1. Neural Vocoder Sample Rate Formatting (24000 Hz constraint)
    # ---------------------------------------------------------
    TARGET_SR = 24000
    if sr != TARGET_SR:
        audio_24k = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    else:
        audio_24k = np.copy(audio)
        
    audio_24k_f64 = audio_24k.astype(np.float64)

    # ---------------------------------------------------------
    # 2. PyWorld Feature Extraction (Phase 1)
    # ---------------------------------------------------------
    # We still use PyWorld to accurately track F0 and model formants
    _f0, t = pw.harvest(audio_24k_f64, TARGET_SR, f0_floor=65.0, f0_ceil=1047.0)
    sp = pw.cheaptrick(audio_24k_f64, _f0, t, TARGET_SR)
    ap = pw.d4c(audio_24k_f64, _f0, t, TARGET_SR)

    native_shifts = np.zeros_like(_f0)
    for i in range(len(_f0)):
        if _f0[i] > 0.0:
            native_shifts[i] = compute_harmony_shift(_f0[i], root, scale_type, interval)

    smoothed_shifts = scipy.signal.medfilt(native_shifts, kernel_size=15)
    f0_smooth = scipy.signal.medfilt(_f0, kernel_size=5)

    # Calculate actual shifted pitch contour
    shift_multipliers = 2.0 ** (smoothed_shifts / 12.0)
    f0_shifted = f0_smooth * shift_multipliers

    # Calculate metallic audio using PyWorld
    metallic_audio_24k = pw.synthesize(f0_shifted, sp, ap, TARGET_SR).astype(np.float32)
    
    peak = np.max(np.abs(metallic_audio_24k))
    if peak > 0.99:
        metallic_audio_24k = metallic_audio_24k / peak
        
    # ---------------------------------------------------------
    # 3. Neural Hallucination Phase Bottleneck (The Vocos Overhaul)
    # ---------------------------------------------------------
    # To fix PyWorld's mathematical tearing, we completely strip the phase
    # data by funneling the PyWorld audio into a Mel-Spectrogram, then
    # use Neural Vocoder AI to hallucinate a flawless human phase back in.
    
    vocoder = get_vocos()
    
    # 1. Convert to torch tensor with shape [B, C, T] -> [1, length]
    audio_tensor = torch.from_numpy(metallic_audio_24k).unsqueeze(0)
    
    # 2. Extract Mel-Spectrogram features (destroys the metallic PyWorld phase)
    with torch.no_grad():
        mel_features = vocoder.feature_extractor(audio_tensor)
        
        # 3. Decode features via Deep Learning into flawless acoustic waveform
        neural_audio_tensor = vocoder.decode(mel_features)
        
    harmony_24k = neural_audio_tensor.squeeze().cpu().numpy()

    # ---------------------------------------------------------
    # 4. Final Output Formatting
    # ---------------------------------------------------------
    # Resample the AI generation back to exactly match the DAW/WAV original sr
    if sr != TARGET_SR:
        harmony_final = librosa.resample(harmony_24k, orig_sr=TARGET_SR, target_sr=sr)
    else:
        harmony_final = harmony_24k

    # Precise sample-clipping constraint
    if len(harmony_final) < n_samples_orig:
        harmony_final = np.pad(harmony_final, (0, n_samples_orig - len(harmony_final)))
    elif len(harmony_final) > n_samples_orig:
        harmony_final = harmony_final[:n_samples_orig]

    # Normalize output
    peak = np.max(np.abs(harmony_final))
    if peak > 0:
        harmony_final = harmony_final / peak

    return harmony_final.astype(np.float32)
