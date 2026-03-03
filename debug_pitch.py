import sys
import numpy as np
import parselmouth
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != 22050:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        sr = 22050
    return audio, sr

def detect_pitch(audio: np.ndarray, sr: int):
    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65.0, pitch_ceiling=1047.0)
    
    f0 = pitch.selected_array['frequency']
    times = pitch.xs()
    
    # Pre-process like our real pipeline
    voiced_flag = f0 > 0.0
    f0_nan = np.copy(f0)
    f0_nan[~voiced_flag] = np.nan
    
    return f0_nan, times

def debug_plot(audio, sr, f0, times, output_path):
    plt.figure(figsize=(15, 8))
    
    # Plot 1: Audio Waveform
    plt.subplot(2, 1, 1)
    audio_times = np.arange(len(audio)) / sr
    plt.plot(audio_times, audio, color='royalblue', alpha=0.6, label='Audio Waveform')
    plt.title("Vocal Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Pitch Tracking
    plt.subplot(2, 1, 2)
    plt.scatter(times, f0, color='crimson', s=10, label='Detected Pitch (Hz) - Autocorrelation')
    
    # Optionally overlay a smoothed version to see if smoothing is failing
    smoothed = np.copy(f0)
    # Simple median filter
    for i in range(2, len(f0) - 2):
        chunk = f0[i-2:i+3]
        valid_chunk = chunk[~np.isnan(chunk)]
        if len(valid_chunk) > 0:
            smoothed[i] = np.median(valid_chunk)
            
    plt.plot(times, smoothed, color='blue', alpha=0.7, label='Smoothed Median Pitch')
    
    plt.title("Pitch Detection (Praat) vs Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(max(60, np.nanmin(f0)*0.8), min(1000, np.nanmax(f0)*1.2))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✅ Saved visual debugger plot to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_pitch.py <path_to_vocal.wav>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    output_file = "/tmp/pitch_debug_plot.png"
    
    print(f"Loading {input_file}...")
    audio, sr = load_audio(input_file)
    
    print("Detecting pitch via Praat...")
    f0, times = detect_pitch(audio, sr)
    
    print("Generating visual plots...")
    debug_plot(audio, sr, f0, times, output_file)
