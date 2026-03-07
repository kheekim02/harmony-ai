import sys
import os
import numpy as np
import soundfile as sf
import parselmouth
import matplotlib.pyplot as plt
import librosa

def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    return audio, sr

def extract_features(audio, sr, timestamp, window_size=2.0):
    # Process 2 seconds around the artifact
    start_time = max(0, timestamp - window_size/2)
    end_time = min(len(audio)/sr, timestamp + window_size/2)
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    chunk = audio[start_sample:end_sample]
    
    sound = parselmouth.Sound(chunk, sampling_frequency=sr)
    
    # 1. Pitch
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=1047)
    f0 = pitch.selected_array['frequency']
    times = pitch.xs() + start_time
    
    # 2. Harmonics-to-Noise Ratio (HNR)
    harmonicity = sound.to_harmonicity_cc(time_step=0.01, minimum_pitch=65)
    hnr = harmonicity.values[0]
    hnr_times = harmonicity.xs() + start_time
    
    # 3. RMS Energy
    rms = librosa.feature.rms(y=chunk, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512) + start_time
    
    return chunk, times, f0, hnr_times, hnr, rms_times, rms, start_time, end_time

def main():
    markers_file = "/tmp/artifact_markers.txt"
    audio_file = "/Users/geoffrey/Desktop/audio_test.mp3"
    out_dir = "/tmp/artifact_analysis"
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(markers_file):
        print("Marker file not found.")
        return
        
    with open(markers_file, 'r') as f:
        markers = [float(line.strip()) for line in f.readlines() if line.strip()]
        
    print(f"Loaded {len(markers)} markers.")
    print(f"Loading {audio_file}...")
    audio, sr = load_audio(audio_file)
    
    # Analyze the first 10 markers to avoid generating too many plots
    process_count = min(10, len(markers))
    
    print(f"Generating spectral analysis for the first {process_count} markers...")
    
    for i, marker in enumerate(markers[:process_count]):
        print(f"Processing marker {i+1}/{process_count} around {marker}s...")
        chunk, t_pitch, f0, t_hnr, hnr, t_rms, rms, t_start, t_end = extract_features(audio, sr, marker, window_size=1.0)
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Acoustic Failure Analysis: Frame {marker}s", fontsize=16)
        
        # Plot 1: Waveform & Target
        plt.subplot(3, 1, 1)
        t_audio = np.linspace(t_start, t_end, len(chunk))
        plt.plot(t_audio, chunk, color='lightgray', label='Waveform')
        plt.axvline(x=marker, color='red', linestyle='--', linewidth=2, label='Reported Artifact')
        plt.title("Waveform (Red = Teardown Point)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: F0 Contour
        plt.subplot(3, 1, 2)
        f0_valid = np.copy(f0)
        f0_valid[f0_valid == 0] = np.nan
        plt.plot(t_pitch, f0_valid, 'b.', label='Raw F0 (Hz)')
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        plt.title("Fundamental Frequency (Pitch Contour)")
        plt.ylabel("Hz")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: HNR (Harmonics to Noise)
        plt.subplot(3, 1, 3)
        plt.plot(t_hnr, hnr, 'g-', label='HNR (dB)')
        plt.axhline(y=4.3, color='orange', linestyle='--', label='Current Threshold (4.3dB)')
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        plt.title("Harmonics-to-Noise Ratio (Voiced vs Breathy/Fry)")
        plt.ylabel("dB")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"analysis_marker_{marker}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    print(f"✅ Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
