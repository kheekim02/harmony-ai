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

def extract_features(audio, sr, timestamp, window_size=1.0):
    start_time = max(0, timestamp - window_size/2)
    end_time = min(len(audio)/sr, timestamp + window_size/2)
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    
    # Extract overall chunk
    chunk = audio[start_sample:end_sample]
    
    # Extract HPSS
    harmonic, percussive = librosa.effects.hpss(chunk, margin=2.0)
    
    # Pitch detect just the harmonic part
    sound = parselmouth.Sound(harmonic, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01, pitch_floor=65, pitch_ceiling=1047)
    f0 = pitch.selected_array['frequency']
    times = pitch.xs() + start_time
    
    # Run a mock PSOLA synthesis natively just to see the output waveform
    manipulation = parselmouth.praat.call(sound, "To Manipulation", 0.01, 75.0, 600.0)
    pitch_tier = parselmouth.praat.call(manipulation, "Extract pitch tier")
    parselmouth.praat.call([pitch_tier, manipulation], "Replace pitch tier")
    
    duration = parselmouth.praat.call(sound, "Get total duration")
    new_pitch_tier = parselmouth.praat.call("Create PitchTier", "harmony", 0.0, duration)
    
    # Shift up a major 3rd (4 semitones)
    shift_hz = 2.0 ** (4.0 / 12.0)
    for i, t in enumerate(pitch.xs()):
        if f0[i] > 0 and not np.isnan(f0[i]):
            new_f0 = f0[i] * shift_hz
            parselmouth.praat.call(new_pitch_tier, "Add point", t, new_f0)
            
    parselmouth.praat.call([new_pitch_tier, manipulation], "Replace pitch tier")
    harmony_sound = parselmouth.praat.call(manipulation, "Get resynthesis (overlap-add)")
    harmony_chunk = harmony_sound.values[0]
    
    return chunk, harmonic, percussive, harmony_chunk, times, f0, start_time, end_time

def main():
    markers_file = "/tmp/artifact_markers.txt"
    audio_file = "/Users/geoffrey/Desktop/audio_test.mp3"
    out_dir = "/tmp/artifact_analysis_hpss"
    
    os.makedirs(out_dir, exist_ok=True)
    
    if not os.path.exists(markers_file):
        print("Marker file not found.")
        return
        
    with open(markers_file, 'r') as f:
        markers = [float(line.strip()) for line in f.readlines() if line.strip()]
        
    print(f"Loaded {len(markers)} markers.")
    print(f"Loading {audio_file}...")
    audio, sr = load_audio(audio_file)
    
    process_count = min(10, len(markers))
    print(f"Generating HPSS spectral analysis for {process_count} markers...")
    
    for i, marker in enumerate(markers[:process_count]):
        print(f"Processing marker {i+1}/{process_count} around {marker}s...")
        chunk, harmonic, percussive, harmony_chunk, t_pitch, f0, t_start, t_end = extract_features(audio, sr, marker, window_size=1.0)
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"HPSS Validation Analysis: Frame {marker}s", fontsize=16)
        
        t_audio = np.linspace(t_start, t_end, len(chunk))
        
        # Plot 1: Original vs Percussive vs Harmonic
        plt.subplot(3, 1, 1)
        plt.plot(t_audio, chunk, color='lightgray', label='Original Waveform', alpha=0.5)
        plt.plot(t_audio, harmonic, color='blue', label='HPSS Harmonic (Feed to PSOLA)', alpha=0.8)
        plt.axvline(x=marker, color='red', linestyle='--', linewidth=2, label='Reported Artifact')
        plt.title("HPSS Separation Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Final Shifted Harmony Output
        plt.subplot(3, 1, 2)
        plt.plot(t_audio, harmony_chunk[:len(t_audio)], color='orange', label='PSOLA Shifted Harmony')
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        plt.title("Synthesized Harmony (To Visualize Tearing)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Harmonic Pitch Contour
        plt.subplot(3, 1, 3)
        f0_valid = np.copy(f0)
        f0_valid[f0_valid == 0] = np.nan
        plt.plot(t_pitch, f0_valid, 'b.', label='Harmonic F0 (Hz)')
        plt.axvline(x=marker, color='red', linestyle='--', alpha=0.5)
        plt.title("Pitch Contour of the Harmonic Extract")
        plt.ylabel("Hz")
        plt.xlabel("Time (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"hpss_analysis_{marker}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        
    print(f"✅ Saved plots to {out_dir}")

if __name__ == "__main__":
    main()
