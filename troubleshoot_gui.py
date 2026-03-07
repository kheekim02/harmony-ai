import sys
import numpy as np
import soundfile as sf
import librosa
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QMessageBox, QGroupBox, QFormLayout, QDoubleSpinBox, QListWidget)
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

import subprocess

from audio.pitch import detect_pitch
from audio.harmony import generate_harmony


class HarmonyWorker(QThread):
    finished = Signal(str)
    error = Signal(str)
    
    def __init__(self, audio_path, start_time, end_time, pitch_floor, hnr_thresh, med_win):
        super().__init__()
        self.audio_path = audio_path
        self.start_time = start_time
        self.end_time = end_time
        self.pitch_floor = pitch_floor
        self.hnr_thresh = hnr_thresh
        self.med_win = med_win
        
    def run(self):
        try:
            # 1. Load Audio
            audio, sr = sf.read(self.audio_path)
            if len(audio.shape) > 1: audio = audio.mean(axis=1)
            if sr != 22050: audio = librosa.resample(audio, orig_sr=sr, target_sr=22050); sr=22050
            
            # Crop audio based on time
            start_sample = int(self.start_time * sr)
            end_sample = int(self.end_time * sr) if self.end_time > self.start_time else len(audio)
            
            # Ensure within bounds
            start_sample = max(0, min(start_sample, len(audio) - 1))
            end_sample = max(start_sample + int(sr*0.5), min(end_sample, len(audio)))
            
            cropped_audio = audio[start_sample:end_sample]
            
            # 2. Detect Pitch with params
            f0, times, _ = detect_pitch(
                cropped_audio, sr, 
                fmin=self.pitch_floor, 
                fmax=1047.0, 
                hnr_threshold=self.hnr_thresh, 
                median_window=self.med_win
            )
            
            # 3. Generate Harmony (Upper 3rd)
            from audio.key_detect import detect_key
            valid_pitches = f0[~np.isnan(f0)]
            if len(valid_pitches) > 0:
                key, _ = detect_key(f0)
            else:
                key = "C Major"
                
            harmony = generate_harmony(cropped_audio, sr, f0, times, key, harmony_type="Upper 3rd")
            
            # Save to tmp
            out_path = "/tmp/troubleshoot_harmony.wav"
            sf.write(out_path, harmony, sr)
            self.finished.emit(out_path)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class TroubleshootGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HarmonyAI - Advanced Troubleshooter")
        self.setMinimumWidth(500)
        
        # Audio Player Process (Native Qt)
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(1.0)
        
        self.audio_path = "/Users/geoffrey/Desktop/audio_test.mp3"
        self.generated_audio_path = None
        
        # Setup UI
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # File selection
        file_layout = QHBoxLayout()
        self.file_label = QLabel(f"File: {self.audio_path}")
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(btn_browse)
        layout.addLayout(file_layout)
        
        # Parameters group
        param_group = QGroupBox("Target Region & Noise Parameters")
        param_layout = QFormLayout()
        
        # Time crop
        time_layout = QHBoxLayout()
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 3600)
        self.start_spin.setValue(170.0) # 2:50
        self.start_spin.setSingleStep(1.0)
        self.start_spin.setSuffix(" s")
        
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 3600)
        self.end_spin.setValue(210.0) # 3:30
        self.end_spin.setSingleStep(1.0)
        self.end_spin.setSuffix(" s")
        
        time_layout.addWidget(QLabel("Start:"))
        time_layout.addWidget(self.start_spin)
        time_layout.addWidget(QLabel("End:"))
        time_layout.addWidget(self.end_spin)
        param_layout.addRow("Audio Region (Crop):", time_layout)
        
        # 1. Pitch Floor
        self.floor_slider = QSlider(Qt.Horizontal)
        self.floor_slider.setRange(50, 200)
        self.floor_slider.setValue(65)
        self.floor_label = QLabel("65 Hz")
        self.floor_slider.valueChanged.connect(lambda v: self.floor_label.setText(f"{v} Hz"))
        param_layout.addRow("Pitch Floor (Avoids subharmonic rumble):", self.floor_slider)
        param_layout.addRow("", self.floor_label)
        
        # 2. HNR Threshold
        self.hnr_slider = QSlider(Qt.Horizontal)
        self.hnr_slider.setRange(0, 100) # 0.0 to 10.0
        self.hnr_slider.setValue(20)
        self.hnr_label = QLabel("2.0 dB")
        self.hnr_slider.valueChanged.connect(lambda v: self.hnr_label.setText(f"{v/10.0:.1f} dB"))
        param_layout.addRow("HNR Gate (Mutes raspy/noisy syllables):", self.hnr_slider)
        param_layout.addRow("", self.hnr_label)
        
        # 3. Median Window
        self.median_slider = QSlider(Qt.Horizontal)
        self.median_slider.setRange(1, 20) # 10 to 200 ms -> 1*10 to 20*10 -> 11 to 201 frames (must be odd)
        self.median_slider.setValue(5) # 51 frames
        self.median_label = QLabel("51 frames (510ms)")
        self.median_slider.valueChanged.connect(lambda v: self.median_label.setText(f"{v*10 + 1} frames ({(v*10+1)*10}ms)"))
        param_layout.addRow("Subharmonic Drop Filter Window:", self.median_slider)
        param_layout.addRow("", self.median_label)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Action Buttons
        self.status_label = QLabel("Ready.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)
        
        btn_layout = QHBoxLayout()
        
        self.btn_generate = QPushButton("Generate Harmony")
        self.btn_generate.setFixedHeight(40)
        self.btn_generate.setStyleSheet("background-color: #a78bfa; color: white; font-weight: bold;")
        self.btn_generate.clicked.connect(self.start_generation)
        btn_layout.addWidget(self.btn_generate)
        
        self.btn_play = QPushButton("▶ Play Harmony")
        self.btn_play.setFixedHeight(40)
        self.btn_play.setStyleSheet("background-color: #34d399; color: black; font-weight: bold;")
        self.btn_play.clicked.connect(self.play_audio)
        self.btn_play.setEnabled(False) # Only enable after generation
        btn_layout.addWidget(self.btn_play)
        
        layout.addLayout(btn_layout)
        
        self.btn_stop = QPushButton("Stop Playback")
        self.btn_stop.clicked.connect(self.stop_playback)
        layout.addWidget(self.btn_stop)
        
        # Artifact Logger
        logger_group = QGroupBox("Artifact Logger (Bookmark tearing spots)")
        logger_layout = QVBoxLayout()
        
        self.marker_list = QListWidget()
        self.marker_list.setFixedHeight(60)
        logger_layout.addWidget(self.marker_list)
        
        logger_btns = QHBoxLayout()
        self.btn_log = QPushButton("📌 Log Current Start Time")
        self.btn_log.clicked.connect(self.log_marker)
        
        self.btn_export = QPushButton("💾 Export Log to Agent")
        self.btn_export.clicked.connect(self.export_log)
        
        logger_btns.addWidget(self.btn_log)
        logger_btns.addWidget(self.btn_export)
        logger_layout.addLayout(logger_btns)
        
        logger_group.setLayout(logger_layout)
        layout.addWidget(logger_group)
        
    def log_marker(self):
        # Base start time of the crop window
        base_t = self.start_spin.value()
        
        # If playing, add the real-time playback position
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            current_pos_sec = self.player.position() / 1000.0
            actual_time = base_t + current_pos_sec
            self.marker_list.addItem(f"{actual_time:.2f}")
        else:
            # If not playing, just log the start of the crop
            self.marker_list.addItem(f"{base_t:.2f}")
        
    def export_log(self):
        items = [self.marker_list.item(i).text() for i in range(self.marker_list.count())]
        if not items:
            QMessageBox.warning(self, "Empty", "No markers to export.")
            return
            
        out_path = "/tmp/artifact_markers.txt"
        try:
            with open(out_path, "w") as f:
                f.write("\n".join(items))
            QMessageBox.information(self, "Exported", f"Saved {len(items)} markers to {out_path}.\n\nPlease tell the Agent you have exported them!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save:\n{e}")
            
    def stop_playback(self):
        self.player.stop()
        
    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Audio", "", "Audio Files (*.wav *.mp3 *.flac)")
        if file_name:
            self.audio_path = file_name
            self.file_label.setText(f"File: {self.audio_path}")

    def start_generation(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Error", "No audio file selected.")
            return
            
        self.btn_generate.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.status_label.setText("Processing audio... Please wait (this may take a minute).")
        self.status_label.setStyleSheet("color: #fb923c; font-weight: bold;")
        
        # Get Params
        p_floor = float(self.floor_slider.value())
        hnr_thresh = self.hnr_slider.value() / 10.0
        med_win = self.median_slider.value() * 10 + 1
        start_time = float(self.start_spin.value())
        end_time = float(self.end_spin.value())
        
        # Start Thread
        self.worker = HarmonyWorker(self.audio_path, start_time, end_time, p_floor, hnr_thresh, med_win)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.error.connect(self.on_generation_error)
        self.worker.start()

    def on_generation_finished(self, out_path):
        self.generated_audio_path = out_path
        self.status_label.setText(f"Done! Harmony saved locally. Ready to play.")
        self.status_label.setStyleSheet("color: #34d399; font-weight: bold;")
        self.btn_generate.setEnabled(True)
        self.btn_play.setEnabled(True)
        
    def on_generation_error(self, err_msg):
        QMessageBox.critical(self, "Generation Error", err_msg)
        self.status_label.setText("Error during generation.")
        self.status_label.setStyleSheet("color: #ef4444; font-weight: bold;")
        self.btn_generate.setEnabled(True)

    def play_audio(self):
        if self.generated_audio_path:
            self.stop_playback()
            self.player.setSource(QUrl.fromLocalFile(self.generated_audio_path))
            self.player.play()
            self.status_label.setText("Playing harmony...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TroubleshootGUI()
    window.show()
    sys.exit(app.exec())
