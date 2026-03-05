"""Main application window for HarmonyAI."""

import os
import sys
import threading
import tempfile

import numpy as np
import soundfile as sf
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QRadioButton, QButtonGroup,
    QSlider, QFileDialog, QProgressBar, QApplication,
)
from PySide6.QtCore import Qt, Signal, QObject, QUrl, QTimer
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui.waveform_widget import WaveformWidget
from audio.loader import load_audio, save_audio, DEFAULT_SR
from audio.pitch import detect_pitch
from audio.key_detect import detect_key
from audio.harmony import generate_harmony
from audio.mixer import mix_audio
from music.scales import get_all_keys
from music.theory import HARMONY_TYPES


class WorkerSignals(QObject):
    """Signals for background processing."""
    progress = Signal(int, str)  # percent, message
    finished = Signal(dict)      # result data
    error = Signal(str)          # error message


class MainWindow(QMainWindow):
    """HarmonyAI main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎤 HarmonyAI — Vocal Harmony Generator")
        self.setMinimumSize(700, 680)
        self.resize(780, 750)

        # State
        self._audio = None           # Original audio (numpy)
        self._audio_sr = DEFAULT_SR  # Sample rate
        self._harmony = None         # Generated harmony (numpy)
        self._f0 = None              # Pitch contour
        self._times = None           # Pitch contour timestamps
        self._detected_key = None    # Auto-detected key string
        self._source_path = None     # Original file path

        # Audio playback
        self._player = QMediaPlayer()
        self._audio_output = QAudioOutput()
        self._player.setAudioOutput(self._audio_output)
        self._audio_output.setVolume(0.8)
        self._temp_dir = tempfile.mkdtemp()

        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

        self._build_ui()
        self.setAcceptDrops(True)

    def _build_ui(self):
        """Construct the entire UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #0f0f1a;")

        layout = QVBoxLayout(central)
        layout.setContentsMargins(28, 24, 28, 20)
        layout.setSpacing(16)

        # ── Title ──
        title = QLabel("🎤 HarmonyAI")
        title.setStyleSheet("font-size: 28px; font-weight: 700; color: #c4b5fd; letter-spacing: -1px;")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        subtitle = QLabel("Drop a vocal file. Get instant harmonies.")
        subtitle.setStyleSheet("font-size: 13px; color: rgba(255,255,255,0.35); margin-bottom: 4px;")
        layout.addWidget(subtitle)

        # ── Drop Zone ──
        self._drop_zone = QWidget()
        self._drop_zone.setObjectName("dropZone")
        self._drop_zone.setFixedHeight(100)
        drop_layout = QVBoxLayout(self._drop_zone)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._drop_label = QLabel("Drop audio file here — or click to browse")
        self._drop_label.setObjectName("dropLabel")
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self._drop_label)

        drop_sub = QLabel("WAV · MP3 · FLAC · OGG")
        drop_sub.setObjectName("dropSubLabel")
        drop_sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_sub)

        self._drop_zone.mousePressEvent = self._browse_file
        layout.addWidget(self._drop_zone)

        # ── Waveform ──
        self._waveform = WaveformWidget()
        self._waveform.setFixedHeight(160)
        layout.addWidget(self._waveform)

        # ── Key Detection Row ──
        key_row = QHBoxLayout()
        key_row.setSpacing(12)

        key_header = QLabel("KEY")
        key_header.setProperty("class", "sectionHeader")
        key_header.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
        key_row.addWidget(key_header)

        self._key_label = QLabel("—")
        self._key_label.setObjectName("keyLabel")
        key_row.addWidget(self._key_label)

        self._key_confidence = QLabel("")
        self._key_confidence.setObjectName("keyConfidence")
        self._key_confidence.setStyleSheet("color: rgba(255,255,255,0.3); font-size: 11px;")
        key_row.addWidget(self._key_confidence)

        key_row.addStretch()

        override_label = QLabel("Override:")
        override_label.setStyleSheet("color: rgba(255,255,255,0.35); font-size: 11px;")
        key_row.addWidget(override_label)

        self._key_combo = QComboBox()
        self._key_combo.addItem("Auto-detect")
        self._key_combo.addItems(get_all_keys())
        self._key_combo.setFixedWidth(140)
        key_row.addWidget(self._key_combo)

        layout.addLayout(key_row)

        # ── Harmony Type ──
        harmony_header = QLabel("HARMONY")
        harmony_header.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; font-weight: 700; letter-spacing: 1.5px;")
        layout.addWidget(harmony_header)

        harmony_row = QHBoxLayout()
        harmony_row.setSpacing(4)
        self._harmony_group = QButtonGroup(self)

        for i, name in enumerate(HARMONY_TYPES.keys()):
            rb = QRadioButton(name)
            if i == 0:
                rb.setChecked(True)
            self._harmony_group.addButton(rb, i)
            harmony_row.addWidget(rb)

        harmony_row.addStretch()
        layout.addLayout(harmony_row)

        # ── Sliders Row ──
        sliders_row = QHBoxLayout()
        sliders_row.setSpacing(24)

        # Original Volume
        orig_vol_layout = QVBoxLayout()
        orig_vol_label = QLabel("ORIGINAL VOL")
        orig_vol_label.setObjectName("sliderLabel")
        orig_vol_label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        orig_vol_layout.addWidget(orig_vol_label)

        orig_vol_h = QHBoxLayout()
        self._orig_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._orig_vol_slider.setRange(0, 100)
        self._orig_vol_slider.setValue(100)
        self._orig_vol_slider.setFixedWidth(160)
        orig_vol_h.addWidget(self._orig_vol_slider)
        self._orig_vol_value = QLabel("100%")
        self._orig_vol_value.setObjectName("sliderValue")
        self._orig_vol_value.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 12px; min-width: 35px;")
        orig_vol_h.addWidget(self._orig_vol_value)
        orig_vol_layout.addLayout(orig_vol_h)
        sliders_row.addLayout(orig_vol_layout)

        self._orig_vol_slider.valueChanged.connect(
            lambda v: self._orig_vol_value.setText(f"{v}%")
        )

        # Harmony Volume
        vol_layout = QVBoxLayout()
        vol_label = QLabel("HARMONY VOL")
        vol_label.setObjectName("sliderLabel")
        vol_label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        vol_layout.addWidget(vol_label)

        vol_h = QHBoxLayout()
        self._vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._vol_slider.setRange(0, 100)
        self._vol_slider.setValue(70)
        self._vol_slider.setFixedWidth(160)
        vol_h.addWidget(self._vol_slider)
        self._vol_value = QLabel("70%")
        self._vol_value.setObjectName("sliderValue")
        self._vol_value.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 12px; min-width: 35px;")
        vol_h.addWidget(self._vol_value)
        vol_layout.addLayout(vol_h)
        sliders_row.addLayout(vol_layout)

        self._vol_slider.valueChanged.connect(
            lambda v: self._vol_value.setText(f"{v}%")
        )

        # Stereo Width
        width_layout = QVBoxLayout()
        width_label = QLabel("STEREO WIDTH")
        width_label.setObjectName("sliderLabel")
        width_label.setStyleSheet("color: rgba(255,255,255,0.4); font-size: 10px; font-weight: 700; letter-spacing: 1px;")
        width_layout.addWidget(width_label)

        width_h = QHBoxLayout()
        self._width_slider = QSlider(Qt.Orientation.Horizontal)
        self._width_slider.setRange(0, 100)
        self._width_slider.setValue(40)
        self._width_slider.setFixedWidth(160)
        width_h.addWidget(self._width_slider)
        self._width_value = QLabel("40%")
        self._width_value.setObjectName("sliderValue")
        self._width_value.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 12px; min-width: 35px;")
        width_h.addWidget(self._width_value)
        width_layout.addLayout(width_h)
        sliders_row.addLayout(width_layout)

        self._width_slider.valueChanged.connect(
            lambda v: self._width_value.setText(f"{v}%")
        )

        sliders_row.addStretch()
        layout.addLayout(sliders_row)

        # ── Progress Bar ──
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFixedHeight(6)
        self._progress.setTextVisible(False)
        self._progress.hide()
        layout.addWidget(self._progress)

        # ── Seek Bar ──
        seek_layout = QHBoxLayout()
        seek_layout.setSpacing(12)
        
        self._time_label = QLabel("0:00")
        self._time_label.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 11px;")
        seek_layout.addWidget(self._time_label)
        
        self._seek_slider = QSlider(Qt.Orientation.Horizontal)
        self._seek_slider.setRange(0, 0)
        self._seek_slider.setEnabled(False)
        self._seek_slider.sliderMoved.connect(self._on_seek)
        seek_layout.addWidget(self._seek_slider)
        
        self._duration_label = QLabel("0:00")
        self._duration_label.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 11px;")
        seek_layout.addWidget(self._duration_label)
        
        layout.addLayout(seek_layout)

        # ── Buttons Row ──
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._generate_btn = QPushButton("✨  Generate Harmony")
        self._generate_btn.setObjectName("generateBtn")
        self._generate_btn.setFixedHeight(44)
        self._generate_btn.setEnabled(False)
        self._generate_btn.clicked.connect(self._on_generate)
        btn_row.addWidget(self._generate_btn)

        self._play_btn = QPushButton("▶  Play Mix")
        self._play_btn.setFixedHeight(44)
        self._play_btn.setEnabled(False)
        self._play_btn.clicked.connect(self._on_play)
        btn_row.addWidget(self._play_btn)

        self._stop_btn = QPushButton("⏹  Stop")
        self._stop_btn.setFixedHeight(44)
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_row.addWidget(self._stop_btn)

        self._export_btn = QPushButton("💾  Export")
        self._export_btn.setObjectName("exportBtn")
        self._export_btn.setFixedHeight(44)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        btn_row.addWidget(self._export_btn)

        layout.addLayout(btn_row)

        # ── Status ──
        self._status = QLabel("Ready — drop a vocal file to begin")
        self._status.setObjectName("statusBar")
        layout.addWidget(self._status)

    # ── Drag & Drop ──

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._drop_zone.setStyleSheet(
                "#dropZone { border-color: rgba(167,139,250,0.7); background-color: rgba(30,25,55,0.9); }"
            )

    def dragLeaveEvent(self, event):
        self._drop_zone.setStyleSheet("")

    def dropEvent(self, event: QDropEvent):
        self._drop_zone.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_file(path)

    def _browse_file(self, _event=None):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"
        )
        if path:
            self._load_file(path)

    # ── File Loading ──

    def _load_file(self, path: str):
        """Load an audio file and run pitch detection."""
        self._status.setText(f"Loading: {os.path.basename(path)}")
        self._source_path = path
        self._harmony = None
        self._waveform.clear()
        self._progress.show()
        self._progress.setValue(10)
        QApplication.processEvents()

        # Run in background thread
        def process():
            signals = WorkerSignals()
            signals.progress.connect(self._on_progress)
            signals.finished.connect(self._on_load_finished)
            signals.error.connect(self._on_error)

            try:
                signals.progress.emit(20, "Loading audio...")
                audio, sr = load_audio(path)

                signals.progress.emit(50, "Detecting pitch...")
                f0, times, _ = detect_pitch(audio, sr)

                signals.progress.emit(80, "Detecting key...")
                key, confidence = detect_key(f0)

                signals.progress.emit(100, "Done")
                signals.finished.emit({
                    'audio': audio,
                    'sr': sr,
                    'f0': f0,
                    'times': times,
                    'key': key,
                    'confidence': confidence,
                    'filename': os.path.basename(path),
                })
            except Exception as e:
                signals.error.emit(str(e))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _on_progress(self, percent: int, message: str):
        self._progress.setValue(percent)
        self._status.setText(message)

    def _on_load_finished(self, result: dict):
        self._audio = result['audio']
        self._audio_sr = result['sr']
        self._f0 = result['f0']
        self._times = result['times']
        self._detected_key = result['key']

        self._waveform.set_original(self._audio)
        self._key_label.setText(result['key'])
        self._key_confidence.setText(f"({result['confidence']:.0%} confidence)")
        self._drop_label.setText(f"✅ {result['filename']}")
        self._generate_btn.setEnabled(True)
        self._status.setText(f"Loaded — {len(self._audio)/self._audio_sr:.1f}s — Key: {result['key']}")
        self._progress.hide()

    def _on_error(self, message: str):
        self._status.setText(f"❌ Error: {message}")
        self._progress.hide()

    # ── Harmony Generation ──

    def _get_selected_key(self) -> str:
        """Get the key to use (override or auto-detected)."""
        override = self._key_combo.currentText()
        if override != "Auto-detect":
            return override
        return self._detected_key or "C Major"

    def _get_selected_harmony(self) -> str:
        """Get the selected harmony type string."""
        btn = self._harmony_group.checkedButton()
        return btn.text() if btn else "Upper 3rd"

    def _on_generate(self):
        """Generate harmony in background thread."""
        if self._audio is None:
            return

        self._generate_btn.setEnabled(False)
        self._progress.show()
        self._progress.setValue(0)
        self._status.setText("Generating harmony...")

        key = self._get_selected_key()
        harmony_type = self._get_selected_harmony()

        def process():
            signals = WorkerSignals()
            signals.progress.connect(self._on_progress)
            signals.finished.connect(self._on_harmony_finished)
            signals.error.connect(self._on_error)

            try:
                signals.progress.emit(30, f"Generating {harmony_type} in {key}...")
                harmony = generate_harmony(
                    self._audio, self._audio_sr, self._f0, self._times,
                    key, harmony_type
                )
                signals.progress.emit(100, "Harmony generated!")
                signals.finished.emit({'harmony': harmony})
            except Exception as e:
                signals.error.emit(str(e))

        thread = threading.Thread(target=process, daemon=True)
        thread.start()

    def _on_harmony_finished(self, result: dict):
        self._harmony = result['harmony']
        self._waveform.set_harmony(self._harmony)
        self._generate_btn.setEnabled(True)
        self._play_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        self._progress.hide()
        self._status.setText("✅ Harmony ready — play or export")

    # ── Playback ──

    def _on_play(self):
        """Play the mixed audio."""
        # If already playing, don't restart, just ensure it's playing
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            return

        if self._audio is None or self._harmony is None:
            return

        orig_vol = self._orig_vol_slider.value() / 100.0
        harm_vol = self._vol_slider.value() / 100.0
        width = self._width_slider.value() / 100.0
        stereo = mix_audio(self._audio, self._harmony, harm_vol, orig_vol, width)

        # Output file logic
        temp_path = os.path.join(self._temp_dir, "preview.wav")
        save_audio(temp_path, stereo, self._audio_sr)

        # Only set source if it's not the same or if we are loaded new
        self._player.setSource(QUrl.fromLocalFile(temp_path))
        self._player.play()
        self._stop_btn.setEnabled(True)
        self._status.setText("▶ Playing mix...")

    def _on_stop(self):
        self._player.stop()
        self._stop_btn.setEnabled(False)
        self._seek_slider.setValue(0)
        self._time_label.setText("0:00")
        self._status.setText("⏹ Stopped")

    def _on_position_changed(self, position):
        if not self._seek_slider.isSliderDown():
            self._seek_slider.setValue(position)
        s = position // 1000
        m = s // 60
        s = s % 60
        self._time_label.setText(f"{m}:{s:02d}")

    def _on_duration_changed(self, duration):
        self._seek_slider.setRange(0, duration)
        self._seek_slider.setEnabled(duration > 0)
        s = duration // 1000
        m = s // 60
        s = s % 60
        self._duration_label.setText(f"{m}:{s:02d}")

    def _on_seek(self, position):
        self._player.setPosition(position)

    # ── Export ──

    def _on_export(self):
        """Export harmony or mix to file."""
        if self._harmony is None:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Audio", "harmony_mix.wav",
            "WAV Files (*.wav);;All Files (*)"
        )
        if not path:
            return

        orig_vol = self._orig_vol_slider.value() / 100.0
        harm_vol = self._vol_slider.value() / 100.0
        width = self._width_slider.value() / 100.0
        stereo = mix_audio(self._audio, self._harmony, harm_vol, orig_vol, width)
        save_audio(path, stereo, self._audio_sr)
        self._status.setText(f"💾 Exported to {os.path.basename(path)}")

    def closeEvent(self, event):
        self._player.stop()
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except:
            pass
        event.accept()
