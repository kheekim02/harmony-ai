"""Dark theme Qt stylesheet for HarmonyAI."""

STYLESHEET = """
QMainWindow {
    background-color: #0f0f1a;
}

QWidget {
    background-color: transparent;
    color: #e0e0e0;
    font-family: 'Inter', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}

/* Drop zone */
#dropZone {
    background-color: rgba(20, 20, 35, 0.8);
    border: 2px dashed rgba(167, 139, 250, 0.3);
    border-radius: 16px;
    min-height: 120px;
}

#dropZone:hover {
    border-color: rgba(167, 139, 250, 0.6);
    background-color: rgba(25, 25, 45, 0.9);
}

#dropLabel {
    color: rgba(255, 255, 255, 0.5);
    font-size: 15px;
    font-weight: 500;
}

#dropSubLabel {
    color: rgba(255, 255, 255, 0.25);
    font-size: 11px;
}

/* Section headers */
.sectionHeader {
    color: rgba(255, 255, 255, 0.4);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 4px 0;
}

/* Key display */
#keyLabel {
    color: #a78bfa;
    font-size: 18px;
    font-weight: 700;
}

#keyConfidence {
    color: rgba(255, 255, 255, 0.3);
    font-size: 11px;
}

/* Combo box (key override) */
QComboBox {
    background-color: rgba(30, 30, 50, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 6px 12px;
    color: #e0e0e0;
    min-width: 120px;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox::down-arrow {
    image: none;
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1a1a2e;
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    color: #e0e0e0;
    selection-background-color: rgba(167, 139, 250, 0.3);
    padding: 4px;
}

/* Radio buttons */
QRadioButton {
    color: rgba(255, 255, 255, 0.7);
    spacing: 8px;
    padding: 6px 12px;
    font-size: 13px;
}

QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid rgba(255, 255, 255, 0.2);
    background-color: transparent;
}

QRadioButton::indicator:checked {
    background-color: #a78bfa;
    border-color: #a78bfa;
}

QRadioButton::indicator:hover {
    border-color: rgba(167, 139, 250, 0.5);
}

/* Sliders */
QSlider::groove:horizontal {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
}

QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -6px 0;
    border-radius: 8px;
    background: #a78bfa;
}

QSlider::handle:horizontal:hover {
    background: #c4b5fd;
}

QSlider::sub-page:horizontal {
    background: rgba(167, 139, 250, 0.4);
    border-radius: 2px;
}

/* Buttons */
QPushButton {
    background-color: rgba(30, 30, 50, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    padding: 10px 20px;
    color: #e0e0e0;
    font-weight: 600;
    font-size: 13px;
}

QPushButton:hover {
    background-color: rgba(40, 40, 65, 0.9);
    border-color: rgba(255, 255, 255, 0.15);
}

QPushButton:pressed {
    background-color: rgba(25, 25, 40, 0.9);
}

QPushButton:disabled {
    color: rgba(255, 255, 255, 0.2);
    border-color: rgba(255, 255, 255, 0.05);
}

#generateBtn {
    background-color: rgba(167, 139, 250, 0.2);
    border-color: rgba(167, 139, 250, 0.3);
    color: #c4b5fd;
}

#generateBtn:hover {
    background-color: rgba(167, 139, 250, 0.3);
    border-color: rgba(167, 139, 250, 0.5);
}

#exportBtn {
    background-color: rgba(52, 211, 153, 0.15);
    border-color: rgba(52, 211, 153, 0.3);
    color: #6ee7b7;
}

#exportBtn:hover {
    background-color: rgba(52, 211, 153, 0.25);
    border-color: rgba(52, 211, 153, 0.5);
}

/* Status bar */
#statusBar {
    color: rgba(255, 255, 255, 0.3);
    font-size: 11px;
    padding: 8px 0;
}

/* Waveform */
#waveformWidget {
    background-color: rgba(15, 15, 30, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
}

/* Progress bar */
QProgressBar {
    background-color: rgba(255, 255, 255, 0.05);
    border: none;
    border-radius: 4px;
    height: 6px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #a78bfa, stop:1 #7c6aef);
    border-radius: 4px;
}

/* Slider labels */
#sliderValue {
    color: rgba(255, 255, 255, 0.5);
    font-size: 12px;
    min-width: 30px;
}

#sliderLabel {
    color: rgba(255, 255, 255, 0.4);
    font-size: 11px;
    font-weight: 600;
}
"""
