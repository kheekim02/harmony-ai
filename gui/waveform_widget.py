"""Custom waveform display widget using QPainter."""

import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient


class WaveformWidget(QWidget):
    """Dual waveform display showing original and harmony audio."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("waveformWidget")
        self.setMinimumHeight(160)

        self._original: np.ndarray | None = None
        self._harmony: np.ndarray | None = None
        self._downsample = 512  # samples per pixel

    def set_original(self, audio: np.ndarray):
        """Set the original audio waveform."""
        self._original = audio
        self.update()

    def set_harmony(self, audio: np.ndarray):
        """Set the harmony audio waveform."""
        self._harmony = audio
        self.update()

    def clear(self):
        """Clear both waveforms."""
        self._original = None
        self._harmony = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(15, 15, 30, 153))

        # Draw border
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        painter.drawRoundedRect(0, 0, w - 1, h - 1, 12, 12)

        if self._original is None:
            # Empty state label
            painter.setPen(QColor(255, 255, 255, 50))
            painter.drawText(QRect(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, "No audio loaded")
            painter.end()
            return

        half_h = h // 2

        # Draw original waveform (top half, blue)
        self._draw_waveform(
            painter, self._original, 0, half_h,
            color_start=QColor(96, 165, 250, 180),
            color_end=QColor(96, 165, 250, 40),
            label="Original"
        )

        # Separator line
        painter.setPen(QPen(QColor(255, 255, 255, 20), 1, Qt.PenStyle.DashLine))
        painter.drawLine(12, half_h, w - 12, half_h)

        # Draw harmony waveform (bottom half, green)
        if self._harmony is not None:
            self._draw_waveform(
                painter, self._harmony, half_h, half_h,
                color_start=QColor(52, 211, 153, 180),
                color_end=QColor(52, 211, 153, 40),
                label="Harmony"
            )
        else:
            painter.setPen(QColor(255, 255, 255, 30))
            painter.drawText(
                QRect(0, half_h, w, half_h),
                Qt.AlignmentFlag.AlignCenter,
                "Generate harmony to see waveform"
            )

        painter.end()

    def _draw_waveform(self, painter, audio, y_offset, height, color_start, color_end, label):
        """Draw a single waveform in the given region."""
        w = self.width()
        padding = 16
        draw_w = w - padding * 2
        mid_y = y_offset + height // 2

        if draw_w <= 0 or len(audio) == 0:
            return

        # Label
        painter.setPen(QColor(255, 255, 255, 60))
        painter.drawText(padding, y_offset + 14, label)

        # Downsample for display
        samples_per_pixel = max(1, len(audio) // draw_w)
        n_points = min(draw_w, len(audio) // samples_per_pixel)

        if n_points < 2:
            return

        # Create gradient
        gradient = QLinearGradient(0, y_offset, 0, y_offset + height)
        gradient.setColorAt(0.0, color_start)
        gradient.setColorAt(0.5, color_end)
        gradient.setColorAt(1.0, color_start)

        pen = QPen(color_start, 1.2)
        painter.setPen(pen)

        max_amplitude = height * 0.35

        # Draw waveform as min/max envelope
        prev_x = padding
        for i in range(n_points):
            start_idx = i * samples_per_pixel
            end_idx = min(start_idx + samples_per_pixel, len(audio))
            chunk = audio[start_idx:end_idx]

            if len(chunk) == 0:
                continue

            min_val = np.min(chunk)
            max_val = np.max(chunk)

            x = padding + int(i * draw_w / n_points)
            y_top = int(mid_y - max_val * max_amplitude)
            y_bot = int(mid_y - min_val * max_amplitude)

            # Draw vertical line for this sample range
            painter.drawLine(x, y_top, x, y_bot)

        # Center line
        painter.setPen(QPen(QColor(255, 255, 255, 15), 1))
        painter.drawLine(padding, mid_y, padding + draw_w, mid_y)
