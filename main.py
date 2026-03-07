#!/usr/bin/env python3
"""HarmonyAI — Auto Vocal Harmony Generator.

A standalone desktop app that generates musically-correct vocal harmonies
from audio input. 100% offline, no cloud, no API keys.
"""

import os
import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QIcon
from gui.main_window import MainWindow
from gui.styles import STYLESHEET


def main():
    app = QApplication(sys.argv)

    # Apply global stylesheet
    app.setStyleSheet(STYLESHEET)

    # Set default font
    font = QFont("Inter", 13)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    
    # Set window icon
    icon_path = os.path.join(os.path.dirname(__file__), "gui", "assets", "icon.png")
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
        app.setWindowIcon(QIcon(icon_path))
        
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
