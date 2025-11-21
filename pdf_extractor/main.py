# -*- coding: utf-8 -*-
import sys
import os
import re
from PySide6.QtWidgets import QApplication

# Corrected relative imports for the new structure
from .ui.app_ui import MainWindow, DEFAULT_FONT_SIZE
from . import settings

def apply_stylesheet(app, font_size):
    """
    Reads the stylesheet, replaces font size placeholders, and applies it.
    """
    try:
        # Build path relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        stylesheet_path = os.path.join(script_dir, 'ui', 'assets', 'style.qss')
        
        with open(stylesheet_path, "r", encoding="utf-8") as f:
            style = f.read()
            
        # Replace placeholders
        large_size = font_size + 1
        small_size = font_size - 1

        style = style.replace("__FONT_SIZE_BASE__", str(font_size))
        style = style.replace("__FONT_SIZE_LARGE__", str(large_size))
        style = style.replace("__FONT_SIZE_SMALL__", str(small_size))
        
        app.setStyleSheet(style)
    except FileNotFoundError:
        print(f"Warning: style.qss not found at '{stylesheet_path}'. Using default styles.")

def main():
    """应用程序主入口"""
    app = QApplication(sys.argv)
    
    # Load settings to get initial font size
    s = settings.load_settings()
    font_size = s.get("font_size", DEFAULT_FONT_SIZE)
    apply_stylesheet(app, font_size)

    window = MainWindow()
    
    # Connect the font size change signal to the handler
    window.fontSizeChanged.connect(lambda size: apply_stylesheet(app, size))
    
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()