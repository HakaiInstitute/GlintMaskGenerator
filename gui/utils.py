"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-16
Description:
"""

import os
import sys


def resource_path(relative_path):
    """
    Define function to import external files when using PyInstaller.
    Get absolute path to resource, works for dev and for PyInstaller
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath("resources")

    return os.path.join(base_path, relative_path)
