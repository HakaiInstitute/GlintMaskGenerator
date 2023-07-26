import sys
from os import path


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base_path = getattr(sys, "_MEIPASS", path.dirname(__file__))
    return path.abspath(path.join(base_path, relative_path))
