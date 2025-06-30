import sys
from pathlib import Path


def resource_path(relative_path: str) -> str:
    """Get the absolute path to a resource.

    Works for dev and for PyInstaller.
    """
    base_path = getattr(sys, "_MEIPASS", Path(__file__).parent)
    return str((Path(base_path) / relative_path).resolve())
