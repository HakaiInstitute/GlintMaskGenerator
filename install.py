# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Generate a executable file for the CLI and GUI interface using PyInstaller.
#   Requires pyinstaller installation.

import PyInstaller.__main__
from os import path

if __name__ == '__main__':
    PyInstaller.__main__.run([
        '--name=glint-mask-generator',
        '--onefile',
        '--clean',
        # '--console',
        '--windowed',
        f'--icon={path.join("resources", "gmt.ico")}',
        '--version-file=VERSION',
        'gui.py'
    ])
