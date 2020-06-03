# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Generate a executable file for the CLI and GUI interface using PyInstaller.
#   Requires pyinstaller installation.

from os import path

import PyInstaller.__main__

if __name__ == '__main__':
    PyInstaller.__main__.run([
        '--name=GlintMaskGenerator',
        '--onefile',
        '--windowed',
        f'--icon={path.join("resources", "gmt.ico")}',
        'gui.py'
    ])

    PyInstaller.__main__.run([
        '--name=glint-mask',
        '--onefile',
        'cli.py'
    ])
