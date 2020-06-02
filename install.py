# Created by: Taylor Denouden
# Organization: Hakai Institute
# Date: 2020-05-30
# Description: Generate a executable file for the CLI and GUI interface using PyInstaller.
#   Requires pyinstaller installation.

import os

import PyInstaller.__main__

if __name__ == '__main__':
    PyInstaller.__main__.run([
        '--name=glint-mask-tools',
        '--onefile',
        '--clean',
        '--console',
        # '--icon=%s' % os.path.join('resources', 'icon.ico'),
        '--version-file=VERSION',
        'gui.py',
    ])
