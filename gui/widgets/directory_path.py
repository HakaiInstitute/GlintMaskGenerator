"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
"""
import sys
from os import path

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

bundle_dir = getattr(sys, '_MEIPASS', path.abspath(path.dirname(path.dirname(__file__))))
UI_PATH = path.abspath(path.join(bundle_dir, 'resources/directory_path.ui'))


class DirectoryPath(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        uic.loadUi(UI_PATH, self)
        self.show()

    @property
    def value(self) -> str:
        return self.textedit.text()

    @value.setter
    def value(self, path: str):
        self.textedit.setText(path)

    def dir_btn_clicked(self):
        self.value = QFileDialog.getExistingDirectory(
            self, 'Select directory', self.value, QFileDialog.ShowDirsOnly)
