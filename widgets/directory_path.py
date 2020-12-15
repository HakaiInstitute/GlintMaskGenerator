"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
Description: 
"""

import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

UI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/directory_path.ui'))


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
