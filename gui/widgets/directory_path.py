"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17.
"""

from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog

from gui.utils import resource_path


class DirectoryPath(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)

        uic.loadUi(resource_path("resources/directory_path.ui"), self)
        self.show()

    @property
    def value(self) -> str:
        return self.textedit.text()

    @value.setter
    def value(self, path: str) -> None:
        self.textedit.setText(path)

    def dir_btn_clicked(self) -> None:
        self.value = QFileDialog.getExistingDirectory(
            self,
            "Select directory",
            self.value,
            QFileDialog.Option.ShowDirsOnly,
        )
