"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17.
"""

from PyQt6 import QtWidgets, uic

from gui.utils import resource_path


class BufferCtrl(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)

        uic.loadUi(resource_path("resources/buffer_ctrl.ui"), self)

        self.slider.valueChanged.connect(
            lambda value: self.spinbox.setValue(int(value)),
        )
        self.spinbox.valueChanged.connect(
            lambda value: self.slider.setValue(int(value)),
        )

        self.show()

    @property
    def value(self) -> int:
        return self.spinbox.value()

    @value.setter
    def value(self, v: int) -> None:
        self.slider.setValue(v)
        self.spinbox.setValue(v)
