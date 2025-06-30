"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17.
"""

from PyQt6 import QtWidgets, uic

from gui.utils import resource_path


class ThresholdCtrl(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent)

        uic.loadUi(resource_path("resources/threshold_ctrl.ui"), self)

        self.slider.valueChanged.connect(
            lambda value: self.spinbox.setValue(value / 1000.0),
        )
        self.spinbox.valueChanged.connect(
            lambda value: self.slider.setValue(int(value * 1000)),
        )

        self.show()

    @property
    def value(self) -> float:
        return self.spinbox.value()

    @value.setter
    def value(self, v: float) -> None:
        self.slider.setValue(int(v * 1000))
        self.spinbox.setValue(v)
