"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
Description: 
"""

from PyQt5 import QtWidgets, uic

from gui.utils import resource_path


class ThresholdCtrl(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        uic.loadUi(resource_path('threshold_ctrl.ui'), self)
        self.show()

        self.slider.valueChanged.connect(lambda value: self.spinbox.setValue(value / 1000.))
        self.spinbox.valueChanged.connect(lambda value: self.slider.setValue(int(value * 1000)))

    @property
    def value(self) -> float:
        return self.spinbox.value()

    @value.setter
    def value(self, v: float):
        self.slider.setValue(int(v * 1000))
        self.spinbox.setValue(v)
