"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
Description: 
"""

import os

from PyQt5 import QtWidgets, uic

UI_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../resources/threshold_ctrl.ui'))


class ThresholdCtrl(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        uic.loadUi(UI_PATH, self)
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
