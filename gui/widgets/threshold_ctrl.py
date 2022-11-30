"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
"""

import sys
from os import path

from PyQt6 import QtWidgets, uic

bundle_dir = getattr(sys, '_MEIPASS', path.abspath(path.dirname(path.dirname(__file__))))
UI_PATH = path.abspath(path.join(bundle_dir, 'resources/threshold_ctrl.ui'))


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
