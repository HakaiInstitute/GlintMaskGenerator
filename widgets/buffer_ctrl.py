"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
"""

import sys
from os import path

from PyQt5 import QtWidgets, uic

bundle_dir = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))
UI_PATH = path.abspath(path.join(bundle_dir, 'buffer_ctrl.ui'))


class BufferCtrl(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        uic.loadUi(UI_PATH, self)
        self.show()

        self.slider.valueChanged.connect(lambda value: self.spinbox.setValue(int(value)))
        self.spinbox.valueChanged.connect(lambda value: self.slider.setValue(int(value)))

    @property
    def value(self) -> int:
        return self.spinbox.value()

    @value.setter
    def value(self, v: int):
        self.slider.setValue(v)
        self.spinbox.setValue(v)
