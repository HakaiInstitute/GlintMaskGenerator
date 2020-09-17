"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-17
Description: 
"""

from PyQt5 import QtWidgets, uic

from utils import resource_path


class BufferCtrl(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        uic.loadUi(resource_path('buffer_ctrl.ui'), self)
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
