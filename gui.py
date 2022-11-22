"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-16
Description: 
"""
import sys
from os import path
from typing import List, Sequence

from PyQt5 import QtWidgets, uic
from loguru import logger

from core.maskers import Masker, MicasenseRedEdgeThresholdMasker, P4MSThresholdMasker, RGBThresholdMasker

# String constants reduce occurrence of silent errors due to typos when doing comparisons
BLUE = "BLUE"
GREEN = "GREEN"
RED = "RED"
REDEDGE = "REDEDGE"
NIR = "NIR"

METHOD_THRESHOLD = "METHOD_THRESHOLD"
METHOD_RATIO = "METHOD_RATIO"

IMG_TYPE_RGB = "IMG_TYPE_RGB"
IMG_TYPE_ACO = "IMG_TYPE_ACO"
IMG_TYPE_P4MS = "IMG_TYPE_P4MS"
IMG_TYPE_MICASENSE_REDEDGE = "IMG_TYPE_MICASENSE_REDEDGE"

# Default slider values in GUI
DEFAULT_BLUE_THRESH = 0.875
DEFAULT_GREEN_THRESH = 1.000
DEFAULT_RED_THRESH = 1.000
DEFAULT_REDEDGE_THRESH = 1.000
DEFAULT_NIR_THRESH = 1.000
DEFAULT_PIXEL_BUFFER = 0
DEFAULT_MAX_WORKERS = 0

bundle_dir = getattr(sys, '_MEIPASS', path.abspath(path.dirname(__file__)))
UI_PATH = path.abspath(path.join(bundle_dir, 'gui.ui'))


class MessageBox(QtWidgets.QMessageBox):
    def __init__(self, parent, title, icon):
        super().__init__(parent)
        self.setIcon(icon)
        self.setWindowTitle(title)

    def show_message(self, message):
        self.setText(message)
        self.exec_()


class InfoMessageBox(MessageBox):
    def __init__(self, parent):
        super().__init__(parent, "Info", QtWidgets.QMessageBox.Information)


class ErrorMessageBox(MessageBox):
    def __init__(self, parent):
        super().__init__(parent, "Error", QtWidgets.QMessageBox.Critical)


class GlintMaskGenerator(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_PATH, self)
        self.show()

        # Set default values
        self.reset_thresholds()
        self.pixel_buffer_w.value = DEFAULT_PIXEL_BUFFER
        self.max_workers_spinbox.setValue(DEFAULT_MAX_WORKERS)

        # Enable/disable threshold controls based on imagery type
        self.enable_available_thresholds()

        self.progress_val = 0

        # Message popup boxes
        self.err_msg = ErrorMessageBox(self)
        self.info_msg = InfoMessageBox(self)

    def enable_available_thresholds(self) -> None:
        self.blue_thresh_w.setEnabled(BLUE in self.band_order)
        self.green_thresh_w.setEnabled(GREEN in self.band_order)
        self.red_thresh_w.setEnabled(RED in self.band_order)
        self.rededge_thresh_w.setEnabled(REDEDGE in self.band_order)
        self.nir_thresh_w.setEnabled(NIR in self.band_order)

    def reset_thresholds(self) -> None:
        self.blue_thresh_w.value = DEFAULT_BLUE_THRESH
        self.green_thresh_w.value = DEFAULT_GREEN_THRESH
        self.red_thresh_w.value = DEFAULT_RED_THRESH
        self.rededge_thresh_w.value = DEFAULT_REDEDGE_THRESH
        self.nir_thresh_w.value = DEFAULT_NIR_THRESH

    @property
    def img_type(self) -> str:
        if self.img_type_aco_radio.isChecked():
            return IMG_TYPE_ACO
        elif self.img_type_p4ms_radio.isChecked():
            return IMG_TYPE_P4MS
        elif self.img_type_micasense_radio.isChecked():
            return IMG_TYPE_MICASENSE_REDEDGE
        else:  # self.img_type_rgb_radio.isChecked()
            return IMG_TYPE_RGB

    @property
    def max_workers(self) -> int:
        return max(self.max_workers_spinbox.value(), 0)

    @property
    def band_order_ints(self) -> Sequence[int]:
        return [{BLUE: 0, GREEN: 1, RED: 2, REDEDGE: 3, NIR: 4}[k] for k in self.band_order]

    @property
    def band_order(self) -> Sequence[str]:
        if self.img_type == IMG_TYPE_RGB:
            return RED, GREEN, BLUE
        elif self.img_type == IMG_TYPE_ACO:
            return RED, GREEN, BLUE, NIR
        elif self.img_type == IMG_TYPE_P4MS:
            return BLUE, GREEN, RED, REDEDGE, NIR
        if self.img_type == IMG_TYPE_MICASENSE_REDEDGE:
            return BLUE, GREEN, RED, NIR, REDEDGE

    @property
    def threshold_values(self) -> Sequence[float]:
        """Returns the thresholds in the order corresponding to the imagery type band order."""
        thresholds: List[float] = [
            self.blue_thresh_w.value,
            self.green_thresh_w.value,
            self.red_thresh_w.value,
            self.rededge_thresh_w.value,
            self.nir_thresh_w.value,
        ]
        return [thresholds[i] for i in self.band_order_ints]

    def create_masker(self) -> Masker:
        """Returns an instance of the appropriate glint mask generator given selected options."""
        threshold_params = dict(
            img_dir=self.img_dir_w.value,
            mask_dir=self.mask_dir_w.value,
            thresholds=self.threshold_values,
            pixel_buffer=self.pixel_buffer_w.value
        )

        if self.img_type == IMG_TYPE_RGB or self.img_type == IMG_TYPE_ACO:
            return RGBThresholdMasker(**threshold_params)
        elif self.img_type == IMG_TYPE_P4MS:
            return P4MSThresholdMasker(**threshold_params)
        elif self.img_type == IMG_TYPE_MICASENSE_REDEDGE:
            return MicasenseRedEdgeThresholdMasker(**threshold_params)
        else:
            raise ValueError(f"No masker available for img type {self.img_type}")

    @property
    def progress_val(self):
        return self.progress_bar.value()

    @progress_val.setter
    def progress_val(self, v):
        self.progress_bar.setValue(v)

    @property
    def progress_maximum(self):
        return self.progress_bar.maximum()

    @progress_maximum.setter
    def progress_maximum(self, v):
        self.progress_bar.setMaximum(v)

    def _inc_progress(self, _):
        self.progress_val += 1

        if self.progress_val == self.progress_maximum:
            self.info_msg.show_message('Processing is complete')

    def _err_callback(self, img_path, err):
        msg = '%r generated an exception: %s' % (img_path, err)
        self.err_msg.show_message(msg)

    @logger.catch
    def preview_btn_clicked(self) -> None:
        raise NotImplemented

    @logger.catch
    def run_btn_clicked(self) -> None:
        masker = self.create_masker()
        self.progress_val = 0
        self.progress_maximum = len(masker)

        if len(masker) < 1:
            self.err_msg.show_message("No files found in the given input directory.")

        masker(max_workers=self.max_workers, callback=self._inc_progress, err_callback=self._err_callback)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = GlintMaskGenerator()
    app.exec_()
