"""
Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-16
"""
import sys
from os import path
from typing import List, Sequence

from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal, pyqtSlot
from loguru import logger

from src.glint_mask_tools import CIRThresholdMasker, Masker, MicasenseRedEdgeThresholdMasker, P4MSThresholdMasker, RGBThresholdMasker

# String constants reduce occurrence of silent errors due to typos when doing comparisons
BLUE = "BLUE"
GREEN = "GREEN"
RED = "RED"
REDEDGE = "REDEDGE"
NIR = "NIR"

IMG_TYPE_RGB = "IMG_TYPE_RGB"
IMG_TYPE_CIR = "IMG_TYPE_CIR"
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
UI_PATH = path.abspath(path.join(bundle_dir, 'resources/gui.ui'))


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

        # Run non-ui jobs in a separate thread
        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())

        # Connect signals/slots
        self.run_btn.released.connect(self.run_btn_clicked)
        self.reset_thresholds_btn.released.connect(self.reset_thresholds)
        self.img_type_rgb_radio.clicked.connect(self.enable_available_thresholds)
        self.img_type_cir_radio.clicked.connect(self.enable_available_thresholds)
        self.img_type_p4ms_radio.clicked.connect(self.enable_available_thresholds)
        self.img_type_micasense_radio.clicked.connect(self.enable_available_thresholds)

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
        if self.img_type_cir_radio.isChecked():
            return IMG_TYPE_CIR
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
        elif self.img_type == IMG_TYPE_CIR:
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

        if self.img_type == IMG_TYPE_RGB:
            return RGBThresholdMasker(**threshold_params)
        elif self.img_type == IMG_TYPE_CIR:
            return CIRThresholdMasker(**threshold_params)
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
    def run_btn_clicked(self) -> None:
        self.run_btn.setEnabled(False)

        masker = self.create_masker()
        self.progress_val = 0
        self.progress_maximum = len(masker)

        if len(masker) < 1:
            self.err_msg.show_message("No files found in the given input directory.")
            self.run_btn.setEnabled(True)

        worker = Worker(masker, max_workers=self.max_workers)
        worker.signals.progress.connect(self._inc_progress)
        worker.signals.error.connect(self._err_callback)
        worker.signals.finished.connect(lambda: self.run_btn.setEnabled(True))

        # Execute
        self.threadpool.start(worker)


class Signals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(str, object)
    progress = pyqtSignal(int)


class Worker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Signals()

        # Add callbacks to kwargs
        self.kwargs['callback'] = self.signals.progress.emit
        self.kwargs['err_callback'] = self.signals.error.emit

    @pyqtSlot()
    def run(self):
        self.fn(*self.args, **self.kwargs)
        self.signals.finished.emit()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main_window = GlintMaskGenerator()
    app.exec_()
