"""Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-16.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QIcon

from glint_mask_generator import Masker, __version__
from glint_mask_generator.glint_algorithms import ThresholdAlgorithm
from glint_mask_generator.image_loaders import (
    CIRLoader,
    ImageLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    RGBLoader,
)
from gui.utils import resource_path
from gui.widgets.threshold_ctrl import ThresholdCtrl


@dataclass(frozen=True)
class BandConfig:
    name: str
    default_threshold: float


@dataclass
class SensorConfig:
    name: str
    bands: list[BandConfig]
    loader_class: type[ImageLoader]

    def create_masker(self, img_dir: str, mask_dir: str, thresholds: list[float], pixel_buffer: int) -> Masker:
        """Create a masker instance for this sensor configuration."""
        return Masker(
            algorithm=ThresholdAlgorithm(thresholds),
            image_loader=self.loader_class(img_dir, mask_dir),
            pixel_buffer=pixel_buffer,
        )

    def get_default_thresholds(self) -> list[float]:
        """Get the default threshold values for all bands."""
        return [band.default_threshold for band in self.bands]


_bands = {
    "CB": BandConfig("Coastal Blue", 1.000),
    "B": BandConfig("Blue", 0.875),
    "G": BandConfig("Green", 1.000),
    "G2": BandConfig("Green 2", 1.000),
    "R": BandConfig("Red", 1.000),
    "R2": BandConfig("Red 2", 1.000),
    "RE": BandConfig("Red Edge", 1.000),
    "RE2": BandConfig("Red Edge 2", 1.000),
    "NIR": BandConfig("NIR", 1.000),
    "NIR2": BandConfig("NIR 2", 1.000),
}

sensors = (
    SensorConfig(name="RGB", bands=[_bands.get(b) for b in ["R", "G", "B"]], loader_class=RGBLoader),
    SensorConfig(name="CIR", bands=[_bands.get(b) for b in ["R", "G", "B", "NIR"]], loader_class=CIRLoader),
    SensorConfig(
        name="P4MS",
        bands=[_bands.get(b) for b in ["B", "G", "R", "RE", "NIR"]],
        loader_class=P4MSLoader,
    ),
    SensorConfig(
        name="MicaSense RE",
        bands=[_bands.get(b) for b in ["B", "G", "R", "RE", "NIR"]],
        loader_class=MicasenseRedEdgeLoader,
    ),
)

DEFAULT_PIXEL_BUFFER = 0
DEFAULT_MAX_WORKERS = 0


class MessageBox(QtWidgets.QMessageBox):
    def __init__(self, parent: QtWidgets.QWidget, title: str, icon: str) -> None:
        super().__init__(parent)
        self.setIcon(icon)
        self.setWindowTitle(title)

    def show_message(self, message: str) -> None:
        self.setText(message)
        self.exec()


class InfoMessageBox(MessageBox):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent, "Info", QtWidgets.QMessageBox.Icon.Information)


class ErrorMessageBox(MessageBox):
    def __init__(self, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent, "Error", QtWidgets.QMessageBox.Icon.Critical)


class GlintMaskGenerator(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        uic.loadUi(resource_path("resources/gui.ui"), self)

        # Setup window
        self.setWindowTitle(f"Glint Mask Generator v{__version__}")
        self.setWindowIcon(QIcon(resource_path("resources/gmt.ico")))

        # Initialize sensor management
        self.selected_sensor: SensorConfig | None = None
        self.sensor_radios: list[QtWidgets.QRadioButton] = []
        self.threshold_widgets: list[ThresholdCtrl] = []
        self.threshold_labels: list[QtWidgets.QLabel] = []

        # Create dynamic sensor radio buttons
        self.create_sensor_radios()

        # Set default sensor and thresholds
        if self.sensor_radios:
            self.sensor_radios[0].setChecked(True)
            self.on_sensor_changed()

        # Set default values
        self.pixel_buffer_w.value = DEFAULT_PIXEL_BUFFER
        self.max_workers_spinbox.setValue(DEFAULT_MAX_WORKERS)

        self.progress_val = 0

        # Message popup boxes
        self.err_msg = ErrorMessageBox(self)
        self.info_msg = InfoMessageBox(self)

        # Run non-ui jobs in a separate thread
        self.threadpool = QThreadPool()

        # Set max workers to good default
        self.max_workers = min(4, os.cpu_count())

        # Connect signals/slots
        self.run_btn.released.connect(self.run_btn_clicked)
        self.reset_thresholds_btn.released.connect(self.reset_thresholds)

        self.show()

    def create_sensor_radios(self) -> None:
        """Dynamically create radio buttons for each sensor configuration."""
        # Clear existing radio buttons from layout
        for radio in self.sensor_radios:
            radio.deleteLater()
        self.sensor_radios.clear()

        # Create new radio buttons
        for i, sensor in enumerate(sensors):
            radio = QtWidgets.QRadioButton(sensor.name)
            if i == 0:  # First sensor is default
                radio.setChecked(True)
            radio.clicked.connect(self.on_sensor_changed)
            self.sensor_radios.append(radio)
            self.box_img_types.addWidget(radio)

    def on_sensor_changed(self) -> None:
        """Handle sensor selection change."""
        # Find which radio button is checked
        for i, radio in enumerate(self.sensor_radios):
            if radio.isChecked():
                self.selected_sensor = sensors[i]
                break

        # Update threshold sliders for the selected sensor
        self.create_threshold_sliders()

    def create_threshold_sliders(self) -> None:
        """Dynamically create threshold sliders for the selected sensor's bands."""
        if not self.selected_sensor:
            return

        # Clear existing threshold widgets
        for widget in self.threshold_widgets:
            widget.deleteLater()
        for label in self.threshold_labels:
            label.deleteLater()
        self.threshold_widgets.clear()
        self.threshold_labels.clear()

        # Get the grid layout from the threshold group box
        grid_layout = self.box_band_threshes.layout()

        # Create threshold sliders for each band
        for i, band in enumerate(self.selected_sensor.bands):
            # Create label
            label = QtWidgets.QLabel(band.name)
            label.setAlignment(Qt.AlignmentFlag.AlignLeft)

            # Create threshold control
            threshold_ctrl = ThresholdCtrl(self)
            threshold_ctrl.value = band.default_threshold

            # Add to layout (row i, columns 0 and 2)
            grid_layout.addWidget(label, i, 0)
            grid_layout.addWidget(threshold_ctrl, i, 2)

            # Store references
            self.threshold_labels.append(label)
            self.threshold_widgets.append(threshold_ctrl)

        # Keep the reset button at the bottom
        reset_row = len(self.selected_sensor.bands)
        if hasattr(self, "reset_thresholds_btn"):
            grid_layout.addWidget(self.reset_thresholds_btn.parent(), reset_row, 2)

    def reset_thresholds(self) -> None:
        """Reset all threshold sliders to their default values."""
        if not self.selected_sensor:
            return

        for i, band in enumerate(self.selected_sensor.bands):
            if i < len(self.threshold_widgets):
                self.threshold_widgets[i].value = band.default_threshold

    @property
    def selected_sensor_config(self) -> SensorConfig:
        """Get the currently selected sensor configuration."""
        if self.selected_sensor is None:
            msg = "No sensor selected"
            raise ValueError(msg)
        return self.selected_sensor

    @property
    def max_workers(self) -> int:
        return max(self.max_workers_spinbox.value(), 0)

    @max_workers.setter
    def max_workers(self, v: int) -> None:
        self.max_workers_spinbox.setValue(v)

    @property
    def threshold_values(self) -> list[float]:
        """Returns the current threshold values for all bands."""
        return [widget.value for widget in self.threshold_widgets]

    def create_masker(self) -> Masker:
        """Returns an instance of the appropriate glint mask generator given selected options."""
        return self.selected_sensor_config.create_masker(
            img_dir=self.img_dir_w.value,
            mask_dir=self.mask_dir_w.value,
            thresholds=self.threshold_values,
            pixel_buffer=self.pixel_buffer_w.value,
        )

    @property
    def progress_val(self) -> int:
        return self.progress_bar.value()

    @progress_val.setter
    def progress_val(self, v: int) -> None:
        self.progress_bar.setValue(v)

    @property
    def progress_maximum(self) -> int:
        return self.progress_bar.maximum()

    @progress_maximum.setter
    def progress_maximum(self, v: int) -> None:
        self.progress_bar.setMaximum(v)

    def _inc_progress(self, _: int) -> None:
        self.progress_val += 1

        if self.progress_val == self.progress_maximum:
            self.info_msg.show_message("Processing is complete")

    def _err_callback(self, img_path: str, err: Exception) -> None:
        msg = f"{img_path!r} generated an exception: {err}"
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
    def __init__(self, fn: Callable[[Any], Any], *args: list[Any], **kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Signals()

        # Add callbacks to kwargs
        self.kwargs["callback"] = self.signals.progress.emit
        self.kwargs["err_callback"] = self.signals.error.emit

    @pyqtSlot()
    def run(self) -> None:
        self.fn(*self.args, **self.kwargs)
        self.signals.finished.emit()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = GlintMaskGenerator()
    app.exec()
