"""Sensor configuration module for dynamic generation of CLI commands and GUI interfaces.

This module defines sensor configurations that specify the bands and image loaders
for different camera _known_sensors. The configurations are used by both the CLI and GUI
to dynamically generate appropriate interfaces for each sensor type.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .glint_algorithms import ThresholdAlgorithm
from .image_loaders import (
    BigTiffLoader,
    DJIM3MLoader,
    ImageLoader,
    MicasenseRedEdgeDualLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    SingleFileImageLoader,
)
from .maskers import Masker
from .utils import normalize_img

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Band:
    """A sensor band with a name and default threshold value."""

    name: str
    default_threshold: float = 1.000


B = Band("Blue", 0.875)
G = Band("Green")
R = Band("Red")
RE = Band("Red Edge")
NIR = Band("Near-IR")


@dataclass
class Sensor:
    """Sensor configuration class that specifies the name and band order, as well as loader class to handle imagery."""

    name: str
    bands: list[Band]
    bit_depth: int
    loader_class: type[ImageLoader]
    supports_alignment: bool = False

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Scale the values in the imagery or do other preprocessing logic when overridden."""
        return normalize_img(img, bit_depth=self.bit_depth)

    def create_masker(  # noqa: PLR0913
        self,
        img_dir: str,
        mask_dir: str,
        thresholds: list[float],
        pixel_buffer: int,
        *,
        per_band: bool = False,
        align_bands: bool = True,
    ) -> Masker:
        """Create a masker instance for this sensor configuration."""
        aligner = None
        if self.supports_alignment and align_bands:
            from .band_alignment import BandAligner  # noqa: PLC0415

            aligner = BandAligner(enabled=True)

        return Masker(
            algorithm=ThresholdAlgorithm(thresholds, per_band=per_band),
            image_loader=self.loader_class(img_dir, mask_dir),
            image_preprocessor=self.preprocess_image,
            pixel_buffer=pixel_buffer,
            per_band=per_band,
            band_aligner=aligner,
        )

    def get_default_thresholds(self) -> list[float]:
        """Get the default threshold values for all bands."""
        return [band.default_threshold for band in self.bands]


rgb_sensor = Sensor(
    name="RGB",
    bands=[R, G, B],
    bit_depth=8,
    loader_class=SingleFileImageLoader,
)
cir_sensor = Sensor(
    name="PhaseOne 4-band CIR",
    bands=[R, G, B, NIR],
    bit_depth=8,
    loader_class=BigTiffLoader,
)
p4ms_sensor = Sensor(
    name="DJI P4MS",
    bands=[B, G, R, RE, NIR],
    bit_depth=16,
    loader_class=P4MSLoader,
    supports_alignment=True,
)
m3m_sensor = Sensor(
    name="DJI M3M",
    bands=[Band("Green", 0.875), R, RE, NIR],
    bit_depth=16,
    loader_class=DJIM3MLoader,
    supports_alignment=True,
)
msre_sensor = Sensor(
    name="MicaSense RedEdge",
    bands=[B, G, R, RE, NIR],
    bit_depth=16,
    loader_class=MicasenseRedEdgeLoader,
    supports_alignment=True,
)

msre_dual_sensor = Sensor(
    name="MicaSense RedEdge-MX Dual",
    bands=[
        Band("Coastal Blue 444(28)", 1.000),
        Band("Blue 475(32)", 0.875),
        Band("Green 531(14)", 1.000),
        Band("Green 560(27)", 1.000),
        Band("Red 650(16)", 1.000),
        Band("Red 668(14)", 1.000),
        Band("Red Edge 705(10)", 1.000),
        Band("Red Edge 717(12)", 1.000),
        Band("Red Edge 740(18)", 1.000),
        Band("Near-IR 842(57)", 1.000),
    ],
    bit_depth=16,
    loader_class=MicasenseRedEdgeDualLoader,
    supports_alignment=True,
)


# Auto populate GUI and CLI options
@dataclass(frozen=True)
class _KnownSensor:
    sensor: Sensor
    cli_name: str


_known_sensors = (
    _KnownSensor(rgb_sensor, cli_name="rgb"),
    _KnownSensor(cir_sensor, cli_name="cir"),
    _KnownSensor(p4ms_sensor, cli_name="p4ms"),
    _KnownSensor(m3m_sensor, cli_name="m3m"),
    _KnownSensor(msre_sensor, cli_name="msre"),
    _KnownSensor(msre_dual_sensor, cli_name="msre-dual"),
)
