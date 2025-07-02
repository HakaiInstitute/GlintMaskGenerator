"""Sensor configuration module for dynamic generation of CLI commands and GUI interfaces.

This module defines sensor configurations that specify the bands and image loaders
for different camera sensors. The configurations are used by both the CLI and GUI
to dynamically generate appropriate interfaces for each sensor type.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

from __future__ import annotations

from dataclasses import dataclass

from glint_mask_generator.glint_algorithms import ThresholdAlgorithm
from glint_mask_generator.image_loaders import (
    CIRLoader,
    DJIM3MLoader,
    ImageLoader,
    MicasenseRedEdgeLoader,
    P4MSLoader,
    RGBLoader,
)
from glint_mask_generator.maskers import Masker


@dataclass(frozen=True)
class _BandConfig:
    name: str
    default_threshold: float


@dataclass
class _SensorConfig:
    name: str
    cli_command: str
    bands: list[_BandConfig]
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
    "B": _BandConfig("Blue", 0.875),
    "G": _BandConfig("Green", 1.000),
    "R": _BandConfig("Red", 1.000),
    "RE": _BandConfig("Red Edge", 1.000),
    "NIR": _BandConfig("Near-IR", 1.000),
}
sensors = (
    _SensorConfig(
        name="RGB",
        cli_command="rgb",
        bands=[_bands.get(b) for b in ["R", "G", "B"]],
        loader_class=RGBLoader,
    ),
    _SensorConfig(
        name="PhaseOne 4-band CIR",
        cli_command="cir",
        bands=[_bands.get(b) for b in ["R", "G", "B", "NIR"]],
        loader_class=CIRLoader,
    ),
    _SensorConfig(
        name="DJI P4MS",
        cli_command="p4ms",
        bands=[_bands.get(b) for b in ["B", "G", "R", "RE", "NIR"]],
        loader_class=P4MSLoader,
    ),
    _SensorConfig(
        name="DJI M3M",
        cli_command="m3m",
        bands=[_BandConfig("Green", 0.875)] + [_bands.get(b) for b in ["R", "RE", "NIR"]],
        loader_class=DJIM3MLoader,
    ),
    _SensorConfig(
        name="MicaSense RedEdge",
        cli_command="msre",
        bands=[_bands.get(b) for b in ["B", "G", "R", "RE", "NIR"]],
        loader_class=MicasenseRedEdgeLoader,
    ),
)
