"""Sensor configuration module for dynamic generation of CLI commands and GUI interfaces.

This module defines sensor configurations that specify the bands and image loaders
for different camera sensor_configs. The configurations are used by both the CLI and GUI
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
class Band:
    """A sensor band with a name and default threshold value."""

    name: str
    default_threshold: float


B = Band("Blue", 0.875)
G = Band("Green", 1.000)
R = Band("Red", 1.000)
RE = Band("Red Edge", 1.000)
NIR = Band("Near-IR", 1.000)


@dataclass
class Sensor:
    """Sensor configuration class that specifies the name and band order, as well as loader class to handle imagery."""

    name: str
    cli_command: str
    bands: list[Band]
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


sensor_configs = (
    Sensor(
        name="RGB",
        cli_command="rgb",
        bands=[R, G, B],
        loader_class=RGBLoader,
    ),
    Sensor(
        name="PhaseOne 4-band CIR",
        cli_command="cir",
        bands=[R, G, B, NIR],
        loader_class=CIRLoader,
    ),
    Sensor(
        name="DJI P4MS",
        cli_command="p4ms",
        bands=[B, G, R, RE, NIR],
        loader_class=P4MSLoader,
    ),
    Sensor(
        name="DJI M3M",
        cli_command="m3m",
        bands=[Band("Green", 0.875), R, RE, NIR],
        loader_class=DJIM3MLoader,
    ),
    Sensor(
        name="MicaSense RedEdge",
        cli_command="msre",
        bands=[B, G, R, RE, NIR],
        loader_class=MicasenseRedEdgeLoader,
    ),
)
