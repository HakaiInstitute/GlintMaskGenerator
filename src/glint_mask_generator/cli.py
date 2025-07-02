"""CLI functions for the Glint Masker Generator.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Command line interface to the glint-mask-tools.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Annotated, Callable

import typer
from tqdm import tqdm

from .sensors import _SensorConfig, sensors

if TYPE_CHECKING:
    from .maskers import Masker


app = typer.Typer()


def _err_callback(paths: list[str], exception: Exception) -> None:
    tqdm.write(f"{paths} failed with err:\n{exception}", file=sys.stderr)


def _process(masker: Masker, max_workers: int) -> None:
    with tqdm(total=len(masker)) as progress:
        masker(
            max_workers=max_workers,
            callback=lambda _: progress.update(1),
            err_callback=_err_callback,
        )


def _create_sensor_command(sensor_cfg: _SensorConfig) -> Callable[..., None]:
    """Create a CLI command function for a sensor configuration."""

    def sensor_command(
        img_dir: Annotated[
            Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=True,
                help="The path to a named input image or directory containing images. "
                "If img_dir is a directory, all tif, jpg, jpeg, and png images in that directory will be processed.",
            ),
        ],
        out_dir: Annotated[
            Path,
            typer.Argument(
                exists=True,
                file_okay=True,
                dir_okay=True,
                help='The path to send your out image including the file name and type. e.g. "/path/to/mask.png". '
                "The out_dir must be a directory if img_dir is specified as a directory.",
            ),
        ],
        thresholds: Annotated[
            list[float] | None,
            typer.Option(
                help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
            ),
        ] = None,
        pixel_buffer: Annotated[
            int,
            typer.Option(help="The pixel distance to buffer out the mask."),
        ] = 0,
        max_workers: Annotated[
            int,
            typer.Option(help="The maximum number of threads to use for processing."),
        ] = min(4, os.cpu_count()),
    ) -> None:
        if thresholds is None:
            thresholds = sensor_cfg.get_default_thresholds()

        masker = sensor_cfg.create_masker(str(img_dir), str(out_dir), thresholds, pixel_buffer)
        _process(masker, max_workers)

    sensor_command.__doc__ = f"Generate glint masks for {sensor_cfg.name} sensors using threshold algorithm."
    return sensor_command


# Dynamically register sensor commands
for sensor_config in sensors:
    command_func = _create_sensor_command(sensor_config)
    app.command(name=sensor_config.cli_command)(command_func)


if __name__ == "__main__":
    app()
