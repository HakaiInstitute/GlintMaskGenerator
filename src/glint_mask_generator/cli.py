"""CLI functions for the Glint Masker Generator.

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-05-30
Description: Command line interface to the glint-mask-tools.
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Annotated

import typer
from tqdm import tqdm

from .maskers import (
    CIRThresholdMasker,
    Masker,
    MicasenseRedEdgeThresholdMasker,
    P4MSThresholdMasker,
    RGBThresholdMasker,
)

if TYPE_CHECKING:
    from pathlib import Path

app = typer.Typer()


def _err_callback(path: str, exception: Exception) -> None:
    tqdm.write(f"{path} failed with err:\n{exception}", file=sys.stderr)


def _process(masker: Masker, max_workers: int) -> None:
    with tqdm(total=len(masker)) as progress:
        masker(
            max_workers=max_workers,
            callback=lambda _: progress.update(1),
            err_callback=_err_callback,
        )


@app.command()
def rgb_threshold(
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
        list[float],
        typer.Option(
            help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
        ),
    ] = (0.875, 1, 1, 1, 1),
    pixel_buffer: Annotated[
        int,
        typer.Option(help="The pixel distance to buffer out the mask."),
    ] = 0,
    max_workers: Annotated[
        int,
        typer.Option(help="The maximum number of threads to use for processing."),
    ] = min(4, os.cpu_count()),
) -> None:
    """Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm."""
    _process(
        RGBThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer),
        max_workers,
    )


@app.command()
def cir_threshold(
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
        list[float],
        typer.Option(
            help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
        ),
    ] = (0.875, 1, 1, 1, 1),
    pixel_buffer: Annotated[
        int,
        typer.Option(help="The pixel distance to buffer out the mask."),
    ] = 0,
    max_workers: Annotated[
        int,
        typer.Option(help="The maximum number of threads to use for processing."),
    ] = min(4, os.cpu_count()),
) -> None:
    """Generate glint masks for 4-band CIR sensors using disjunctive pixel intensity thresholding.

    CIR refers to RGB+NIR imagery from PhaseOne cameras that have been pre-processed by IX Capture.
    """
    _process(
        CIRThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer),
        max_workers,
    )


@app.command()
def p4ms_threshold(
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
        list[float],
        typer.Option(
            help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
        ),
    ] = (0.875, 1, 1, 1, 1),
    pixel_buffer: Annotated[
        int,
        typer.Option(help="The pixel distance to buffer out the mask."),
    ] = 0,
    max_workers: Annotated[
        int,
        typer.Option(help="The maximum number of threads to use for processing."),
    ] = min(4, os.cpu_count()),
) -> None:
    """Generate glint masks for DJI P4MS sensors using disjunctive pixel intensity thresholding."""
    _process(
        P4MSThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer),
        max_workers,
    )


@app.command()
def micasense_threshold(
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
        list[float],
        typer.Option(
            help="The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0).",
        ),
    ] = (0.875, 1, 1, 1, 1),
    pixel_buffer: Annotated[
        int,
        typer.Option(help="The pixel distance to buffer out the mask."),
    ] = 0,
    max_workers: Annotated[
        int,
        typer.Option(help="The maximum number of threads to use for processing."),
    ] = min(4, os.cpu_count()),
) -> None:
    """Generate glint masks for Micasense RedEdge sensors using disjunctive pixel intensity thresholding."""
    _process(
        MicasenseRedEdgeThresholdMasker(img_dir, out_dir, thresholds, pixel_buffer),
        max_workers,
    )


if __name__ == "__main__":
    app()
