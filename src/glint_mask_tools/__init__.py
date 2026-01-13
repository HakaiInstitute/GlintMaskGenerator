"""GlintMaskGenerator: Tools for detecting and masking specular reflections in imagery.

This package identifies areas of high specular reflectance (glint) in images and generates
binary masks to mark these regions. The output masks can be imported into structure-from-motion
software to automatically replace glinted areas with clean imagery from overlapping photos,
significantly reducing specular artifacts in final mosaiced outputs.

Key workflow:
1. Detect glint areas in input images
2. Generate binary masks marking glinted regions
3. Use masks in photogrammetry software to substitute glinted areas with clean overlapping imagery
4. Produce higher-quality mosaics with reduced specular reflections

Created by: Taylor Denouden
Organization: Hakai Institute
Date: 2020-09-18
"""

import importlib.metadata

__version__ = importlib.metadata.version("glint-mask-tools")
