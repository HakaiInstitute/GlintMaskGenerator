# Python API

The Python library provides flexible integration into existing workflows and custom processing pipelines.

## Quick Start

Install the package:

```bash
pip install glint-mask-tools
```

Basic usage with RGB imagery:

```python
from glint_mask_tools.sensors import rgb_sensor

# Create masker using sensor configuration
masker = rgb_sensor.create_masker(
    img_dir="/path/to/images",
    mask_dir="/path/to/masks",
    thresholds=[0.85, 0.85, 0.75],  # R, G, B thresholds
    pixel_buffer=2
)

# Process all images
masker(max_workers=4)
```

## Supported Sensor Types

The library includes built-in support for multiple sensor types:

| Sensor | CLI Command | Bands | Default Thresholds | File Pattern |
|--------|-------------|-------|-------------------|--------------|
| RGB | `rgb` | Red, Green, Blue | [1.0, 1.0, 0.875] | Single file |
| CIR | `cir` | Red, Green, Blue, Near-IR | [1.0, 1.0, 0.875, 1.0] | Single file |
| DJI P4MS | `p4ms` | Blue, Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0, 1.0] | `DJI_###[1-5].TIF` |
| DJI M3M | `m3m` | Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0] | `DJI_*_MS_[G\|R\|RE\|NIR].TIF` |
| MicaSense RedEdge | `msre` | Blue, Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0, 1.0] | `IMG_####_[1-5].tif` |
| MicaSense RedEdge-MX Dual | `msre-dual` | 10 bands (various wavelengths) | [1.0, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] | `IMG_####_[1-10].tif` |

## Using Sensor Configurations

Each sensor has a pre-configured setup accessible via the sensors module:

```python
from glint_mask_tools.sensors import p4ms_sensor, msre_sensor, cir_sensor

# DJI Phantom 4 Multispectral
masker = p4ms_sensor.create_masker(
    img_dir="/path/to/p4ms_images",
    mask_dir="/path/to/masks",
    thresholds=p4ms_sensor.get_default_thresholds(),
    pixel_buffer=3
)

# MicaSense RedEdge
masker = msre_sensor.create_masker(
    img_dir="/path/to/micasense_images",
    mask_dir="/path/to/masks",
    pixel_buffer=2
)

# MicaSense RedEdge-MX Dual (10-band)
from glint_mask_tools.sensors import msre_dual_sensor

masker = msre_dual_sensor.create_masker(
    img_dir="/path/to/micasense_dual_images",
    mask_dir="/path/to/masks",
    thresholds=msre_dual_sensor.get_default_thresholds(),  # 10 thresholds
    pixel_buffer=2
)

# Color Infrared (large images with tiling)
masker = cir_sensor.create_masker(
    img_dir="/path/to/cir_images",
    mask_dir="/path/to/masks"
)
```

## Advanced Usage with Components

For more control, you can directly use the individual components:

```python
from glint_mask_tools.image_loaders import SingleFileImageLoader
from glint_mask_tools.glint_algorithms import ThresholdAlgorithm
from glint_mask_tools.maskers import Masker
from glint_mask_tools.utils import normalize_8bit_img

# Custom RGB setup with advanced algorithm
loader = SingleFileImageLoader("/path/to/images", "/path/to/masks")
algorithm = ThresholdAlgorithm(thresholds=[0.8, 0.8, 0.8])
masker = Masker(algorithm, loader, normalize_8bit_img, pixel_buffer=1)

# Process with callbacks
def progress_callback(completed_paths):
    print(f"Completed: {len(completed_paths)} images")

def error_callback(failed_paths, exception):
    print(f"Error processing {failed_paths}: {exception}")

masker(max_workers=2, callback=progress_callback, err_callback=error_callback)
```

## Image Loader Classes

The library uses a hierarchical image loader system:

### Single-File Loaders

**`SingleFileImageLoader`**: Standard 8-bit RGB imagery in one file

- Supports: JPG, PNG, TIFF formats
- Use case: Standard photography, drone RGB cameras

**`BigTiffLoader`**: 8-bit Color Infrared imagery in BigTIFF format

- Features: Memory-efficient tiled processing (256x256 tiles)
- Use case: Large 4-band CIR images that don't fit in memory

### Multi-File Loaders

**`MicasenseRedEdgeLoader`**: 16-bit multi-file MicaSense format

- Pattern: `IMG_####_[1-5].tif` (Blue, Green, Red, RedEdge, NIR)
- Use case: MicaSense RedEdge cameras

**`P4MSLoader`**: 16-bit multi-file DJI P4 Multispectral format

- Pattern: `DJI_###[1-5].TIF` (Blue, Green, Red, RedEdge, NIR)
- Use case: DJI Phantom 4 Multispectral camera

**`DJIM3MLoader`**: 16-bit multi-file DJI Mavic 3M format

- Pattern: `DJI_*_MS_[G|R|RE|NIR].TIF` (Green, Red, RedEdge, NIR)
- Use case: DJI Mavic 3 Multispectral camera

## Implementing Custom Sensor Classes

To support a new sensor type, create a sensor configuration:

```python
from glint_mask_tools.image_loaders import MultiFileImageLoader
from glint_mask_tools.sensors import Sensor, Band
import glob
import os

# Step 1: Create custom ImageLoader subclass
class CustomSensorLoader(MultiFileImageLoader):
    """Loader for custom sensor with pattern: SENSOR_####_[B1|B2|B3].tif"""

    @property
    def paths(self) -> list[list[str]]:  # This method should return a list of lists of strings for multi-file loaders
        """Group related band files together"""

        pattern = os.path.join(self.img_dir, "SENSOR_*_B1.tif")
        base_files = glob.glob(pattern)

        groups = []
        for base_file in sorted(base_files):
            # Find corresponding band files
            base_name = base_file.replace("_B1.tif", "")
            band_files = [
                f"{base_name}_B1.tif",  # Blue
                f"{base_name}_B2.tif",  # Green
                f"{base_name}_B3.tif",  # Red
            ]

            # Verify all files exist
            if all(os.path.exists(f) for f in band_files):
                groups.append(band_files)

        return groups

# Step 2: Create and use the custom sensor
custom_sensor = Sensor(
    name="Custom Sensor",
    bands=[Band("Blue"), Band("Green"), Band("Red")],
    bit_depth=8,
    loader_class=CustomSensorLoader,
)
masker = custom_sensor.create_masker(
    img_dir="/path/to/custom_images",
    mask_dir="/path/to/masks",
    thresholds=[0.8, 0.8, 0.8],
    pixel_buffer=2
)
masker(max_workers=4)
```

### Custom ImageLoader Implementation

For single-file sensors with custom loading requirements:

```python
from glint_mask_tools.image_loaders import SingleFileImageLoader
import glob
import os
import numpy as np
from pathlib import Path
import tifffile

class CustomSingleFileLoader(SingleFileImageLoader):
    """Custom loader for multi-file sensor format"""
    @staticmethod
    def load_image(paths: list[str | Path]) -> np.ndarray:
        """Custom loader to build a numpy array."""
        # e.g. We override this method to load images with tifffile instead of using default PIL.Image loader
        imgs = [tifffile.imread(p) for p in paths]

        # e.g. Maybe we just want the first 3 bands in these tif files
        imgs = imgs[:, :, :3]

        return np.stack(imgs, axis=2).astype(float)

    @property
    def paths(self) -> list[str]:  # This method should return a list of strings for single-file loaders
        """Get grouped paths of imagery to load."""
        # Find base files (e.g., files ending with specific pattern)
        base_pattern = os.path.join(self.img_dir, "*_RGBA.tif")
        return glob.glob(base_pattern)
```

## Glint Detection Algorithms

The library provides two built-in algorithms:

### ThresholdAlgorithm (Default)

Simple per-band thresholding with OR logic:

```python
from glint_mask_tools.glint_algorithms import ThresholdAlgorithm

# Different threshold per band
algorithm = ThresholdAlgorithm([0.8, 0.9, 0.7, 0.85])  # B, G, R, NIR
```

### IntensityRatioAlgorithm (Advanced)

Estimates specular reflection component for RGB imagery:

```python
from glint_mask_tools.glint_algorithms import IntensityRatioAlgorithm

# Advanced glint detection for RGB
algorithm = IntensityRatioAlgorithm(
    percent_diffuse=0.95,  # Assumed diffuse reflection percentage
    threshold=0.99         # Specular component threshold
)
```

## Error Handling and Progress Tracking

```python
from glint_mask_tools.sensors import rgb_sensor

def progress_callback(completed_paths):
    """Called after each successful image processing"""
    print(f"✓ Processed {len(completed_paths)} images")

def error_callback(failed_paths, exception):
    """Called when image processing fails"""
    print(f"✗ Failed to process {failed_paths}: {str(exception)}")

masker = rgb_sensor.create_masker("/input", "/output")
masker(
    max_workers=4,
    callback=progress_callback,
    err_callback=error_callback
)
```

## API Reference

For complete API documentation, see the source code or use Python's help system:

```python
from glint_mask_tools import sensors
help(sensors.Sensor)
```

## Next Steps

- [Learn about the GUI →](gui.md)
- [Explore CLI commands →](cli.md)
- [Return to Usage Overview →](usage.md)
