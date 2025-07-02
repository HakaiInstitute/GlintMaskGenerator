# Usage

## General usage notes

### Accepted file types

- Supported file types are currently .jpg, .jpeg, .tif, .tiff, and .png (all are case-insensitive).

### Output files:

- Saved in the specified output directory
- Named as original filename + "_mask" suffix and maintain the same file type as the input file
    - Example: `image1.jpg` → `image1_mask.jpg`
- When processing multi-band imagery (e.g., Micasense RedEdge or P4MS), masks will be generated for all sibling band
  images.
    - This caters to the expectations of SfM software like Agisoft Metashape.

### Understanding Pixel Thresholds

Pixel thresholds determine how the software identifies glint in your imagery. The thresholds are specified as decimal
values between 0.0 and 1.0, which are then applied to the full range of possible pixel values in your image.

#### How Thresholds Work

For example, in an 8-bit image (pixel values 0-255):

- A threshold of 0.5 means: pixel value > (0.5 × 255) = 127
- A threshold of 0.875 means: pixel value > (0.875 × 255) = 223

#### Multiple Band Behavior

When multiple bands are present (like in RGB images):

1. Each band is checked against its respective threshold
2. If ANY band exceeds its threshold, that pixel is marked as glint
3. The resulting masks are combined using a union operation

#### Example

For an RGB image with thresholds:

- Blue: 0.875 (triggers at values > 223)
- Green: 1.000 (never triggers)
- Red: 1.000 (never triggers)

A pixel will be marked as glint if its blue value exceeds 223, regardless of its red and green values.

## Interfaces

### GUI

The GUI version provides an intuitive interface for generating glint masks from imagery. Launch the application by
double-clicking the executable file on Windows, or running `./GlintMaskGenerator` from the terminal on Linux.

#### Main Options

1. **Imagery Type Selection**
    - Choose the appropriate camera/sensor type for your imagery:
        - RGB: Standard RGB camera imagery
        - CIR: 4-band Color Infrared imagery
        - P4MS: DJI Phantom 4 Multispectral camera imagery
        - M3M: DJI Mavic 3 Multispectral camera imagery
        - MicaSense RedEdge: MicaSense RedEdge multispectral camera imagery

2. **Directory Selection**
    - Image Directory: Select the input folder containing your imagery files using the "..." button
    - Output Directory: Choose where the generated mask files will be saved

3. **Band Thresholds**
    - Adjust thresholds for each available band using the sliders
    - Range: 0.0 to 1.0 (higher values = less masking)
        - Default values:
            - Blue: 0.875
            - Green: 1.000
            - Red: 1.000
            - Red Edge: 1.000 (when applicable)
            - NIR: 1.000 (when applicable)
        - Use the "Reset all" button to restore default values

4. **Processing Options**
    - Pixel Buffer Radius: Adjusts the expansion of masked regions (default: 0)
    - Max Workers: Controls the number of parallel processing threads (default: 4)

5. **Processing**
    - Click "Run" to start generating masks

### CLI

For information about the parameters expected by the CLI, run `glint-mask --help` in a bash terminal or command line
interface. All the functionality of the CLI is documented there.

```
❯ glint-mask --help

 Usage: glint-mask [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.      │
│ --show-completion             Show completion for the current shell, to copy │
│                               it or customize the installation.              │
│ --help                        Show this message and exit.                    │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────╮
│ rgb    Generate glint masks for RGB sensors using threshold algorithm.       │
│ cir    Generate glint masks for PhaseOne 4-band CIR sensors using threshold  │
│        algorithm.                                                            │
│ p4ms   Generate glint masks for DJI P4MS sensors using threshold algorithm.  │
│ m3m    Generate glint masks for DJI M3M sensors using threshold algorithm.   │
│ msre   Generate glint masks for MicaSense RedEdge sensors using threshold    │
│        algorithm.                                                            │
╰──────────────────────────────────────────────────────────────────────────────╯
```

```
# Get additional parameters for one of the cameras/methods available
❯ glint-mask rgb --help

 Usage: glint-mask rgb [OPTIONS] IMG_DIR OUT_DIR

 Generate glint masks for RGB sensors using threshold algorithm.

╭─ Arguments ──────────────────────────────────────────────────────────────────╮
│ *    img_dir      PATH  The path to a named input image or directory         │
│                         containing images. If img_dir is a directory, all    │
│                         tif, jpg, jpeg, and png images in that directory     │
│                         will be processed.                                   │
│                         [default: None]                                      │
│                         [required]                                           │
│ *    out_dir      PATH  The path to send your out image including the file   │
│                         name and type. e.g. "/path/to/mask.png". The out_dir │
│                         must be a directory if img_dir is specified as a     │
│                         directory.                                           │
│                         [default: None]                                      │
│                         [required]                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────╮
│ --thresholds          FLOAT    The pixel band thresholds indicating glint.   │
│                                Domain for values is (0.0, 1.0).              │
│                                [default: None]                               │
│ --pixel-buffer        INTEGER  The pixel distance to buffer out the mask.    │
│                                [default: 0]                                  │
│ --max-workers         INTEGER  The maximum number of threads to use for      │
│                                processing.                                   │
│                                [default: 4]                                  │
│ --help                         Show this message and exit.                   │
╰──────────────────────────────────────────────────────────────────────────────╯
```

#### CLI Examples

```bash
# Process RGB imagery directory with default parameters
# - Uses default thresholds (Red: 1.0, Green: 1.0, Blue: 0.875)
# - No pixel buffer
# - 4 worker threads
glint-mask rgb /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process PhaseONE CIR imagery with custom settings
# - Specify custom thresholds with --thresholds
# - Add 2-pixel buffer with --pixel-buffer
# - Use 8 worker threads with --max-workers
glint-mask cir \
    --thresholds 0.8 0.9 0.9 0.9 \
    --pixel-buffer 2 \
    --max-workers 8 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/

# Process DJI P4MS imagery with minimal masking
# - Higher thresholds mean less aggressive masking
# - Useful for scenes with minimal glint
glint-mask p4ms \
    --thresholds 0.95 1.0 1.0 1.0 1.0 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/

# Process DJI Mavic 3 Multispectral imagery
# - 4-band sensor (Green, Red, Red Edge, NIR)
glint-mask m3m \
    --thresholds 0.875 1.0 1.0 1.0 \
    --pixel-buffer 2 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/

# Process MicaSense RedEdge imagery with aggressive masking
# - Lower thresholds mean more aggressive masking
# - Larger pixel buffer for broader masked areas
glint-mask msre \
    --thresholds 0.8 0.9 0.9 0.9 0.9 \
    --pixel-buffer 5 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

### Python Library

Installing the PyPI package allows integrating the mask generation workflow into existing Python scripts with ease.

## Python Usage Guide

This library provides a flexible Python API for generating glint masks from various sensor types. The architecture consists of sensor configurations, image loaders, glint detection algorithms, and processing maskers.

### Quick Start

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

### Supported Sensor Types

The library includes built-in support for multiple sensor types:

| Sensor | CLI Command | Bands | Default Thresholds | File Pattern |
|--------|-------------|-------|-------------------|--------------|
| RGB | `rgb` | Red, Green, Blue | [1.0, 1.0, 0.875] | Single file |
| CIR | `cir` | Red, Green, Blue, Near-IR | [1.0, 1.0, 0.875, 1.0] | Single file |
| DJI P4MS | `p4ms` | Blue, Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0, 1.0] | `DJI_###[1-5].TIF` |
| DJI M3M | `m3m` | Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0] | `DJI_*_MS_[G\|R\|RE\|NIR].TIF` |
| MicaSense RedEdge | `msre` | Blue, Green, Red, RedEdge, Near-IR | [0.875, 1.0, 1.0, 1.0, 1.0] | `IMG_####_[1-5].tif` |

### Using Sensor Configurations

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

# Color Infrared (large images with tiling)
masker = cir_sensor.create_masker(
    img_dir="/path/to/cir_images",
    mask_dir="/path/to/masks"
)
```

### Advanced Usage with Components

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

### Image Loader Classes

The library uses a hierarchical image loader system:

#### Single-File Loaders

**`SingleFileImageLoader`**: Standard 8-bit RGB imagery in one file
- Supports: JPG, PNG, TIFF formats
- Use case: Standard photography, drone RGB cameras

**`BigTiffLoader`**: 8-bit Color Infrared imagery in BigTIFF format
- Features: Memory-efficient tiled processing (256x256 tiles)
- Use case: Large 4-band CIR images that don't fit in memory

#### Multi-File Loaders

**`MicasenseRedEdgeLoader`**: 16-bit multi-file MicaSense format
- Pattern: `IMG_####_[1-5].tif` (Blue, Green, Red, RedEdge, NIR)
- Use case: MicaSense RedEdge cameras

**`P4MSLoader`**: 16-bit multi-file DJI P4 Multispectral format
- Pattern: `DJI_###[1-5].TIF` (Blue, Green, Red, RedEdge, NIR)
- Use case: DJI Phantom 4 Multispectral camera

**`DJIM3MLoader`**: 16-bit multi-file DJI Mavic 3M format
- Pattern: `DJI_*_MS_[G|R|RE|NIR].TIF` (Green, Red, RedEdge, NIR)
- Use case: DJI Mavic 3 Multispectral camera

### Implementing Custom Sensor Classes

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

#### For Multi-File Sensors

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

### Glint Detection Algorithms

The library provides two built-in algorithms:

#### ThresholdAlgorithm (Default)
Simple per-band thresholding with OR logic:
```python
from glint_mask_tools.glint_algorithms import ThresholdAlgorithm

# Different threshold per band
algorithm = ThresholdAlgorithm([0.8, 0.9, 0.7, 0.85])  # B, G, R, NIR
```

#### IntensityRatioAlgorithm (Advanced)
Estimates specular reflection component for RGB imagery:
```python
from glint_mask_tools.glint_algorithms import IntensityRatioAlgorithm

# Advanced glint detection for RGB
algorithm = IntensityRatioAlgorithm(
    percent_diffuse=0.95,  # Assumed diffuse reflection percentage
    threshold=0.99         # Specular component threshold
)
```

### Error Handling and Progress Tracking

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
