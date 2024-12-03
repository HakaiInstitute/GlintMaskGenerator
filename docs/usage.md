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
        - MicasenseRedEdge: Micasense RedEdge multispectral camera imagery

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

For information about the parameters expected by the CLI, run glint-mask --help in a bash terminal or command line
interface. All the functionality of the CLI is documented there.

```
❯ glint-mask --help

 Usage: glint-mask [OPTIONS] COMMAND [ARGS]...                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                    
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                       │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                │
│ --help                        Show this message and exit.                                                                                                     │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ rgb-threshold         Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.                                                     │
│ cir-threshold         Generate masks for glint regions in 4 Band CIR imagery using Tom Bell's binning algorithm.                                              │
│ p4ms-threshold        Generate masks for glint regions in multispectral imagery from the DJI camera using Tom Bell's algorithm on the Blue image band.        │
│ micasense-threshold   Generate masks for glint regions in multispectral imagery from the Micasense camera using Tom Bell's algorithm on the blue image band.  │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
# Get addition parameters for one of the cameras/methods available
❯ glint-mask rgb-threshold --help
                                                                                                                                                                                                                                                                    
 Usage: glint-mask rgb-threshold [OPTIONS] IMG_DIR OUT_DIR                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                    
 Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.                                                                                                                                                                                
                                                                                                                                                                                                                                                                    
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    img_dir      PATH  The path to a named input image or directory containing images. If img_dir is a directory, all tif, jpg, jpeg, and png images in that directory will be processed. [default: None] [required]  │
│ *    out_dir      PATH  The path to send your out image including the file name and type. e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is specified as a directory. [default: None] [required]     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --thresholds          FLOAT    The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0). [default: 0.875, 1, 1, 1, 1]                                                                               │
│ --pixel-buffer        INTEGER  The pixel distance to buffer out the mask. [default: 0]                                                                                                                                 │
│ --max-workers         INTEGER  The maximum number of threads to use for processing. [default: 4]                                                                                                                       │
│ --help                         Show this message and exit.                                                                                                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

#### CLI Examples

```bash
# Process RGB imagery directory with default parameters
# - Uses default thresholds (Blue: 0.875, Green: 1.0, Red: 1.0)
# - No pixel buffer
# - 4 worker threads
glint-mask rgb-threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process PhaseONE CIR imagery with custom settings
# - Specify custom thresholds with --thresholds
# - Add 2-pixel buffer with --pixel-buffer
# - Use 8 worker threads with --max-workers
glint-mask cir-threshold \
    --thresholds 0.8,0.9,0.9,0.9 \
    --pixel-buffer 2 \
    --max-workers 8 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/

# Process DJI P4MS imagery with minimal masking
# - Higher thresholds mean less aggressive masking
# - Useful for scenes with minimal glint
glint-mask p4ms-threshold \
    --thresholds 0.95,1.0,1.0,1.0,1.0 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/

# Process Micasense RedEdge imagery with aggressive masking
# - Lower thresholds mean more aggressive masking
# - Larger pixel buffer for broader masked areas
glint-mask micasense-threshold \
    --thresholds 0.8,0.9,0.9,0.9,0.9 \
    --pixel-buffer 5 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

### Python Library

Installing the PyPi package allows integrating the mask generation workflow into existing python scripts with ease.

```py
from glint_mask_generator import MicasenseRedEdgeThresholdMasker

# Also available: P4MSThresholdMasker, RGBIntensityRatioMasker, RGBThresholdMasker

masker = MicasenseRedEdgeThresholdMasker(
    img_dir="path/to/micasense/images/",
    mask_dir="path/to/output/dir/",
    thresholds=(0.875, 1, 1, 1, 1),
    pixel_buffer=5
)
masker.process(max_workers=5, callback=print, err_callback=print)
```
