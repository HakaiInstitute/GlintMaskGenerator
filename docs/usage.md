# Usage

## GUI

In Windows, launch the GUI by double clicking the executable file. In Linux, you'll have to launch the GUI from the
terminal, e.g. `./GlintMaskGenerator`.

For now, generating masks by passing directory paths containing images is the supported workflow. Be sure to change the
image type option when processing imagery for cameras other than RGB cameras (e.g. Micasense RedEdge or DJI P4MS
cameras). You will be notified of any
processing errors via a pop-up dialog.

## CLI

For information about the parameters expected by the CLI, run glint-mask --help in a bash terminal or command line
interface. All the functionality of the CLI is documented there.

```bash
glint-mask --help

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

```bash
# Get addition parameters for one of the cameras/methods available
glint-mask rgb_threshold --help
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

### Examples

```bash
# Process rgb imagery directory with default parameters
glint-mask rgb_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process PhaseONE camera imagery with image bands split over multiple files
glint-mask aco_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process DJI P4MS imagery
glint-mask p4ms_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process Micasense RedEdge imagery
glint-mask micasense_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/
```

## Python Package

Installing the PyPi package allows integrating the mask generation workflow into existing python scripts with ease.

```python
from glint_mask_generator import MicasenseRedEdgeThresholdMasker

# Also available: P4MSThresholdMasker, RGBIntensityRatioMasker, RGBThresholdMasker

masker = MicasenseRedEdgeThresholdMasker(img_dir="path/to/micasense/images/", mask_dir="path/to/output/dir/",
                                         thresholds=(0.875, 1, 1, 1, 1), pixel_buffer=5)
masker.process(max_workers=5, callback=print, err_callback=print)
```

## Notes

### Directory of images processing

- All files with "jpg", "jpeg", "tif", "tiff" and "png" extensions will be processed. This can be extended as needed.
  File extension matching is case-insensitive.
- Output mask files with be in the specified directory, and have the same name as the input file with "_mask" appended
  to the end of the file name stem. The file type will match the input type.

### Multi-band image processing

- For imagery types where each band is spread over multiple files, a mask will be generated for all the sibling band
  images.
    - For example, if a mask is generated using a threshold on the blue band image, identical masks are saved for
      sibling red, green, blue, nir, and red_edge bands as well.
    - If thresholds are passed for multiple bands, these mask outputs combined with a union operator before being saved
      for all the sibling bands associated with that capture event.
