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

# NAME
#     glint-mask-v*.*.*
#
# SYNOPSIS
#     glint-mask-v*.*.* - COMMAND | VALUE
#
# COMMANDS
#     COMMAND is one of the following:
#
#      cir_threshold
#        Generate masks for glint regions in 4 Band CIR imagery using Tom Bell's binning algorithm.
#
#      micasense_threshold
#        Generate masks for glint regions in multispectral imagery from the Micasense camera using Tom Bell's algorithm on the blue image band.
#
#      p4ms_threshold
#        Generate masks for glint regions in multispectral imagery from the DJI camera using Tom Bell's algorithm on the Blue image band.
#
#      process
#
#      rgb_threshold
#        Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.
#
# VALUES
#     VALUE is one of the following:
#
#      max_workers
#        The maximum number of threads to use for processing.
```

```bash
# Get addition parameters for one of the cameras/methods available
glint-mask-v*.*.* rgb_threshold --help

# NAME
#     glint-mask-v*.*.* rgb_threshold - Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.
#
# SYNOPSIS
#     glint-mask-v*.*.* rgb_threshold IMG_DIR OUT_DIR <flags>
#
# DESCRIPTION
#     Generate masks for glint regions in RGB imagery using Tom Bell's binning algorithm.
#
# POSITIONAL ARGUMENTS
#     IMG_DIR
#         The path to a named input image or directory containing images. If img_dir is a directory, all tif, jpg, jpeg, and png images in that directory will be # processed.
#     OUT_DIR
#         The path to send your out image including the file name and type. e.g. "/path/to/mask.png". out_dir must be a directory if img_dir is specified as a # # # directory.
#
# FLAGS
#     --thresholds=THRESHOLDS
#         The pixel band thresholds indicating glint. Domain for values is (0.0, 1.0). Default is [1, 1, 0.875].
#     --pixel_buffer=PIXEL_BUFFER
#         The pixel distance to buffer out the mask. Defaults to 0 (off).
#
# NOTES
#     You can also use flags syntax for POSITIONAL ARGUMENTS
```

### Examples

```bash
# Process rgb imagery directory with default parameters
glint-mask-v*.*.* rgb_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process PhaseONE camera imagery with image bands split over multiple files
glint-mask-v*.*.* aco_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process DJI P4MS imagery
glint-mask-v*.*.* p4ms_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/

# Process Micasense RedEdge imagery
glint-mask-v*.*.* micasense_threshold /path/to/dir/with/images/ /path/to/out_masks/dir/
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
