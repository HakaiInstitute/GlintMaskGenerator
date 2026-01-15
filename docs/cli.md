# CLI Reference

The command-line interface provides powerful automation capabilities and scripting support.

## Available Commands

For information about the parameters expected by the CLI, run `glint-mask --help` in a bash terminal or command line interface. All the functionality of the CLI is documented there.

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
│ rgb         Generate glint masks for RGB sensors using threshold algorithm.  │
│ cir         Generate glint masks for PhaseOne 4-band CIR sensors using       │
│             threshold algorithm.                                             │
│ p4ms        Generate glint masks for DJI P4MS sensors using threshold        │
│             algorithm.                                                       │
│ m3m         Generate glint masks for DJI M3M sensors using threshold         │
│             algorithm.                                                       │
│ msre        Generate glint masks for MicaSense RedEdge sensors using         │
│             threshold algorithm.                                             │
│ msre-dual   Generate glint masks for MicaSense RedEdge-MX Dual sensors using │
│             threshold algorithm.                                             │
╰──────────────────────────────────────────────────────────────────────────────╯
```

## Common Parameters

Get additional parameters for any sensor command:

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

## Examples

### RGB Imagery

```bash
# Process RGB imagery directory with default parameters
# - Uses default thresholds (Red: 1.0, Green: 1.0, Blue: 0.875)
# - No pixel buffer
# - 4 worker threads
glint-mask rgb /path/to/dir/with/images/ /path/to/out_masks/dir/
```

### PhaseOne CIR

```bash
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
```

### DJI P4MS

```bash
# Process DJI P4MS imagery with minimal masking
# - Higher thresholds mean less aggressive masking
# - Useful for scenes with minimal glint
glint-mask p4ms \
    --thresholds 0.95 1.0 1.0 1.0 1.0 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

### DJI Mavic 3M

```bash
# Process DJI Mavic 3 Multispectral imagery
# - 4-band sensor (Green, Red, Red Edge, NIR)
glint-mask m3m \
    --thresholds 0.875 1.0 1.0 1.0 \
    --pixel-buffer 2 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

### MicaSense RedEdge

```bash
# Process MicaSense RedEdge imagery with aggressive masking
# - Lower thresholds mean more aggressive masking
# - Larger pixel buffer for broader masked areas
glint-mask msre \
    --thresholds 0.8 0.9 0.9 0.9 0.9 \
    --pixel-buffer 5 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

### MicaSense RedEdge-MX Dual

```bash
# Process MicaSense RedEdge-MX Dual imagery (10-band sensor)
# - Requires 10 threshold values (one per band)
# - Supports all multi-sensor options
glint-mask msre-dual \
    --thresholds 1.0 0.875 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 \
    --pixel-buffer 2 \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

## Multi-Sensor Options

Multi-sensor systems (P4MS, M3M, MicaSense RedEdge, MicaSense RedEdge-MX Dual) support additional options:

```bash
# - --no-align: Disable automatic band alignment (enabled by default)
# - --per-band: Generate independent masks per band instead of union
glint-mask p4ms \
    --per-band \
    --no-align \
    /path/to/dir/with/images/ \
    /path/to/out_masks/dir/
```

## Tips

- **Use higher thresholds** for less aggressive masking (values closer to 1.0)
- **Use lower thresholds** for more aggressive masking (values closer to 0.0)
- **Adjust `--max-workers`** based on your CPU cores for optimal performance
- **Use `--per-band`** if you need independent masks for each spectral band
- **Use `--no-align`** only if automatic alignment causes issues with your imagery

## Shell Completion

Install shell completion for easier command usage:

```bash
glint-mask --install-completion
```

## Next Steps

- [Learn about the GUI →](gui.md)
- [Explore the Python API →](python-api.md)
- [Return to Usage Overview →](usage.md)
