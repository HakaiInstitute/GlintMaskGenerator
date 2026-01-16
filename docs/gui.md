# GUI Guide

The GUI version provides an intuitive interface for generating glint masks from imagery. Launch the application by double-clicking the executable file on Windows, or running `./GlintMaskGenerator` from the terminal on Linux.

<div style="margin-top: 20px; margin-bottom: 20px; overflow: hidden; display: flex; justify-content:center; gap:10px;">
    <img alt="GUI Screenshot" src="/images/app_screenshot.png" width="80%" />
</div>

## Main Options

### 1. Imagery Type Selection

Choose the appropriate camera/sensor type for your imagery:

- **RGB**: Standard RGB camera imagery
- **CIR**: 4-band Color Infrared imagery
- **P4MS**: DJI Phantom 4 Multispectral camera imagery
- **M3M**: DJI Mavic 3 Multispectral camera imagery
- **MicaSense RedEdge**: MicaSense RedEdge multispectral camera imagery
- **MicaSense RedEdge-MX Dual**: MicaSense RedEdge-MX Dual 10-band multispectral camera imagery

### 2. Directory Selection

- **Image Directory**: Select the input folder containing your imagery files using the "..." button
- **Output Directory**: Choose where the generated mask files will be saved

### 3. Band Thresholds

Adjust thresholds for each available band using the sliders:

- **Range**: 0.0 to 1.0 (higher values = less masking)
- **Default values**:
    - Blue: 0.875
    - Green: 1.000
    - Red: 1.000
    - Red Edge: 1.000 (when applicable)
    - NIR: 1.000 (when applicable)
- Use the **Reset all** button to restore default values

### 4. Processing Options

- **Pixel Buffer Radius**: Adjusts the expansion of masked regions (default: 0)
- **Max Workers**: Controls the number of parallel processing threads (default: 4)

### 5. Multi-Sensor Options

The following options are only available for multi-sensor systems (P4MS, M3M, MicaSense RedEdge). Single-sensor systems (RGB, CIR) capture all bands in a single file that is already aligned, and can only output a single mask file per image.

- **Auto-align bands before masking** (default: enabled): Automatically aligns bands from different sensors using phase correlation before applying thresholds. Each image is aligned independently to compensate for slight positional differences between sensors in multi-camera systems.

- **Mask each band independently**: When enabled, each band's mask contains only that band's threshold detection instead of the union of all bands. By default (disabled), all band masks are combined so that a pixel marked as glint in any band is masked in all output files. Enable this option if you need per-band masks that reflect only that specific band's glint detection.

### 6. Processing

Click **Run** to start generating masks. The progress bar will show completion status.

## Tips and Best Practices

- Start with default thresholds and adjust based on your results
- Use pixel buffer to expand masked regions if glint detection is too conservative
- Enable auto-align for multi-sensor imagery (this is the default and recommended setting)
- Increase max workers for faster batch processing on machines with more CPU cores
- Review a few masked images before processing your entire dataset

## Troubleshooting

If you encounter issues:

- Ensure your imagery files are in supported formats (.jpg, .jpeg, .tif, .tiff, .png)
- Check that you have sufficient disk space for the output masks
- Reduce max workers if you experience memory issues
- See the [FAQs](faq.md) for common questions and solutions

## Next Steps

- [Learn about CLI usage →](cli.md)
- [Explore the Python API →](python-api.md)
- [Read the FAQs →](faq.md)
