# Usage Guide

Glint Mask Generator provides three interfaces to suit different workflows:

- **[GUI Guide](gui.md)** - User-friendly interface for interactive use
- **[CLI Reference](cli.md)** - Command-line interface for automation and scripting
- **[Python API](python-api.md)** - Python library for integration into existing pipelines

## General Usage Notes

### Accepted File Types

- Supported file types are currently .jpg, .jpeg, .tif, .tiff, and .png (all are case-insensitive).

### Output Files

- Saved in the specified output directory
- Named as original filename + "_mask" suffix and maintain the same file type as the input file
    - Example: `image1.jpg` → `image1_mask.jpg`
- When processing multi-band imagery (e.g., Micasense RedEdge or P4MS), masks will be generated for all sibling band images.
    - This caters to the expectations of SfM software like Agisoft Metashape.

### Understanding Pixel Thresholds

Pixel thresholds determine how the software identifies glint in your imagery. The thresholds are specified as decimal values between 0.0 and 1.0, which are then applied to the full range of possible pixel values in your image.

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

## Choosing an Interface

| Feature                  | GUI              | CLI                | Python Package   |
|--------------------------|------------------|--------------------|------------------|
| Ease of Use              | ★★★★★            | ★★★                | ★★★              |
| Automation Support       | ★                | ★★★★★              | ★★★★★            |
| Integration Capabilities | ★                | ★★★★               | ★★★★★            |
| Customization            | ★                | ★★★                | ★★★★★            |
| Learning Curve           | Minimal          | Moderate           | Moderate         |
| Best For                 | Individual users | Automation/Scripts | Custom workflows |

## Next Steps

Choose your interface and get started:

- **[Get started with the GUI →](gui.md)** - Interactive interface with visual controls
- **[Learn CLI commands →](cli.md)** - Command-line automation and scripting
- **[Integrate with Python →](python-api.md)** - Build custom processing pipelines

Need help? Check out the [FAQs](faq.md) or visit the [Issue Tracker](https://github.com/HakaiInstitute/GlintMaskGenerator/issues).
