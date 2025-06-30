# Glint Mask Generator

Glint Mask Generator is a utility for generating mask images that correspond to the area of spectral glint in some
source imagery. Once generated, the masks can be used by 3rd party structure-from-motion (SfM) software to replace glint
areas with data from overlapping imagery that are not glinty.

## Quick Start

1. Download the [latest release](https://github.com/HakaiInstitute/glint-mask-tools/releases) for your platform
2. Launch the GUI application
3. Select your imagery type and input directory
4. Click Run to generate masks

<div style="margin-top: 50px; overflow: hidden; display: flex; justify-content:center; gap:10px;">
    <img alt="Glint" src="./images/glint.gif" width="80%" />
</div>

## Features

### Multiple interfaces to suit your workflow:
  - User-friendly GUI for interactive use
  - CLI for automation and scripting
  - Python package for integration into existing pipelines

### Support for various camera types:
  - RGB cameras
  - PhaseOne 4-band CIRs
  - MicaSense RedEdge
  - DJI Phantom 4 MS

### Advanced processing capabilities:
  - Configurable band thresholds
  - Pixel buffering around glint areas
  - Parallel processing for improved performance
  - Batch processing of multiple images

## System Requirements

- Operating Systems:
    - Windows 10 or later
    - Linux (modern distributions)
- Python 3.9 - 3.12 (for CLI/Python package)
- Minimum 4GB RAM
- Storage space: 100MB + space for your imagery

## Interface Comparison

| Feature                  | GUI              | CLI                | Python Package   |
|--------------------------|------------------|--------------------|------------------|
| Ease of Use              | ★★★★★            | ★★★                | ★★★              |
| Automation Support       | ★                | ★★★★★              | ★★★★★            |
| Integration Capabilities | ★                | ★★★★               | ★★★★★            |
| Customization            | ★                | ★★★                | ★★★★★            |
| Learning Curve           | Minimal          | Moderate           | Moderate         |
| Best For                 | Individual users | Automation/Scripts | Custom workflows |

## Next Steps

- [How it Works](how_it_works.md) - How glint masking works
- [Installation Guide](installation.md) - Get started with installation
- [Usage Guide](usage.md) - Learn how to use the tool
- [FAQs](faq.md) - Answers to common questions

## License

GlintMaskGenerator is released under
the [MIT license](https://raw.githubusercontent.com/tayden/GlintMaskGenerator/main/LICENSE.txt).

## Contribute

We welcome contributions! Please file bug reports, feature requests, or propose improvements using
our [GitHub Issue Tracker :material-github:](https://github.com/HakaiInstitute/GlintMaskGenerator/issues).

<div style="margin-top: 50px; overflow: hidden; display: flex; justify-content:center; gap:10px;">
    <img alt="Hakai" src="./images/hakai_logo.png" width="30%" />
</div>
