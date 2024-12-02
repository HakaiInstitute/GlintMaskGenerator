# Glint Mask Generator

Glint Mask Generator is a utility for generating mask images that correspond to the area of spectral glint in some
source imagery. Once generated, the masks can used by 3rd party structure-from-motion (SfM) software to replace glint
areas with data from overlapping imagery that are not glinty.

<div style="margin-top: 50px; overflow: hidden; display: flex; justify-content:center; gap:10px;">
    <img alt="Glint" src="./images/glint.gif" width="80%" />
</div>


## Features

* Support for single and multi-file sensors.
  * RGB, PhaseOne 4-band CIRs, MicaSense RedEdge, and Phantom 4 MS images currently supported.
* Pixel buffering around glint areas.
* Parallel processing.

## License

GlintMaskGenerator is released under
the [MIT license](https://raw.githubusercontent.com/tayden/GlintMaskGenerator/main/LICENSE.txt).

## Contribute

Please file a bug report using our
[GitHub Issue Tracker :material-github:](https://github.com/HakaiInstitute/GlintMaskGenerator/issues) if
you encounter any problems or would like to help add additional functionality.

<div style="margin-top: 50px; overflow: hidden; display: flex; justify-content:center; gap:10px;">
    <img alt="Hakai" src="./images/hakai_logo.png" width="30%" />
</div>
