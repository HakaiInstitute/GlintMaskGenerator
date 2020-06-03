# Glint Mask Tools
![Python application](https://github.com/HakaiInstitute/glint-mask-tools/workflows/Python%20application/badge.svg?branch=master)


## Description 
Generate masks for glint regions in RGB images.

## Installation
1. You must have the Anaconda package manager installed. Do that. [Instructions here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install the necessary packages listed in environment.yml file. 
    - See the [Anaconda documentation here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for installing from an environment.yml file. 
    - You can most likely just do `conda env create -f environment.yml` with the environment.yml file provided in this repository.

## Methods
#### glint_mask_RGB.py
This is the method based on Matlab code written by Tom Bell at UC Santa Barbara from 2019-06-28.

This method bins pixel values based on the blue band component of the image and then divides very blue parts as being "glinty".
The found mask is padded using a Gaussian filter.

#### specular_mask.py
This is the method based on the method outlined in:\
Wang, S., Yu, C., Sun, Y. et al. "Specular reflection removal of ocean surface remote sensing images from UAVs". Multimed Tools Appl 77, 11363–11379 (2018). https://doi.org/10.1007/s11042-017-5551-7

This method estimates the component of pixel-wise specular reflectance. Following this, a threshold on this estimate can partition the image into the "glinty" and "non-glinty" parts.
The calculated mask is padded with Morphological Opening operations.

## Running the code
`glint_mask_RGB.py` and `specular_mask.py` can both be run as a command line script or incorporated into other Python scripts for more advanced use cases.

### Running from the Terminal
1. Activate the created conda environment: `conda activate glint`
2. Navigate to the directory this file is located in the terminal and run it to process a single file.
3. Run `python glint_mask_RGB.py --help` for a description of the required parameters for this method.
3. Run `python specular_mask.py --help` for a description of the required parameters for this method.

##### Examples
```bash
# Process a single file
python glint_mask_RGB.py /path/to/in_file.jpg /path/to/out_mask.jpg --glint_threshold 0.5

python specular_mask.py /path/to/in_file.jpg /path/to/out_mask.jpg --percent_diffuse 0.2 --mask_thresh 0.5


# Process a directory of files
python glint_mask_RGB.py /path/to/dir/with/images/ /path/to/out_masks/dir/ --glint_threshold 0.5

python specular_mask.py /path/to/dir/with/images/ /path/to/out_masks/dir/ --percent_diffuse 0.2 --mask_thresh 0.5
```

### Running from a Python script
1. Import the glint mask function into your Python script
    - `from glint_mask_RGB import make_mask`
    - or `from specular_mask import make_mask`
2. run `help(make_mask)` for parameter details, or inspect the source code, which is well-documented.
    
### Notes
#### Single image processing
- The [supported file formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) are listed here for processing individual files.

#### Directory of images processing
- All files with "jpg", "jpeg", and "png" extensions will be processed. This can be extended as needed if you ask Taylor to do it.
- Output mask files with be in the specified directory, and have the same name as the input file with "_mask" appended to the end of the file name stem. The file type will match the input type.


---
*Created by Taylor Denouden @ Hakai Institute on 2020-05-22*