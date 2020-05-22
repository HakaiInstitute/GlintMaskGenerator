# Glint Mask RGB

## Description 
Mask Glint in RGB Images. Based on Matlab script by Tom Bell written 6/28/2019

## Installation
1. You must have the Anaconda package manager installed. Do that. [Instructions here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Install the necessary packages listed in environment.yml file. 
    - See the [Anaconda documentation here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for installing from an environment.yml file. 
    - You can most likely just do `conda env create -f environment.yml` with the environment.yml file provided in this repository.

## Running the code
The glint mask code can be run as either a command line script or incorporated into another Python script (i.e. for more complex use cases).

### Terminal/Command line script
1. Activate the created conda environment: `conda activate glint`
2. Navigate to the directory this file is located in the terminal and run it to process a single file.
    - `python glint_mask_RGB.py --help` will list the required parameters.

#### Example: Process a single file
```python
python glint_mask_RGB.py /path/to/in_file.jpg /path/to/out_mask.jpg --glint_threshold 0.5
```
The [supported formats](https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html) are listed here.

#### Example: Process a directory of files
  ```python
python glint_mask_RGB.py /path/to/dir/with/images/ /path/to/out_masks/dir/ --glint_threshold 0.5
```
By default, the output files with have "_mask" appended to the file basename and be output to the specified directory.
All "jpg", "jpeg", and "png" files will be processed.
If you specify a directory as the input path, the mask_out_path must also be a directory.


### As a Python function
1. Import the glint mask function into your Python script
    - `from glint_mask_rgb import make_mask`
    - run `help(make_mask)` for parameter details, or inspect the source code file, which is well-documented.
  

*Created by Taylor Denouden @ Hakai Institute (taylor.denouden@hakai.org) on 2020-05-22*
 
*Based on Matlab code by Tom Bell @ UCSB on 2019-06-28*