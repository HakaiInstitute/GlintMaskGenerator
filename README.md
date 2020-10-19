# Glint Mask Tools
![Python application](https://github.com/HakaiInstitute/glint-mask-tools/workflows/Main/badge.svg?branch=master)

## Description 
Generate masks for glint regions in RGB and multispectral image files.

### Installation
1. Go to the [releases page](https://github.com/HakaiInstitute/glint-mask-tools/releases)
2. Download the latest release file for your operating system.
3. Extract the compressed binary files from the gzipped archive. 
4. This archive contains two files which provide different interfaces to the same glint mask generation program. 
    4. GlintMaskGenerator.exe provides the GUI interface
    4. glint-mask.exe is a command line interface and has a few more options available.
5. Copy these files wherever is convenient for you.
    5. On Windows, you'll probably want to copy the GUI interface to your Desktop.
    5. On Linux, copying glint-mask to `user/local/bin` will allow you to call the CLI from anywhere by typing `glint-mask`.

#### Pypi package
There is a PYPI package available for Python 3.8

1. `pip install glint-mask-tools` to install the tools.
2. Then, `import glint_mask_tools` in your Python script.

### Usage
#### GUI
In Windows, launch the GUI by double clicking the executable file. In Linux, you'll have to launch from the terminal `GlintMaskGenerator`.
The options presented in the GUI are self explanatory. 

For now, generating masks by passing directory paths containing images is the supported workflow.
Be sure to change the image type option when processing imagery from the Micasense or  DJI multi-spectral cameras. You 
will be notified of any processing errors via a pop-up dialog.
 
#### CLI
For information about the parameters expected by the CLI, just run `./glint-mask --help` in a bash terminal or command line interface. 
All the functionality of the CLI is documented there.

##### Examples
```bash
# Process a directory of files
glint-mask rgb /path/to/dir/with/images/ /path/to/out_masks/dir/
glint-mask micasense_rededge /path/to/dir/with/images/ /path/to/out_masks/dir/
glint-mask dji_rededge /path/to/dir/with/images/ /path/to/out_masks/dir/
glint-mask specular /path/to/dir/with/images/ /path/to/out_masks/dir/
```

#### Notes
##### Directory of images processing
- All files with "jpg", "jpeg", "tif", "tiff" and "png" extensions will be processed. This can be extended as needed. File extension matching is case insensitive.
- Output mask files with be in the specified directory, and have the same name as the input file with "_mask" appended to the end of the file name stem. The file type will match the input type.

##### Red Edge file processing
- Currently, the program looks for files that match the patterns "IMG\_\*[0-9]\_5.tif" for Micasense images and "DJI\_\*[0-9]4.TIF" for multi-spectral DJI images.
- As required, the matching logic can be updated in the "core/common.py" source code.
- More file types can be supported as needed and the GUI and CLI can be updated to provide more options.

## Development
The functions available in the executable binaries are actually just Python scripts packaged using PyInstaller. To 
update the logic, you'll need to install the required packages to run the individual gui.py and cli.py scripts found in 
the root of the Git repository.

### Installation
1. Start by cloning the code repository from GitHub.
2. You must have the Anaconda package manager installed. Do that. [Instructions here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
3. Install the necessary packages listed in environment.yml file. 
    - See the [Anaconda documentation here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#create-env-from-file) for installing from an environment.yml file. 
    - You can most likely just do `conda env create -f environment.yml` with the environment.yml file provided in this repository.

### Running the code
1. Activate the created conda environment: `conda activate glint`
2. Navigate to the directory this file is located in the terminal and run it to process a single file.
3. `python gui.py` and `python cli.py` can both be run from the command line for debugging purposes. 
    3. These two scripts are the entry point into the program.

### Project structure
- The bulk of the actual processing logic takes place in the files found in the "core/" directory. 
    - The functions in core are carefully documented with comments and function help strings. 
    - The strings at the beginning of each function is ingested by the "python-fire" package to create help strings for 
    the command line interface. Update these strings in `cli.py` to update the CLI help output.
- The "test/" directory contains some files for testing those functions found in "core". 
    - Tests can be run with pytest. First install pytest with `pip install pytest` and then run `pytest` from the terminal to run the test functions. 
    - All functions beginning with the word "test_" are taken as test functions by pytest.
- The code is written in an easily extensible object-oriented format. To support more kinds of Red Edge-based glint masking,
    all that is needed is to write a new child class of the AbstractBinMasker class located in `core/bin_maskers.py`
    - To do this most easily, copy one of the other child classes like `core/bin_maskers.py#MicasenseRedEdgeMasker` and 
        update the functions so the appropriate files are processed and the imagery is normalized by the appropriate bit depth.
    - After implementing a new Masker child class, just add it and a new drop-down option to the `gui.py` and everything should
        continue to work since all the processing logic is contained in the Masker classes.

### Updating the executable files
This is done automatically via some *Fancy-Pants GitHub Actions* logic. The workflow to trigger this action is as follows:

1. Make the code changes and commit them to GitHub.
    1. Ideally, use a Pull-Request workflow. Make a pull-request to the master branch from some other branch on GitHub.
    1. By doing this, you can see if tests pass and make changes as necessary to get them to pass before merging the updates to master.
    1. See the Actions tab on GitHub for some clues as to why the tests might not be passing.
2. Once the tests are passing, tag the master branch on your copy locally with `git checkout master && git pull && git tag va.b.c` 
    using an appropriate version number instead of `va.b.c`, e.g. `v1.2.1`. Use semantic version numbers as much as possible.
3. Push the tag to GitHub with `git push --tags`. Pushing a tag of the format `v*.*.*` triggers an action where the code is 
    checked to see tests are passing, then executables for Linux and Windows are built and uploaded to the GitHub release page.

---
*Created by Taylor Denouden @ Hakai Institute on 2020-05-22*