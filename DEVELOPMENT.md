# Development
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