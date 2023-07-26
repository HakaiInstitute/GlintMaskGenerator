# Development
The functions available in the executable binaries on GitHub are Python scripts packaged using PyInstaller. The PyPi package and CLI
installation are handled by Poetry, which also handles the package dependencies. Annoyingly, on Windows, the required Rasterio and GDAL
packages are not installable via pip, so pre-compiled binaries are included in the extern folder to allow packaging up the Windows GUI.
To install `glint-mask-tools` PyPi package on Windows, users will have to first install Rasterio by following [their instructions](https://rasterio.readthedocs.io/en/latest/installation.html#id2).

### Installation
1. Clone the code repository from [GitHub](https://github.com/HakaiInstitute/GlintMaskGenerator).
2. You must have the Poetry package manager installed. Do that. [Instructions here](https://python-poetry.org/)
3. Create a Poetry environment and install the packages listed in `pyproject.toml` file of the GlintMaskGenerator repo.
   1.Run `poetry install`.

### Running the code
1. To run any code in poetry, you need to specify that the command is meant to run in the poetry virtual environment for this project.
   1. e.g. Run `poetry run python src/gui.py` to execute the GUI script.

### Project structure
- The bulk of the actual processing logic takes place in the files found in the "src/glint_mask_tools" directory.
    - The functions in this directory are carefully documented with comments and function help strings.
    - The strings at the beginning of each function is ingested by the "python-fire" package to create help strings for
    the command line interface. Updating these strings in `src/glint_mask_tools/cli.py` will update the CLI help output.
- The "tests/" directory contains some files for testing those functions found in "src/glint_mask_tools".
    - Tests can be run with pytest. Run `poetry run pytest` from the terminal to run the test functions.
    - All functions beginning with the word "test_" are taken as test functions by pytest.
- The code is written in using object-oriented principles. So, e.g. to support a new kind of Red Edge-based glint masking,
    what is needed is to write a new child class of the `ImageLoader` class located in `src/glint_mask_tools/image_loaders.py`,
    and then use this new loader class in the declaration of a new `Masker` class in `src/glint_mask_tools/maskers.py`.
    - After implementing a new Masker child class, add it to the GUI using QTDesigner to edit the `src/resources/gui.ui` file.
    - The logic for how the GUI executes the library code is located in `src/gui.py`, and will also need to be updated to use the new `Masker`.

### Updating the executable files
This is done automatically via GitHub Actions scripts. The workflow to trigger this action is as follows:

1. Make the code changes and commit them to GitHub.
    1. Ideally, use a Pull-Request workflow. Make a pull-request to the master branch from some other branch on GitHub.
    2. This will run the tests.
2. Once the tests are passing, tag the main branch on your copy locally with `git checkout main && git pull && git tag v*.*.*`
    using an appropriate semantic version number. The `pyproject.toml` is updated with this version by the actions script at build time.
3. Push the tag to GitHub with `git push --tags`. Pushing a tag of the format `v*.*.*` triggers an action where the code is
    checked to see tests are passing, then executables for Linux and Windows are built and uploaded to the GitHub release page.
4. Similarly, the PyPi package will be built and published using Poetry commands defined by the `pypi-publish.yml`
    GitHub Actions definition file.

---
*Created by Taylor Denouden @ Hakai Institute on 2022-11-30*
