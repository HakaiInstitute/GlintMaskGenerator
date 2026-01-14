# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Package Management
- **Install dependencies**: `uv sync` or `pip install -e .`
- **Install dev dependencies**: `uv sync --group dev`

### Testing
- **Run all tests**: `pytest` or `uv run pytest`
- **Run specific test file**: `pytest tests/test_masker_integration.py`
- **Run tests with coverage**: `pytest --cov=src/glint_mask_tools`

### Code Quality
- **Lint code**: `ruff check src/ tests/`
- **Format code**: `ruff format src/ tests/`
- **Check all linting rules**: `ruff check --select ALL src/ tests/`

### CLI Usage
- **Test CLI**: `uv run glint-mask --help`
- **Run CLI commands**: `uv run glint-mask <sensor-type> <img_dir> <out_dir>`
- **Available sensors**: `rgb`, `cir`, `p4ms`, `m3m`, `msre`

### GUI Development
- **Run GUI**: `uv run python -m gui`
- **Edit UI files**: Use Qt Designer to modify `.ui` files in `src/gui/resources/`

## Architecture Overview

### Core Components
The codebase follows an object-oriented design with three main abstractions:

1. **ImageLoader** (`src/glint_mask_tools/image_loaders.py`): Handles loading and parsing imagery for different sensor types
2. **GlintAlgorithm** (`src/glint_mask_tools/glint_algorithms.py`): Implements glint detection algorithms
3. **Masker** (`src/glint_mask_tools/maskers.py`): Orchestrates the image loading and algorithm execution

### Sensor Configuration System
The `sensors.py` module defines sensor configurations that specify:
- Band definitions (R, G, B, NIR, RE) with default thresholds
- Bit depth and preprocessing logic
- Associated ImageLoader class
- CLI command names

New sensors are added by creating a `Sensor` instance and adding it to `_known_sensors`.

### Dynamic CLI Generation
The CLI (`src/glint_mask_tools/cli.py`) dynamically generates subcommands for each sensor using the `_known_sensors` configuration. Each sensor gets its own CLI command with appropriate parameters.

### GUI Architecture
The GUI (`src/gui/`) uses PyQt6 with:
- UI files in `src/gui/resources/` (designed with Qt Designer)
- Custom widgets in `src/gui/widgets/` for reusable components
- Threading support for long-running masking operations

### Data Processing Flow
1. **Input**: Directory of images or single image file
2. **Loading**: Appropriate ImageLoader parses sensor-specific file formats
3. **Preprocessing**: Images normalized based on bit depth
4. **Algorithm**: Threshold-based glint detection on specified bands
5. **Post-processing**: Optional pixel buffer expansion
6. **Output**: Binary mask images saved to specified directory

## Key Design Patterns

### Extensibility
- **New sensors**: Extend `ImageLoader` class and add sensor configuration
- **New algorithms**: Extend `GlintAlgorithm` class and modify masker creation
- **New preprocessing**: Override `preprocess_image` in sensor configuration

### Error Handling
- Uses callback-based error reporting in concurrent processing
- Comprehensive logging with loguru
- GUI shows progress and error messages

### Threading
- CLI supports configurable worker threads (`max_workers` parameter)
- GUI uses QThreadPool for non-blocking operations
- Progress tracking via callback functions

## Testing Strategy
- **Unit tests**: Individual component testing in `tests/`
- **Integration tests**: End-to-end workflow testing
- **Test data**: Sample imagery in `data/` directory organized by sensor type
- **Coverage**: Tests cover core algorithms, image loading, and masker integration

## Project Structure Notes
- `src/glint_mask_tools/`: Core library code
- `src/gui/`: PyQt6 GUI application
- `tests/`: Test suite
- `data/`: Sample imagery organized by sensor type
- `docs/`: Documentation and images for README

## Development Workflow
1. Changes should maintain backward compatibility in CLI interface
2. New sensors require both CLI and GUI integration
3. UI changes need Qt Designer for `.ui` files
4. Version updates are handled automatically via git tags and GitHub Actions
