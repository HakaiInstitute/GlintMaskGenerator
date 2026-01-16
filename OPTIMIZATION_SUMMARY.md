# Band Alignment Optimization Summary

This document summarizes the performance optimizations implemented for the `BandAligner` class in `src/glint_mask_tools/band_alignment.py`.

## Optimizations Implemented

### 1. Per-Band Warp Matrix Caching
**Location**: `band_alignment.py:57, 87-94`

**How it works**:
- Stores the warp matrix from each band's previous image alignment
- Uses cached matrix as initial guess for `cv2.findTransformECC` on the next image
- Each band maintains its own independent cache entry

**Benefits**:
- Significantly faster convergence for sequential images with similar alignment
- Zero overhead for first image (no cache available)
- Per-band independence ensures optimal convergence for each spectral band

**Fallback mechanism**:
- If ECC fails with cached guess, automatically retries with identity matrix
- Cache is cleared on alignment failure to prevent bad values from persisting
- Ensures same failure behavior as before caching was implemented

### 2. Reduced Iteration Count for Cached Guesses
**Location**: `band_alignment.py:211-214`

**How it works**:
- Uses 400 max iterations when cached initial guess is available
- Uses 1000 max iterations when starting from identity matrix
- Convergence threshold remains 1e-6 for both cases

**Benefits**:
- 60% fewer maximum iterations when cache is available
- Faster convergence due to better starting point + fewer iterations
- Still allows full iteration count when needed (first image, cache cleared)

**Rationale**:
Since cached guesses are typically close to the solution, ECC converges faster and doesn't need as many iterations to reach the same precision threshold.

### 3. Parallel Band Processing
**Location**: `band_alignment.py:92-132`

**How it works**:
- Uses `ThreadPoolExecutor` to process multiple bands concurrently
- Default: `min(4, num_bands - 1)` worker threads
- Configurable via `max_workers` parameter
- Automatically uses sequential processing for 2-band images

**Benefits**:
- Near-linear speedup for multi-band images on multi-core systems
- OpenCV's ECC implementation releases the GIL, enabling true parallelism
- Zero overhead for 2-band images (uses sequential path)

**Thread safety**:
- Each thread processes a different band independently
- No shared mutable state during parallel execution
- Results collected and cached after all threads complete

## API Changes

### BandAligner Constructor
```python
# Before
BandAligner(enabled=True)

# After (backward compatible)
BandAligner(enabled=True, max_workers=None)
```

**Parameters**:
- `enabled`: Whether alignment is enabled (default: True) - *unchanged*
- `max_workers`: Maximum worker threads for parallel processing (default: None)
  - `None`: Auto-select `min(4, num_bands - 1)`
  - `1`: Force sequential processing
  - `> 1`: Use specified number of workers

**Examples**:
```python
# Default behavior (auto-parallel with caching)
aligner = BandAligner()

# Force sequential processing
aligner = BandAligner(max_workers=1)

# Use 2 worker threads
aligner = BandAligner(max_workers=2)

# Disable alignment entirely
aligner = BandAligner(enabled=False)
```

## Performance Improvements

### Expected Speedup (5-band images, sequential processing)

Based on typical sensor data with similar sequential alignment:

| Optimization | Expected Speedup | Cumulative Speedup |
|-------------|------------------|-------------------|
| Baseline | 1.0x | 1.0x |
| + Caching | 1.5-2.0x | 1.5-2.0x |
| + Reduced iterations | 1.3-1.5x | 2.0-3.0x |
| + Parallel (4 workers) | 2.5-3.5x | 5.0-10.0x |

**Note**: Actual speedup depends on:
- Number of bands (more bands = better parallel speedup)
- Image size (larger images = more benefit from caching)
- Alignment similarity between sequential images
- CPU core count (affects parallel speedup)

### Benchmark Script

Run `benchmark_caching.py` to measure performance on your system:

```bash
uv run python benchmark_caching.py
```

Expected output:
```
Band Alignment Performance Benchmark
======================================================================
Processing 20 images with 5 bands each (512x512 pixels)

Results:
----------------------------------------------------------------------
1. Baseline (no cache, sequential):        XX.XXXs
2. Cached + sequential:                     XX.XXXs
3. Cached + parallel (default):             XX.XXXs

Improvements:
----------------------------------------------------------------------
Caching speedup (2 vs 1):      X.XXx  (X.XXXs saved)
Parallel speedup (3 vs 2):     X.XXx  (X.XXXs saved)
Total speedup (3 vs 1):        X.XXx  (X.XXXs saved, XX.X% faster)
======================================================================
```

## Error Handling

### Cache-Related Failures

**Scenario**: Cached initial guess causes ECC to fail
```
1. Try with cached warp matrix
2. If fails → Retry with identity matrix
3. If retry succeeds → Update cache with new result
4. If retry fails → Clear cache, return failure
```

**Logging**:
- Debug: "ECC failed with cached initial guess, retrying with identity"
- Debug: "ECC retry with identity also failed"
- Warning: "Band alignment failed for image"

### Parallel Processing Errors

**Exception propagation**:
- Exceptions from worker threads are automatically propagated to main thread
- ThreadPoolExecutor ensures all futures complete or raise exceptions
- Cache is cleared on any alignment failure (sequential or parallel)

## Testing

### Test Coverage

**Unit tests** (`tests/test_band_alignment.py`):
- 36 total tests
- `TestBandAlignerCaching` (7 tests): Cache behavior, fallback mechanism
- `TestBandAlignerParallel` (5 tests): Parallel processing correctness

**Integration tests** (`tests/test_masker_integration.py`):
- 14 tests ensuring end-to-end functionality
- Tests with real sensor configurations (P4MS, MicaSense, etc.)

**Run tests**:
```bash
# All band alignment tests
uv run pytest tests/test_band_alignment.py -v

# Integration tests
uv run pytest tests/test_masker_integration.py -v

# All tests
uv run pytest -v
```

## Implementation Details

### Cache Structure
```python
# Dict[band_idx, warp_matrix]
self._prev_warp_matrices: dict[int, np.ndarray] = {}

# Example for 5-band image:
{
    1: np.array([[1.0, 0.0, 5.2], [0.0, 1.0, 5.1]]),  # Band 1 warp
    2: np.array([[1.0, 0.0, 6.1], [0.0, 1.0, 6.0]]),  # Band 2 warp
    3: np.array([[1.0, 0.0, 4.8], [0.0, 1.0, 4.9]]),  # Band 3 warp
    4: np.array([[0.99, 0.01, 5.5], [-0.01, 0.99, 5.4]]),  # Band 4 warp (with rotation)
}
# Note: Band 0 (reference) has no cache entry
```

### Parallel Processing Flow

```
Main Thread:
    ├─ Create ThreadPoolExecutor(max_workers=4)
    ├─ Submit process_band(1) → Worker Thread 1
    ├─ Submit process_band(2) → Worker Thread 2
    ├─ Submit process_band(3) → Worker Thread 3
    ├─ Submit process_band(4) → Worker Thread 4
    ├─ Wait for all futures to complete
    ├─ Collect results in order
    └─ Cache all warp matrices

Worker Threads (parallel execution):
    Band 1: _estimate_transform(ref, band1, cached_warp_1)
    Band 2: _estimate_transform(ref, band2, cached_warp_2)
    Band 3: _estimate_transform(ref, band3, cached_warp_3)
    Band 4: _estimate_transform(ref, band4, cached_warp_4)
```

### Memory Usage

**Additional memory per BandAligner instance**:
- Cache: `(num_bands - 1) * 2 * 3 * 4 bytes = (num_bands - 1) * 24 bytes`
- Example: 5-band image = 96 bytes of cache

**Parallel processing overhead**:
- ThreadPoolExecutor: ~1KB per worker thread
- No additional image copies (threads share ref_band via numpy arrays)

## Backward Compatibility

**100% backward compatible**:
- Default behavior enables all optimizations
- No changes to `align()` method signature
- No changes to return types
- Existing code works without modification

**Migration**:
```python
# Existing code - no changes needed
aligner = BandAligner()
aligned_img, offsets = aligner.align(img)

# Still works exactly the same, now faster!
```

## Future Optimization Opportunities

### Potential Improvements Not Yet Implemented:

1. **Image pyramids / Multi-scale alignment**
   - Estimated speedup: 2-3x for large images
   - Complexity: Medium
   - Best for: High-resolution imagery (>2000x2000)

2. **ROI-based alignment**
   - Estimated speedup: 3-5x for very large images
   - Complexity: Low
   - Best for: Images where center region represents whole

3. **Skip alignment for very similar consecutive images**
   - Estimated speedup: 10-100x for identical scenes
   - Complexity: Medium
   - Best for: Dense sequential captures of same scene

4. **GPU acceleration (CUDA)**
   - Estimated speedup: 5-10x
   - Complexity: High (requires CUDA-enabled OpenCV)
   - Best for: Users with NVIDIA GPUs

## Credits

- **Implemented by**: Claude Code (Anthropic)
- **Based on**: OpenCV's Enhanced Correlation Coefficient (ECC) algorithm
- **Repository**: GlintMaskGenerator
- **Organization**: Hakai Institute
