"""Benchmark script to demonstrate performance improvements in band alignment.

This script compares alignment performance across different optimization strategies:
1. Baseline (no caching, sequential)
2. With caching only (sequential)
3. With caching + reduced iterations
4. With caching + reduced iterations + parallel processing
"""

import time

import numpy as np

from src.glint_mask_tools.band_alignment import BandAligner


def create_test_image(offset_x: int, offset_y: int, num_bands: int = 5) -> np.ndarray:
    """Create a synthetic multi-band image with known band offsets."""
    img = np.zeros((512, 512, num_bands), dtype=np.uint8)

    # Reference band (band 0)
    img[100:400, 100:400, 0] = 255

    # Add shifted versions for other bands with slight variation
    for band_idx in range(1, num_bands):
        shift_x = offset_x + band_idx
        shift_y = offset_y + band_idx
        img[100 + shift_y : 400 + shift_y, 100 + shift_x : 400 + shift_x, band_idx] = 255

    return img


def benchmark_baseline(num_images: int = 20, num_bands: int = 5) -> float:
    """Benchmark without caching (create new aligner each time) and sequential processing."""
    start_time = time.perf_counter()

    for i in range(num_images):
        # Create a fresh aligner each time to avoid caching
        aligner = BandAligner(enabled=True, max_workers=1)
        offset = 5 + i % 3
        img = create_test_image(offset, offset, num_bands=num_bands)
        _, _ = aligner.align(img)

    end_time = time.perf_counter()
    return end_time - start_time


def benchmark_with_caching_sequential(num_images: int = 20, num_bands: int = 5) -> float:
    """Benchmark with caching enabled, sequential processing."""
    aligner = BandAligner(enabled=True, max_workers=1)

    start_time = time.perf_counter()

    for i in range(num_images):
        offset = 5 + i % 3
        img = create_test_image(offset, offset, num_bands=num_bands)
        _, _ = aligner.align(img)

    end_time = time.perf_counter()
    return end_time - start_time


def benchmark_with_caching_parallel(num_images: int = 20, num_bands: int = 5, max_workers: int = 4) -> float:
    """Benchmark with caching and parallel processing (default behavior)."""
    aligner = BandAligner(enabled=True, max_workers=max_workers)

    start_time = time.perf_counter()

    for i in range(num_images):
        offset = 5 + i % 3
        img = create_test_image(offset, offset, num_bands=num_bands)
        _, _ = aligner.align(img)

    end_time = time.perf_counter()
    return end_time - start_time


def main():
    """Run benchmarks and report results."""
    num_images = 20
    num_bands = 5

    print("Band Alignment Performance Benchmark")
    print("=" * 70)
    print(f"Processing {num_images} images with {num_bands} bands each (512x512 pixels)")
    print()

    # Warm up
    print("Warming up...")
    _ = benchmark_baseline(num_images=3, num_bands=num_bands)
    _ = benchmark_with_caching_sequential(num_images=3, num_bands=num_bands)
    _ = benchmark_with_caching_parallel(num_images=3, num_bands=num_bands)

    print("\nRunning benchmarks...")

    # Run actual benchmarks
    time_baseline = benchmark_baseline(num_images=num_images, num_bands=num_bands)
    time_cached_sequential = benchmark_with_caching_sequential(num_images=num_images, num_bands=num_bands)
    time_cached_parallel = benchmark_with_caching_parallel(num_images=num_images, num_bands=num_bands)

    print()
    print("Results:")
    print("-" * 70)
    print(f"1. Baseline (no cache, sequential):        {time_baseline:.3f}s")
    print(f"2. Cached + sequential:                     {time_cached_sequential:.3f}s")
    print(f"3. Cached + parallel (default):             {time_cached_parallel:.3f}s")
    print()
    print("Improvements:")
    print("-" * 70)

    # Caching improvement
    speedup_caching = time_baseline / time_cached_sequential
    time_saved_caching = time_baseline - time_cached_sequential
    print(f"Caching speedup (2 vs 1):      {speedup_caching:.2f}x  ({time_saved_caching:.3f}s saved)")

    # Parallel improvement over cached sequential
    speedup_parallel = time_cached_sequential / time_cached_parallel
    time_saved_parallel = time_cached_sequential - time_cached_parallel
    print(f"Parallel speedup (3 vs 2):     {speedup_parallel:.2f}x  ({time_saved_parallel:.3f}s saved)")

    # Total improvement
    speedup_total = time_baseline / time_cached_parallel
    time_saved_total = time_baseline - time_cached_parallel
    percent_faster = (speedup_total - 1) * 100
    print(
        f"Total speedup (3 vs 1):        {speedup_total:.2f}x  ({time_saved_total:.3f}s saved, {percent_faster:.1f}% faster)"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
