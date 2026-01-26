"""Micro-benchmark comparing JAX CPU vs MPS (Apple Silicon GPU) backends."""

import itertools
import time

import numpy as np

# Benchmark configuration
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10


def benchmark_fn(fn, *args, warmup=WARMUP_RUNS, runs=BENCHMARK_RUNS):
    """Benchmark a function, returning mean and std of execution times."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Benchmark
    times = []
    result = None
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times), result


def run_benchmarks():
    print("=" * 70)
    print("JAX Benchmark: CPU vs MPS (Apple Silicon GPU)")
    print("=" * 70)
    print()

    # Import JAX for CPU
    import jax
    import jax.numpy as jnp

    # Get the devices.
    cpu_device = jax.devices("cpu")[0]
    mps_device = jax.devices("mps")[0]
    devices = [cpu_device, mps_device]

    # Define benchmark sizes
    sizes = [
        ("Small (100x100)", 100),
        ("Medium (1000x1000)", 1000),
        ("Large (4000x4000)", 4000),
    ]

    results = {}

    for (name, size), device in itertools.product(sizes, devices):
        print(f"\n{name} matrices ({size}x{size}):")
        print("-" * 50)

        # Generate random data
        np.random.seed(42)
        a = jax.device_put(np.random.randn(size, size).astype(np.float32), device)
        b = jax.device_put(np.random.randn(size, size).astype(np.float32), device)

        # === Matrix Multiplication ===
        print("\n  Matrix Multiplication (matmul):")

        @jax.jit
        def matmul(a, b):
            return jnp.matmul(a, b)

        mean, std, _ = benchmark_fn(matmul, a, b)
        print(f"    {device.platform}:  {mean * 1000:8.2f} ms ± {std * 1000:.2f} ms")
        results.setdefault(("matmul", name), []).append((device.platform, mean))

        # === Element-wise Addition ===
        print("\n  Element-wise Addition (add):")

        @jax.jit
        def add(a, b):
            return a + b

        mean, std, _ = benchmark_fn(add, a, b)
        print(f"    {device.platform}:  {mean * 1000:8.2f} ms ± {std * 1000:.2f} ms")
        results.setdefault(("add", name), []).append((device.platform, mean))

        # === Tanh ===
        print("\n  Tanh activation:")

        @jax.jit
        def tanh(a):
            return jnp.tanh(a)

        mean, std, _ = benchmark_fn(tanh, a)
        print(f"    {device.platform}:  {mean * 1000:8.2f} ms ± {std * 1000:.2f} ms")
        results.setdefault(("tanh", name), []).append((device.platform, mean))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(
        f"\n{'Operation':<15} {'Size':<20} {'CPU (ms)':<12} {'MPS (ms)':<12} {'Speedup':<10}"
    )
    print("-" * 70)
    for (op, size_name), rows in results.items():
        assert len(rows) == 2
        (platform_cpu, mean_cpu) = rows[0]
        assert platform_cpu == "cpu"
        (platform_mps, mean_mps) = rows[1]
        assert platform_mps == "mps"
        speedup = mean_cpu / mean_mps
        print(
            f"{op:<15} {size_name:<20} {mean_cpu * 1000:<12.2f} {mean_mps * 1000:<12.2f} {speedup:<10.2f}x"
        )


if __name__ == "__main__":
    run_benchmarks()
