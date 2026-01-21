#!/usr/bin/env python3
"""Benchmark comparing JAX CPU vs MPS (Apple Silicon GPU) backends."""

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

    # Force CPU backend
    cpu_device = jax.devices("cpu")[0]

    # Try to get MPS backend
    try:
        mps_device = jax.devices("mps")[0]
        has_mps = True
        print(f"CPU Device: {cpu_device}")
        print(f"MPS Device: {mps_device}")
    except Exception as e:
        has_mps = False
        print(f"CPU Device: {cpu_device}")
        print(f"MPS Device: Not available ({e})")

    print()
    print("-" * 70)

    # Define benchmark sizes
    sizes = [
        ("Small (100x100)", 100),
        ("Medium (1000x1000)", 1000),
        ("Large (4000x4000)", 4000),
    ]

    results = []

    for name, size in sizes:
        print(f"\n{name} matrices ({size}x{size}):")
        print("-" * 50)

        # Generate random data
        np.random.seed(42)
        a_np = np.random.randn(size, size).astype(np.float32)
        b_np = np.random.randn(size, size).astype(np.float32)

        # === Matrix Multiplication ===
        print("\n  Matrix Multiplication (matmul):")

        # CPU benchmark
        a_cpu = jax.device_put(a_np, cpu_device)
        b_cpu = jax.device_put(b_np, cpu_device)

        @jax.jit
        def matmul_cpu(a, b):
            return jnp.matmul(a, b)

        cpu_mean, cpu_std, cpu_result = benchmark_fn(matmul_cpu, a_cpu, b_cpu)
        print(f"    CPU:  {cpu_mean * 1000:8.2f} ms ± {cpu_std * 1000:.2f} ms")

        if has_mps:
            a_mps = jax.device_put(a_np, mps_device)
            b_mps = jax.device_put(b_np, mps_device)

            @jax.jit
            def matmul_mps(a, b):
                return jnp.matmul(a, b)

            mps_mean, mps_std, mps_result = benchmark_fn(matmul_mps, a_mps, b_mps)
            print(f"    MPS:  {mps_mean * 1000:8.2f} ms ± {mps_std * 1000:.2f} ms")

            speedup = cpu_mean / mps_mean if mps_mean > 0 else 0
            print(f"    Speedup: {speedup:.2f}x")

            results.append(("matmul", name, cpu_mean, mps_mean, speedup))

        # === Element-wise Addition ===
        print("\n  Element-wise Addition (add):")

        @jax.jit
        def add_cpu(a, b):
            return a + b

        cpu_mean, cpu_std, _ = benchmark_fn(add_cpu, a_cpu, b_cpu)
        print(f"    CPU:  {cpu_mean * 1000:8.2f} ms ± {cpu_std * 1000:.2f} ms")

        if has_mps:

            @jax.jit
            def add_mps(a, b):
                return a + b

            mps_mean, mps_std, _ = benchmark_fn(add_mps, a_mps, b_mps)
            print(f"    MPS:  {mps_mean * 1000:8.2f} ms ± {mps_std * 1000:.2f} ms")

            speedup = cpu_mean / mps_mean if mps_mean > 0 else 0
            print(f"    Speedup: {speedup:.2f}x")

            results.append(("add", name, cpu_mean, mps_mean, speedup))

        # === Tanh ===
        print("\n  Tanh activation:")

        @jax.jit
        def tanh_cpu(a):
            return jnp.tanh(a)

        cpu_mean, cpu_std, _ = benchmark_fn(tanh_cpu, a_cpu)
        print(f"    CPU:  {cpu_mean * 1000:8.2f} ms ± {cpu_std * 1000:.2f} ms")

        if has_mps:

            @jax.jit
            def tanh_mps(a):
                return jnp.tanh(a)

            mps_mean, mps_std, _ = benchmark_fn(tanh_mps, a_mps)
            print(f"    MPS:  {mps_mean * 1000:8.2f} ms ± {mps_std * 1000:.2f} ms")

            speedup = cpu_mean / mps_mean if mps_mean > 0 else 0
            print(f"    Speedup: {speedup:.2f}x")

            results.append(("tanh", name, cpu_mean, mps_mean, speedup))

    # Summary
    if has_mps and results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(
            f"\n{'Operation':<15} {'Size':<20} {'CPU (ms)':<12} {'MPS (ms)':<12} {'Speedup':<10}"
        )
        print("-" * 70)
        for op, size_name, cpu_t, mps_t, speedup in results:
            print(
                f"{op:<15} {size_name:<20} {cpu_t * 1000:<12.2f} {mps_t * 1000:<12.2f} {speedup:<10.2f}x"
            )

        avg_speedup = np.mean([r[4] for r in results])
        print("-" * 70)
        print(f"Average speedup: {avg_speedup:.2f}x")

    print()


if __name__ == "__main__":
    run_benchmarks()
