"""
Example usage of jax-mps plugin.

Before running, build the native library:
    mkdir build && cd build
    cmake ..
    make

Then run this script:
    python main.py
"""

import os
import sys

# Add python package to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

# Set library path for development
build_lib = os.path.join(os.path.dirname(__file__), "build", "lib")
if os.path.exists(build_lib):
    os.environ["JAX_MPS_LIBRARY_PATH"] = os.path.join(
        build_lib, "libpjrt_plugin_mps.dylib"
    )


def main():
    print("JAX-MPS: Metal Performance Shaders backend for JAX")
    print("=" * 50)

    # Try to import JAX
    try:
        import jax
        import jax.numpy as jnp

        print(f"JAX version: {jax.__version__}")
        print(f"Available backends: {jax.devices()}")
    except ImportError:
        print("JAX not installed. Install with: pip install jax")
        return

    # Try to initialize our plugin
    try:
        from jax_plugins import mps

        print(f"jax-mps version: {mps.__version__}")
        mps.initialize()
    except Exception as e:
        print(f"Could not initialize MPS plugin: {e}")
        print("This is expected if the native library hasn't been built yet.")
        print("\nTo build:")
        print("  mkdir build && cd build")
        print("  cmake ..")
        print("  make")
        return

    # Simple test
    print("\nRunning simple test...")
    try:
        # This will use the MPS backend if registered successfully
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([4.0, 5.0, 6.0])
        z = x + y
        print(f"x + y = {z}")

        # Matrix multiplication
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        c = jnp.matmul(a, b)
        print(f"matmul result:\n{c}")

        # Tanh
        t = jnp.tanh(x)
        print(f"tanh(x) = {t}")

        print("\nAll tests passed!")
    except Exception as e:
        print(f"Error during test: {e}")


if __name__ == "__main__":
    main()
