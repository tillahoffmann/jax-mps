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

# Set library path for development (only if the actual dylib exists)
build_lib_path = os.path.join(
    os.path.dirname(__file__), "build", "lib", "libpjrt_plugin_mps.dylib"
)
if os.path.exists(build_lib_path):
    os.environ["JAX_MPS_LIBRARY_PATH"] = build_lib_path


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
        sys.exit("ERROR: JAX not installed. Install with: pip install jax")

    # Try to initialize our plugin
    try:
        from jax_plugins import mps

        print(f"jax-mps version: {mps.__version__}")
        mps.initialize()
    except Exception as e:
        sys.exit(
            f"ERROR: Could not initialize MPS plugin: {e}\n\n"
            "To build the native library:\n"
            "  mkdir build && cd build\n"
            "  cmake ..\n"
            "  make"
        )

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
        sys.exit(f"ERROR: Test failed: {e}")


if __name__ == "__main__":
    main()
