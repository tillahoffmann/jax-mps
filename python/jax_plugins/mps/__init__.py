"""JAX MPS Plugin - Metal Performance Shaders backend for JAX."""

import os
from pathlib import Path


def _find_library():
    """Find the pjrt_plugin_mps shared library."""
    # Look in common locations
    search_paths = [
        # Development build
        Path(__file__).parent.parent.parent.parent / "build" / "lib",
        # Installed package
        Path(__file__).parent / "lib",
        # System paths
        Path("/usr/local/lib"),
        Path("/opt/homebrew/lib"),
    ]

    lib_name = "libpjrt_plugin_mps.dylib"

    for path in search_paths:
        lib_path = path / lib_name
        if lib_path.exists():
            return str(lib_path)

    # Check environment variable
    if "JAX_MPS_LIBRARY_PATH" in os.environ:
        return os.environ["JAX_MPS_LIBRARY_PATH"]

    return None


def initialize():
    """Initialize the MPS plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.
    """
    library_path = _find_library()
    if library_path is None:
        # Don't print warning during JAX startup - it's noisy
        return

    # Disable shardy partitioner - it produces sdy dialect ops that our
    # StableHLO parser doesn't support yet (JAX 0.9+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception:
        pass

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb

        if hasattr(xb, "register_plugin"):
            # Use the register_plugin API which handles everything
            xb.register_plugin(
                "mps",
                priority=500,  # Higher than CPU (0) but lower than GPU (1000)
                library_path=library_path,
                options=None,
            )
    except Exception:
        # Silently fail - don't spam user's console
        pass


# Version info
__version__ = "0.1.0"
