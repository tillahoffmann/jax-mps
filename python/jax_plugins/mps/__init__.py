"""JAX MPS Plugin - Metal Performance Shaders backend for JAX."""

import os
import sys
import warnings
from pathlib import Path


class MPSPluginError(Exception):
    """Exception raised when MPS plugin initialization fails."""

    pass


def _find_library():
    """Find the pjrt_plugin_mps shared library.

    Returns:
        Path to the library, or None if not found.
    """
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
        env_path = os.environ["JAX_MPS_LIBRARY_PATH"]
        if Path(env_path).exists():
            return env_path
        # Environment variable set but path doesn't exist
        raise MPSPluginError(
            f"JAX_MPS_LIBRARY_PATH is set to '{env_path}' but the file does not exist."
        )

    return None


def initialize():
    """Initialize the MPS plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.

    Raises:
        MPSPluginError: If Metal GPU is not available or plugin initialization fails.
    """
    # Check platform first
    if sys.platform != "darwin":
        raise MPSPluginError(
            f"MPS plugin requires macOS, but running on {sys.platform}. "
            "MPS (Metal Performance Shaders) is only available on Apple devices."
        )

    library_path = _find_library()
    if library_path is None:
        raise MPSPluginError(
            "Could not find libpjrt_plugin_mps.dylib. "
            "Searched paths:\n"
            "  - build/lib/ (development)\n"
            "  - python/jax_plugins/mps/lib/ (installed)\n"
            "  - /usr/local/lib/\n"
            "  - /opt/homebrew/lib/\n"
            "You can also set JAX_MPS_LIBRARY_PATH environment variable."
        )

    # Disable shardy partitioner - it produces sdy dialect ops that our
    # StableHLO parser doesn't support yet (JAX 0.9+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception as e:
        warnings.warn(
            f"Failed to disable shardy partitioner: {e}. "
            "Some operations may not work correctly.",
            stacklevel=2,
        )

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb

        if not hasattr(xb, "register_plugin"):
            raise MPSPluginError(
                "JAX version does not support register_plugin API. "
                "Please upgrade JAX to version 0.4.20 or later."
            )

        xb.register_plugin(
            "mps",
            priority=500,  # Higher than CPU (0) but lower than GPU (1000)
            library_path=library_path,
            options=None,
        )
    except MPSPluginError:
        raise
    except ImportError as e:
        raise MPSPluginError(f"Failed to import JAX xla_bridge: {e}") from e
    except Exception as e:
        # Handle "already registered" case - this is fine, not an error
        if "ALREADY_EXISTS" in str(e) and "mps" in str(e).lower():
            return  # Plugin already registered, nothing to do
        raise MPSPluginError(f"Failed to register MPS plugin with JAX: {e}") from e


# Version info
__version__ = "0.1.0"
