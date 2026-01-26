"""JAX MPS Plugin - Metal Performance Shaders backend for JAX."""

import os
import sys
import warnings
from pathlib import Path

# jaxlib version this plugin was built against (major.minor). Used for runtime
# compatibility checking.
_BUILT_FOR_JAXLIB = (0, 9)
_LIB_NAME = "libpjrt_plugin_mps.dylib"


class MPSPluginError(Exception):
    """Exception raised when MPS plugin initialization fails."""

    pass


def _get_search_paths():
    """Return list of (path, description) tuples for library search."""
    pkg_dir = Path(__file__).parent
    project_root = pkg_dir.parent.parent.parent

    return [
        (pkg_dir / _LIB_NAME, "package directory (editable install)"),
        (pkg_dir / "lib" / _LIB_NAME, "package lib/ (wheel install)"),
        (
            project_root / "build" / "*" / "lib" / _LIB_NAME,
            "build/*/lib/ (cmake build)",
        ),
        (Path("/usr/local/lib") / _LIB_NAME, "/usr/local/lib/"),
        (Path("/opt/homebrew/lib") / _LIB_NAME, "/opt/homebrew/lib/"),
    ]


def _find_library():
    """Find the pjrt_plugin_mps shared library.

    Returns:
        Path to the library, or None if not found.
    """
    # Environment variable takes precedence
    if "JAX_MPS_LIBRARY_PATH" in os.environ:
        env_path = os.environ["JAX_MPS_LIBRARY_PATH"]
        if Path(env_path).exists():
            return env_path
        raise MPSPluginError(
            f"JAX_MPS_LIBRARY_PATH is set to '{env_path}', but the file does not exist."
        )

    for path, _ in _get_search_paths():
        # Handle glob patterns
        if "*" in str(path):
            for match in Path("/").glob(str(path).lstrip("/")):
                if match.exists():
                    return str(match)
        elif path.exists():
            return str(path)

    return None


def _check_jaxlib_version():
    """Check if the installed jaxlib version is compatible.

    Warns if the major.minor version doesn't match what the plugin was built for.
    """
    try:
        import jaxlib

        version_str = getattr(jaxlib, "__version__", None)
        if version_str is None:
            return

        parts = version_str.split(".")
        if len(parts) < 2:
            return

        installed = (int(parts[0]), int(parts[1]))
        if installed != _BUILT_FOR_JAXLIB:
            warnings.warn(
                f"jax-mps was built for jaxlib {_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.x, "
                f"but jaxlib {version_str} is installed. This may cause compatibility "
                f"issues with StableHLO bytecode parsing. Consider installing jaxlib "
                f">={_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1]}.0,"
                f"<{_BUILT_FOR_JAXLIB[0]}.{_BUILT_FOR_JAXLIB[1] + 1}",
                stacklevel=3,
            )
    except Exception:
        pass  # Don't fail initialization due to version check issues


def initialize():
    """Initialize the MPS plugin with JAX.

    This function is called by JAX's plugin discovery mechanism.

    Raises:
        MPSPluginError: If Metal GPU is not available or plugin initialization fails.
    """
    # Check platform first
    if sys.platform != "darwin":
        raise MPSPluginError(
            f"MPS plugin requires macOS, but running on {sys.platform}. MPS (Metal "
            "Performance Shaders) is only available on Apple devices."
        )

    # Check jaxlib version compatibility
    _check_jaxlib_version()

    library_path = _find_library()
    if library_path is None:
        searched = "\n".join(f"  - {desc}" for _, desc in _get_search_paths())
        raise MPSPluginError(
            f"Could not find {_LIB_NAME}. Searched paths:\n{searched}\n"
            "You can also set JAX_MPS_LIBRARY_PATH environment variable."
        )

    # Disable shardy partitioner - it produces sdy dialect ops that our StableHLO parser
    # doesn't support yet (JAX 0.9+ enables it by default)
    try:
        import jax

        jax.config.update("jax_use_shardy_partitioner", False)
    except Exception as e:
        warnings.warn(
            f"Failed to disable shardy partitioner: {e}. Some operations may not work correctly.",
            stacklevel=2,
        )

    # Register the plugin using JAX's xla_bridge API
    try:
        from jax._src import xla_bridge as xb
    except ImportError as e:
        raise MPSPluginError(f"Failed to import JAX xla_bridge: {e}") from e

    if not hasattr(xb, "register_plugin"):
        raise MPSPluginError("JAX version does not support register_plugin API.")

    try:
        xb.register_plugin(
            "mps",
            priority=500,  # Higher than CPU (0) but lower than GPU (1000)
            library_path=library_path,
            options=None,
        )
    except Exception as e:
        # Handle "already registered" case - this is fine, not an error
        if "ALREADY_EXISTS" in str(e) and "mps" in str(e).lower():
            return
        raise MPSPluginError(f"Failed to register MPS plugin with JAX: {e}") from e
