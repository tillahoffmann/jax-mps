"""Basic tests for jax-mps plugin."""

import os
import sys

# Ensure the plugin can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


def test_plugin_loads():
    """Test that the plugin can be loaded."""
    from jax_plugins import mps

    assert mps.__version__ == "0.1.0"


def test_library_found():
    """Test that the shared library can be found."""
    from jax_plugins.mps import _find_library

    # This will be None until we build
    # Just ensure the function works
    _find_library()


def test_hlo_parsing():
    """Test HLO text parsing (requires C++ library)."""
    # This test is a placeholder for when we have Python bindings
    # to the HLO parser
    pass


if __name__ == "__main__":
    test_plugin_loads()
    print("Basic tests passed!")
