"""Shared utilities for the MPS plugin."""

from importlib.metadata import PackageNotFoundError, version


def get_package_version(package):
    """Return the installed version string for a package, or None if not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return None
