"""System information for bug reports."""

import platform
import subprocess

from .util import get_package_version


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return "unknown"


def _get_memory_gb():
    try:
        mem_bytes = int(_run(["sysctl", "-n", "hw.memsize"]))
        return f"{mem_bytes // (1024**3)} GB"
    except (ValueError, TypeError):
        return "unknown"


def _get_default_backend():
    try:
        import jax

        return jax.default_backend()
    except Exception:
        return "unknown"


def get_info():
    """Return system information as a dict."""
    build = _run(["sw_vers", "-buildVersion"])
    macos_ver = platform.mac_ver()[0]
    macos = f"{macos_ver} ({build})" if build != "unknown" else macos_ver

    return {
        "jax": get_package_version("jax") or "not installed",
        "jaxlib": get_package_version("jaxlib") or "not installed",
        "jax-mps": get_package_version("jax-mps") or "not installed",
        "Python": platform.python_version(),
        "macOS": macos,
        "Chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "Memory": _get_memory_gb(),
        "Backend": _get_default_backend(),
    }


def format_info():
    """Return formatted system information string."""
    info = get_info()
    width = max(len(k) for k in info)
    return "\n".join(f"{k + ':':<{width + 1}} {v}" for k, v in info.items())


if __name__ == "__main__":
    print(format_info())
