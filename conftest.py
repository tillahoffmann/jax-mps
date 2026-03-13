"""Root pytest configuration for xdist parallelism."""

import os


def pytest_xdist_auto_num_workers(config):
    """Use all cores in CI, half locally (at least 1)."""
    n_cores = os.cpu_count() or 2
    if os.environ.get("GITHUB_ACTIONS"):
        return n_cores
    return max(1, n_cores // 2)
