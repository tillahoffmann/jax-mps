"""Root pytest configuration for xdist parallelism."""

import os


def pytest_xdist_auto_num_workers(config):
    """Use half the available cores (at least 1) for parallel test execution."""
    n_cores = os.cpu_count() or 2
    return max(1, n_cores // 2)
