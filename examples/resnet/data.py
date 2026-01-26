import pickle
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

CACHE_DIR = Path.home() / ".cache" / "cifar10"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def load_cifar10() -> tuple[np.ndarray, np.ndarray]:
    """Download and load CIFAR-10 training data."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tarball_path = CACHE_DIR / "cifar-10-python.tar.gz"

    # Download if not cached.
    if not tarball_path.exists():
        print(f"Downloading CIFAR-10 to {tarball_path}...")
        urllib.request.urlretrieve(CIFAR10_URL, tarball_path)

    # Extract and load training batches.
    images, labels = [], []
    with tarfile.open(tarball_path, "r:gz") as tar:
        for i in range(1, 6):
            member = tar.extractfile(f"cifar-10-batches-py/data_batch_{i}")
            assert member is not None
            batch = pickle.load(member, encoding="bytes")
            images.append(batch[b"data"])
            labels.append(batch[b"labels"])

    # Reshape to (N, 32, 32, 3) and normalize to [0, 1].
    images = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    images = images.astype(np.float32) / 255.0
    labels = np.concatenate(labels).astype(np.int32)

    # Normalize with mean/std.
    mean = images.mean(axis=(0, 1, 2))
    std = images.std(axis=(0, 1, 2))
    images = (images - mean) / std

    return images, labels
