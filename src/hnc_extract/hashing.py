"""sha256 of files, shared by weight checks and provenance records."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 23):
            h.update(chunk)
    return h.hexdigest()
